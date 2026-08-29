"""Tests for F10 store registration + F13 forecast wiring of the SO model
(CRM_PDT_AZ). Offline: InMemoryScoreStore + monkeypatched live fetchers (no sklearn)."""

import pandas as pd
import pytest

from predictive import schema as S
from predictive import ingest as I
from predictive import so_realization as R
from predictive.store import InMemoryScoreStore


def _row(opp, p, model_id, amount=1e6, so_plan="2026-06-15"):
    return {"opp_id": opp, "win_prob": p, "band": S.band(p), "drivers": [],
            "amount": amount, "status": "Open", "so_plan_date": so_plan,
            "at_risk": False, "scored_at": "2026-06-06", "model_run_id": "r",
            "model_id": model_id}


def test_so_model_registered():
    assert S.normalize_model_id("CRM_PDT_AZ") == "CRM_PDT_AZ"   # F10: not coerced away
    assert S.feature_columns("CRM_PDT_AZ") == S.feature_columns("CRM_PDT_BASE")  # shares OPP contract


def test_so_scores_coexist_with_other_models():
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("A", 0.90, "CRM_PDT_BASE"),
        _row("A", 0.55, "CRM_PDT_AZ"),
    ])
    assert s.get_deal_score("A", model_id="CRM_PDT_BASE")["win_prob"] == 0.90
    assert s.get_deal_score("A", model_id="CRM_PDT_AZ")["win_prob"] == 0.55
    so_only = {r["opp_id"] for r in s.get_latest_scores(model_id="CRM_PDT_AZ")}
    assert so_only == {"A"}


def test_run_so_conversion_forecast_assembles_from_store(monkeypatch):
    """F13 wiring: reads CRM_PDT_AZ conversion probs from the store, joins live
    realization + plan lines + actuals → a monthly/yearly forecast. No training."""
    store = InMemoryScoreStore()
    store.upsert_scores([
        _row("OPP1", 0.80, "CRM_PDT_AZ"),
        _row("OPP2", 0.50, "CRM_PDT_AZ"),
        _row("OPP1", 0.99, "CRM_PDT_BASE"),   # other model must be ignored
    ])

    realization = pd.DataFrame([
        {"opp_id": "x", "plan_ym": "2025-04", "plan_amount": 100.0,
         "actual_ym": "2025-04", "actual_amount": 50.0},   # resolved 2025, x converted → cond rate 0.5
    ])
    plan = pd.DataFrame([
        {"opp_id": "OPP1", "ym": "2026-06", "amount": 100.0},
        {"opp_id": "OPP2", "ym": "2026-07", "amount": 40.0},
    ])
    # SO ledger → converted_opps for the conditional realization rate (incl line opp "x")
    ledger = pd.DataFrame([{"opp_id": "x", "so_count": 1, "so_total": 50.0,
                            "so_invoiced": 50.0, "so_first_date": "2025-04-10"}])
    # run_so_conversion_forecast does `from .ingest import ...` at call time → patch ingest
    monkeypatch.setattr(I, "fetch_so_conversions", lambda dataset_id=None: ledger)
    monkeypatch.setattr(I, "fetch_incomeplan_realization", lambda dataset_id=None: realization)
    monkeypatch.setattr(I, "fetch_income_plan_so_lines", lambda year, dataset_id=None: plan)
    monkeypatch.setattr(I, "fetch_so_actual_by_month", lambda year, dataset_id=None: {"2026-03": 30.0})

    out = R.run_so_conversion_forecast(store, asof="2026-06-10")
    assert out["model_id"] == "CRM_PDT_AZ"
    assert out["n_scored_opps"] == 2                       # only CRM_PDT_AZ scores
    assert out["realization_rate"] == pytest.approx(0.5)
    m = {r["month"]: r for r in out["months"]}
    assert m["2026-03"]["expected"] == 30.0                # elapsed actual
    # June = 100 * 0.80 (conv) * 0.5 (rate) = 40
    assert m["2026-06"]["expected"] == pytest.approx(40.0)
    # July = 40 * 0.50 * 0.5 = 10
    assert m["2026-07"]["expected"] == pytest.approx(10.0)
    assert out["yearly_expected"] == pytest.approx(30.0 + 40.0 + 10.0)
