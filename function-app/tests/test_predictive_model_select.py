"""Tests for two-model selection (CRM_PDT_BASE / CRM_PDT_MIX):

  - the model_id dimension on the score store (both models coexist per opp);
  - the Fact_IncomePlan income-line feature group (Group 5) — values, point-in-time
    safety, leakage rule (no realized columns), and per-model contract selection;
  - the API `model` param flowing store → api.

All offline via InMemoryScoreStore + synthetic fixtures (no sklearn / no live data).
"""

import numpy as np
import pandas as pd
import pytest

from predictive import schema as S
from predictive import aux_features as AUX
from predictive.api import deal_detail, list_deals
from predictive.features import build_opp_features
from predictive.store import InMemoryScoreStore

SCORED_AT = pd.Timestamp("2026-06-06", tz="UTC").to_pydatetime()


def _row(opp, prob, band, amount, status="Open", model_id=S.DEFAULT_MODEL_ID):
    return {
        "opp_id": opp, "win_prob": prob, "band": band,
        "drivers": [{"feature": "amount_log", "value": 1, "impact": 0.3}],
        "amount": amount, "status": status, "scored_at": SCORED_AT,
        "model_run_id": "run-1", "so_plan_date": None, "at_risk": False,
        "model_id": model_id,
    }


# --- model registry / whitelist -------------------------------------------

def test_normalize_model_id_whitelists():
    assert S.normalize_model_id("CRM_PDT_MIX") == "CRM_PDT_MIX"
    assert S.normalize_model_id("CRM_PDT_BASE") == "CRM_PDT_BASE"
    assert S.normalize_model_id(None) == "CRM_PDT_BASE"          # default
    assert S.normalize_model_id("garbage") == "CRM_PDT_BASE"     # invalid → default
    assert S.normalize_model_id("crm_pdt_inc") == "CRM_PDT_BASE"  # case-sensitive literal


def test_feature_columns_per_model():
    opp_cols = S.feature_columns("CRM_PDT_BASE")
    inc_cols = S.feature_columns("CRM_PDT_MIX")
    assert opp_cols == S.FEATURE_COLUMNS
    # INC = OPP cols + the income group (no income feature in OPP)
    assert set(inc_cols) - set(opp_cols) == set(S.INCOME_FEATURES)
    assert not any(c in opp_cols for c in S.INCOME_FEATURES)
    # invalid/None → OPP contract
    assert S.feature_columns(None) == S.FEATURE_COLUMNS
    assert S.feature_columns("nope") == S.FEATURE_COLUMNS


# --- store model_id dimension ---------------------------------------------

def test_both_models_coexist_per_opp():
    """The SAME opp scored by both models must be stored separately, not overwrite."""
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("A", 0.90, "High", 1_000_000, model_id="CRM_PDT_BASE"),
        _row("A", 0.40, "Low", 1_000_000, model_id="CRM_PDT_MIX"),
    ])
    assert s.get_deal_score("A", model_id="CRM_PDT_BASE")["win_prob"] == 0.90
    assert s.get_deal_score("A", model_id="CRM_PDT_MIX")["win_prob"] == 0.40


def test_get_latest_filters_by_model():
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("A", 0.90, "High", 1_000_000, model_id="CRM_PDT_BASE"),
        _row("B", 0.80, "High", 2_000_000, model_id="CRM_PDT_BASE"),
        _row("A", 0.30, "Low", 1_000_000, model_id="CRM_PDT_MIX"),
    ])
    opp = {r["opp_id"] for r in s.get_latest_scores(model_id="CRM_PDT_BASE")}
    inc = {r["opp_id"] for r in s.get_latest_scores(model_id="CRM_PDT_MIX")}
    assert opp == {"A", "B"}
    assert inc == {"A"}


def test_default_model_id_is_opp():
    """Existing call sites omit model_id → must see only CRM_PDT_BASE scores."""
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("A", 0.90, "High", 1_000_000),                       # defaulted → OPP
        _row("B", 0.30, "Low", 2_000_000, model_id="CRM_PDT_MIX"),
    ])
    rows = s.get_latest_scores()  # no model_id arg
    assert {r["opp_id"] for r in rows} == {"A"}


def test_summarize_isolated_by_model():
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("A", 0.50, "Mid", 10_000_000, model_id="CRM_PDT_BASE"),
        _row("A", 0.50, "Mid", 10_000_000, model_id="CRM_PDT_MIX"),
        _row("B", 0.50, "Mid", 5_000_000, model_id="CRM_PDT_MIX"),
    ])
    assert s.summarize(model_id="CRM_PDT_BASE")["count"] == 1
    assert s.summarize(model_id="CRM_PDT_MIX")["count"] == 2
    assert s.summarize(model_id="CRM_PDT_MIX")["raw"] == pytest.approx(15e6)


# --- API model param -------------------------------------------------------

def test_list_deals_model_param_routes_to_model_scores():
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("A", 0.95, "High", 1_000_000, model_id="CRM_PDT_BASE"),
        _row("A", 0.20, "Low", 1_000_000, model_id="CRM_PDT_MIX"),
    ])
    opp = list_deals(s, model_id="CRM_PDT_BASE")
    inc = list_deals(s, model_id="CRM_PDT_MIX")
    assert opp["model_id"] == "CRM_PDT_BASE"
    assert opp["deals"][0]["win_prob"] == 0.95
    assert inc["model_id"] == "CRM_PDT_MIX"
    assert inc["deals"][0]["win_prob"] == 0.20


def test_list_deals_invalid_model_defaults_to_opp():
    s = InMemoryScoreStore()
    s.upsert_scores([_row("A", 0.95, "High", 1_000_000, model_id="CRM_PDT_BASE")])
    out = list_deals(s, model_id="bogus")
    assert out["model_id"] == "CRM_PDT_BASE"
    assert {d["opp_id"] for d in out["deals"]} == {"A"}


def test_deal_detail_model_param():
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("A", 0.95, "High", 1_000_000, model_id="CRM_PDT_BASE"),
        _row("A", 0.20, "Low", 1_000_000, model_id="CRM_PDT_MIX"),
    ])
    assert deal_detail(s, "A", model_id="CRM_PDT_BASE")["win_prob"] == 0.95
    assert deal_detail(s, "A", model_id="CRM_PDT_MIX")["win_prob"] == 0.20


# --- Group 5: income-line feature builder ---------------------------------

def _inc(rows):
    """rows: list of (opp_id, ym, amount)."""
    return pd.DataFrame(rows, columns=["opp_id", "ym", "amount"])


def _ref(opp_ids):
    return pd.DataFrame({"opp_id": opp_ids, "acct_id": "", "ref_date": pd.NaT})


def test_income_features_values():
    income = _inc([
        ("O1", "2026-03", 100.0),
        ("O1", "2026-05", 300.0),   # 2-line deal, spread = 2 months
        ("O2", "2026-07", 50.0),    # single line
    ])
    out = AUX.build_income_plan_features(_ref(["O1", "O2"]), income)
    r1 = out.loc["O1"]
    assert r1["income_line_count"] == 2.0
    assert r1["income_total_p_log"] == pytest.approx(np.log1p(400.0))
    assert r1["income_line_month_spread"] == 2.0
    assert r1["income_has_multi_line"] == 1.0
    r2 = out.loc["O2"]
    assert r2["income_line_count"] == 1.0
    assert r2["income_line_month_spread"] == 0.0
    assert r2["income_has_multi_line"] == 0.0


def test_income_features_default_fill_when_absent():
    out = AUX.build_income_plan_features(_ref(["O1", "O2"]), None)
    assert (out[AUX.INCOME_FEATURES] == 0.0).all().all()
    # a deal with no lines still gets a (zero) row
    out2 = AUX.build_income_plan_features(_ref(["O1", "O3"]),
                                          _inc([("O1", "2026-03", 100.0)]))
    assert out2.loc["O3", "income_line_count"] == 0.0


def test_income_created_cutoff_is_point_in_time():
    """If Created On is present, lines created at/after the deal's ref_date must be
    DROPPED (leakage guard) — only lines created strictly before count."""
    ref = pd.DataFrame({
        "opp_id": ["O1"], "acct_id": [""], "ref_date": [pd.Timestamp("2026-03-01")],
    })
    income = pd.DataFrame({
        "opp_id": ["O1", "O1", "O1"],
        "ym": ["2026-04", "2026-05", "2026-06"],
        "amount": [100.0, 200.0, 400.0],
        "created": ["2026-01-15", "2026-02-20", "2026-03-10"],  # last is AFTER ref → drop
    })
    out = AUX.build_income_plan_features(ref, income)
    # only the first two (created < 2026-03-01) survive
    assert out.loc["O1", "income_line_count"] == 2.0
    assert out.loc["O1", "income_total_p_log"] == pytest.approx(np.log1p(300.0))


def test_income_no_created_column_keeps_all_lines():
    """When Created On is absent (open-deal scoring at asof), all lines pass through."""
    income = _inc([("O1", "2026-04", 100.0), ("O1", "2026-05", 200.0)])
    ref = pd.DataFrame({
        "opp_id": ["O1"], "acct_id": [""], "ref_date": [pd.Timestamp("2026-03-01")],
    })
    out = AUX.build_income_plan_features(ref, income)
    assert out.loc["O1", "income_line_count"] == 2.0


def test_income_lines_for_unknown_opps_are_ignored():
    """Income lines can reference opps NOT in the feature frame (other-year / closed).
    The builder must drop those rather than raise — only ref opps get a row."""
    income = _inc([
        ("O1", "2026-03", 100.0),
        ("GHOST", "2026-04", 500.0),   # not in ref → must be ignored, not crash
    ])
    out = AUX.build_income_plan_features(_ref(["O1"]), income)
    assert list(out.index) == ["O1"]
    assert out.loc["O1", "income_line_count"] == 1.0


def test_income_builder_ignores_realized_columns():
    """LEAKAGE GUARD: even if a realized SO Actual / Invoice column is present on the
    source frame, the builder must consume ONLY [opp_id, ym, amount] and never read it."""
    income = _inc([("O1", "2026-03", 100.0)])
    income["SO Actual Amount (P)"] = 999_999.0   # post-outcome — must be ignored
    income["SO Actual Date"] = "2026-04-01"
    out = AUX.build_income_plan_features(_ref(["O1"]), income)
    # value reflects only SO Plan amount (100), not the realized 999_999
    assert out.loc["O1", "income_total_p_log"] == pytest.approx(np.log1p(100.0))


# --- per-model feature contract through build_opp_features ------------------

def _opp_frame():
    return pd.DataFrame({
        S.COL_OPP_ID: ["O1", "O2"],
        S.COL_STATUS: ["Won", "Open"],
        S.COL_AMOUNT: [1_000_000, 2_000_000],
        S.COL_POSSIBILITY: [100, 50],
        S.COL_CREATE: ["2025-01-01", "2025-02-01"],
    })


def test_build_features_opp_model_excludes_income():
    fs = build_opp_features(_opp_frame(), asof="2026-06-06", model_id="CRM_PDT_BASE")
    assert list(fs.X.columns) == S.FEATURE_COLUMNS
    assert not any(c in fs.X.columns for c in S.INCOME_FEATURES)


def test_build_features_inc_model_includes_income():
    income = _inc([("O1", "2026-03", 100.0), ("O1", "2026-05", 300.0)])
    fs = build_opp_features(
        _opp_frame(), asof="2026-06-06", model_id="CRM_PDT_MIX", income=income
    )
    assert list(fs.X.columns) == S.feature_columns("CRM_PDT_MIX")
    assert all(c in fs.X.columns for c in S.INCOME_FEATURES)
    # O1 has 2 income lines; O2 has none → default 0
    o1 = fs.X.loc[fs.ids[fs.ids == "O1"].index[0]]
    assert o1["income_line_count"] == 2.0


def test_build_features_default_model_is_opp():
    fs = build_opp_features(_opp_frame(), asof="2026-06-06")  # no model_id
    assert list(fs.X.columns) == S.FEATURE_COLUMNS
