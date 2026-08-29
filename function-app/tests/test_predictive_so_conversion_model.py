"""Tests for so_conversion Stage-A model — F5 train / F6 score+explain / F7 backtest
(offline, synthetic). Reuses the winprob estimator with the SO-conversion label."""

import numpy as np
import pandas as pd
import pytest

from predictive import schema as S
from predictive.so_conversion import (
    train_so_conversion,
    score_so_conversion,
    explain_so_conversion,
    backtest_so_conversion,
)

ASOF = "2026-06-01"


def _synth(n=80, seed=0):
    """n mature opps; converted ones (have a SO) skew to higher amount + lower aging
    so the model has real signal. Returns (opps, so_conversions)."""
    rng = np.random.RandomState(seed)
    rows, so_rows = [], []
    for i in range(n):
        converted = i % 2 == 0
        oid = f"OPP{i:03d}"
        amount = (rng.uniform(3e6, 9e6) if converted else rng.uniform(2e5, 2e6))
        aging = (rng.uniform(20, 120) if converted else rng.uniform(150, 500))
        rows.append({
            S.COL_OPP_ID: oid,
            S.COL_STATUS: "Open" if i % 3 else "Lost",   # mix; maturity makes them resolved
            S.COL_AMOUNT: amount,
            S.COL_POSSIBILITY: rng.randint(0, 100),
            S.COL_CREATE: "2023-01-15",                  # > 540d before asof → mature
            "Solution Name": "Sol_A" if converted else "Sol_B",
            "Prospect Category Name": "Existing" if i % 2 else "New",
            "Aging Days": aging,
            "Last Activity Date": "2023-06-01",
            "Account ID": f"ACC{i % 10}",
        })
        if converted:
            so_rows.append({"opp_id": oid, "so_count": 1, "so_total": amount,
                            "so_invoiced": 0.0, "so_first_date": "2023-03-01"})
    return pd.DataFrame(rows), pd.DataFrame(so_rows)


def test_train_returns_model_with_both_classes():
    opps, so = _synth()
    model = train_so_conversion(opps, so, ASOF)
    assert model.metrics["n_train"] > 0
    assert 0 < model.metrics["n_pos"] < model.metrics["n_train"]   # both classes present


def test_score_in_unit_range_and_aligned():
    opps, so = _synth()
    model = train_so_conversion(opps, so, ASOF)
    p = score_so_conversion(model, opps, ASOF)
    assert len(p) == len(opps)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_explain_returns_drivers_per_row():
    opps, so = _synth(n=40)
    model = train_so_conversion(opps, so, ASOF)
    drivers = explain_so_conversion(model, opps.head(5), ASOF, top_n=3)
    assert len(drivers) == 5
    assert all(len(d) <= 3 for d in drivers)
    assert all({"feature", "value", "impact"} <= set(d[0].keys()) for d in drivers if d)


def test_backtest_reports_oos_metrics_and_lift():
    opps, so = _synth(n=120)
    out = backtest_so_conversion(opps, so, ASOF, test_frac=0.30)
    assert out["n_train"] > 0 and out["n_test"] > 0
    assert "auc" in out and "brier" in out
    assert "lift" in out and "baseline_accuracy" in out
    assert isinstance(out["calibration"], list)
    # signal is intentionally strong → AUC should beat chance
    if out["auc"] is not None:
        assert out["auc"] >= 0.5


def test_open_with_so_is_learned_as_converted():
    """Sanity: an Open deal with a SO contributes a positive training label
    (the redesign's core), so n_pos reflects converted opps incl Open ones."""
    opps, so = _synth(n=60)
    # count converted via the label path indirectly: all even-i have SO
    model = train_so_conversion(opps, so, ASOF)
    assert model.metrics["n_pos"] == 30   # 30 of 60 have a SO → positive
