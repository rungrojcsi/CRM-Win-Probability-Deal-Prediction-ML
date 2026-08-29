"""Tests for predictive backtest — temporal holdout on historical closed deals.

Train on the EARLIEST (1 - test_frac) of closed deals, evaluate on the most
recent test_frac ("ย้อนหลัง 30%"). Asserts the split is point-in-time (no
look-ahead) and that honest out-of-sample metrics come back.
"""

import numpy as np
import pandas as pd
import pytest

from predictive import schema as S
from predictive.features import build_opp_features
from predictive.pipeline import run_backtest
from predictive.winprob import backtest_winprob

ASOF = "2026-06-05"


def _synth(n=600, seed=42):
    """Closed deals with a learnable signal + a close-date so they can be ordered
    in time. Mirrors test_predictive_winprob._synth but all rows are closed."""
    rng = np.random.default_rng(seed)
    aging = rng.integers(5, 360, n)
    hot = rng.integers(0, 2, n)
    amount = rng.integers(200_000, 8_000_000, n)
    solution = rng.choice(["MES", "ERP", "IoT", "QC"], n)
    prospect = rng.choice(["Manufacturing IT", "ERP", "IoT", "Electronics"], n)

    # signal lives in retained features only (flag_hot removed from the model).
    sol_effect = pd.Series({"MES": 1.6, "ERP": 0.4, "IoT": -1.2, "QC": -0.8})
    z = (
        -0.012 * (aging - 100)
        + 0.5 * (np.log1p(amount) - 14)
        + sol_effect.reindex(solution).to_numpy()
        + rng.normal(0, 0.5, n)
    )
    p = 1 / (1 + np.exp(-z))
    won = rng.binomial(1, p).astype(bool)

    # close date spread across ~2 years so a temporal split is meaningful
    close = pd.Timestamp(ASOF) - pd.to_timedelta(rng.integers(1, 720, n), unit="D")
    create = close - pd.to_timedelta(aging, unit="D")

    return pd.DataFrame(
        {
            S.COL_OPP_ID: [f"OPP-{i}" for i in range(n)],
            S.COL_STATUS: np.where(won, "Won", "Lost"),
            S.COL_AMOUNT: amount,
            S.COL_POSSIBILITY: rng.integers(5, 95, n),
            S.COL_PROGRESS: rng.integers(5, 95, n),
            S.COL_AGING: aging,
            S.COL_CYCLE: aging + rng.integers(0, 30, n),
            S.COL_FLAG_HOT: np.where(hot == 1, "Yes", "No"),
            S.COL_CREATE: create,
            S.COL_LAST_ACT: close,
            S.COL_SOLUTION: solution,
            S.COL_PROSPECT: prospect,
        }
    )


@pytest.fixture
def fs():
    return build_opp_features(_synth(), asof=ASOF)


def test_split_sizes_match_test_frac(fs):
    order = pd.Series(range(len(fs.X)), index=fs.X.index)  # already chronological
    res = backtest_winprob(fs.X, fs.y, order=order, test_frac=0.30)
    n_closed = int(fs.is_closed.sum())
    assert res["n_test"] == round(n_closed * 0.30)
    assert res["n_train"] + res["n_test"] == n_closed


def test_metrics_present_and_sane(fs):
    order = pd.Series(range(len(fs.X)), index=fs.X.index)
    res = backtest_winprob(fs.X, fs.y, order=order, test_frac=0.30)
    assert 0.0 <= res["auc"] <= 1.0
    assert res["auc"] > 0.60          # learns the signal out-of-sample
    assert 0.0 <= res["brier"] <= 1.0
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["test_base_rate"] <= 1.0


def test_no_lookahead_train_is_strictly_earlier(fs):
    """Every train deal must close no later than every test deal."""
    order = pd.Series(np.arange(len(fs.X)), index=fs.X.index)
    res = backtest_winprob(fs.X, fs.y, order=order, test_frac=0.30, return_split=True)
    assert res["_train_order_max"] <= res["_test_order_min"]


def test_calibration_bins_returned(fs):
    order = pd.Series(range(len(fs.X)), index=fs.X.index)
    res = backtest_winprob(fs.X, fs.y, order=order, test_frac=0.30, n_bins=5)
    assert len(res["calibration"]) >= 1
    for b in res["calibration"]:
        assert {"n", "mean_pred", "actual_rate"} <= set(b)
        assert b["n"] >= 1


def test_train_split_single_class_raises(fs):
    # force the early 70% to be all-Lost → no positive class to train on
    y = fs.y.copy()
    order = pd.Series(range(len(y)), index=y.index)
    cut = int(len(y) * 0.70)
    y.iloc[:cut] = 0.0
    with pytest.raises(ValueError):
        backtest_winprob(fs.X, y, order=order, test_frac=0.30)


def test_run_backtest_end_to_end(monkeypatch):
    """run_backtest wires ingest → features → temporal split by create-cohort.
    maturity_days=0 disables cohort filtering so the full synthetic set is used."""
    df = _synth()
    monkeypatch.setattr("predictive.pipeline.fetch_opportunities", lambda *a, **k: df)
    res = run_backtest(None, asof=ASOF, test_frac=0.30, maturity_days=0)
    assert res["n_test"] > 0
    assert res["auc"] > 0.60
    assert res["order_by"] == S.COL_CREATE
    assert res["maturity_days"] == 0


def test_run_backtest_maturity_filters_cohorts(monkeypatch):
    """A positive maturity_days drops immature (recently-created) cohorts, so the
    evaluated set is no larger than the unfiltered one."""
    df = _synth()
    monkeypatch.setattr("predictive.pipeline.fetch_opportunities", lambda *a, **k: df)
    full = run_backtest(None, asof=ASOF, test_frac=0.30, maturity_days=0)
    mature = run_backtest(None, asof=ASOF, test_frac=0.30, maturity_days=400)
    assert mature["maturity_days"] == 400
    assert mature["n_train"] + mature["n_test"] <= full["n_train"] + full["n_test"]
