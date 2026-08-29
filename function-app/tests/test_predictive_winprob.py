"""Tests for predictive.winprob — F6 train, F7 score, F8 explain.

Uses a synthetic Fact_Opportunity frame with a KNOWN signal so we can assert
the model learns it (AUC) and that every score carries drivers (G5).
"""

import numpy as np
import pandas as pd
import pytest

from predictive import schema as S
from predictive.features import build_opp_features
from predictive.schema import band
from predictive.winprob import explain_winprob, score_winprob, train_winprob

ASOF = "2026-06-05"


def _synth(n=600, seed=42):
    """Synthetic deals where Win is driven by high possibility/progress, hot flag,
    low aging — and noise. Returns a raw Fact_Opportunity-shaped frame."""
    rng = np.random.default_rng(seed)
    possibility = rng.integers(5, 95, n)   # present in source but NOT a feature (leakage)
    progress = rng.integers(5, 95, n)      # present in source but NOT a feature (text/NaN)
    aging = rng.integers(5, 360, n)
    hot = rng.integers(0, 2, n)
    amount = rng.integers(200_000, 8_000_000, n)
    solution = rng.choice(["MES", "ERP", "IoT", "QC"], n)
    prospect = rng.choice(["Manufacturing IT", "ERP", "IoT", "Electronics"], n)

    # latent win score driven ONLY by retained features (aging, amount, solution).
    # flag_hot was removed from the model (train/serving skew), so the synthetic
    # signal must live in the features the model actually uses.
    sol_effect = pd.Series({"MES": 1.6, "ERP": 0.4, "IoT": -1.2, "QC": -0.8})
    z = (
        -0.012 * (aging - 100)
        + 0.5 * (np.log1p(amount) - 14)
        + sol_effect.reindex(solution).to_numpy()
        + rng.normal(0, 0.5, n)
    )
    p = 1 / (1 + np.exp(-z))
    won = rng.binomial(1, p).astype(bool)

    create = pd.Timestamp(ASOF) - pd.to_timedelta(aging, unit="D")
    last_act = pd.Timestamp(ASOF) - pd.to_timedelta(rng.integers(1, 60, n), unit="D")

    return pd.DataFrame(
        {
            S.COL_OPP_ID: [f"OPP-{i}" for i in range(n)],
            S.COL_STATUS: np.where(won, "Won", "Lost"),
            S.COL_AMOUNT: amount,
            S.COL_POSSIBILITY: possibility,
            S.COL_PROGRESS: progress,
            S.COL_AGING: aging,
            S.COL_CYCLE: aging + rng.integers(0, 30, n),
            S.COL_FLAG_HOT: np.where(hot == 1, "Yes", "No"),
            S.COL_CREATE: create,
            S.COL_LAST_ACT: last_act,
            S.COL_SOLUTION: solution,
            S.COL_PROSPECT: prospect,
        }
    )


def _synth_activity(df, seed=7):
    """Synthetic Fact_Activity rows so the aux engagement features vary per deal
    (mirrors live data — without this the aux columns are all default-filled)."""
    rng = np.random.default_rng(seed)
    rows = []
    for opp in df[S.COL_OPP_ID]:
        for _ in range(int(rng.integers(0, 6))):
            d = pd.Timestamp(ASOF) - pd.to_timedelta(int(rng.integers(1, 180)), unit="D")
            rows.append({
                S.COL_ACT_OPP_ID: opp,
                S.COL_ACT_DATE: d,
                S.COL_ACT_TYPE: rng.choice(["Email", "Phone Call", "Appointment"]),
                S.COL_ACT_DURATION: int(rng.integers(5, 90)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fs():
    df = _synth()
    return build_opp_features(df, asof=ASOF, activity=_synth_activity(df))


def test_train_learns_signal(fs):
    wm = train_winprob(fs.X, fs.y)
    assert "auc_cv" in wm.metrics
    assert wm.metrics["auc_cv"] > 0.70  # NFR target
    assert wm.metrics["n_train"] == int(fs.is_closed.sum())


def test_train_requires_both_classes():
    df = _synth(50)
    df[S.COL_STATUS] = "Won"  # single class
    fs = build_opp_features(df, asof=ASOF)
    with pytest.raises(ValueError):
        train_winprob(fs.X, fs.y)


def test_score_range_and_shape(fs):
    wm = train_winprob(fs.X, fs.y)
    p = score_winprob(wm, fs.X)
    assert p.shape == (len(fs.X),)
    assert ((p >= 0) & (p <= 1)).all()


def test_every_score_has_drivers(fs):
    """G5 hard requirement: no probability without a reason."""
    wm = train_winprob(fs.X, fs.y)
    drivers = explain_winprob(wm, fs.X.head(20), top_n=3)
    assert len(drivers) == 20
    for row in drivers:
        assert 1 <= len(row) <= 3
        for d in row:
            assert "feature" in d and "impact" in d and "value" in d
            assert d["feature"] in S.FEATURE_COLUMNS


def test_explain_all_features(fs):
    """top_n=None must return every feature (full SHAP vector) per row."""
    wm = train_winprob(fs.X, fs.y)
    n_feats = len(S.feature_columns(None))   # total feature count for the default model
    drivers = explain_winprob(wm, fs.X.head(5), top_n=None)
    assert len(drivers) == 5
    for row in drivers:
        assert len(row) == n_feats, f"expected {n_feats} drivers, got {len(row)}"
        names = {d["feature"] for d in row}
        assert names.issubset(set(S.FEATURE_COLUMNS))


def test_drivers_are_per_row(fs):
    """SHAP must give DIFFERENT drivers across deals (not global fallback)."""
    wm = train_winprob(fs.X, fs.y)
    drivers = explain_winprob(wm, fs.X.head(30), top_n=3)
    signatures = {tuple((d["feature"], d["impact"]) for d in row) for row in drivers}
    assert len(signatures) > 1  # would be 1 if fallback/global fired


def test_band_thresholds():
    assert band(0.85) == "High"
    assert band(0.55) == "Mid"
    assert band(0.20) == "Low"
