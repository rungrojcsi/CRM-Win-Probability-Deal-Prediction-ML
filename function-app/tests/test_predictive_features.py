"""Tests for predictive.features — F2 validate + F3 build_opp_features (offline)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive import schema as S
from predictive.features import build_opp_features, validate_features

FIXTURE = Path(__file__).parent / "fixtures" / "winprob_sample.csv"
ASOF = "2025-06-01"


@pytest.fixture
def raw():
    return pd.read_csv(FIXTURE)


def test_validate_ok(raw):
    ok, issues = validate_features(raw)
    assert ok and issues == []


def test_validate_missing_column():
    df = pd.DataFrame({"Opportunity ID": ["X"]})  # missing Status/amount/possibility
    ok, issues = validate_features(df)
    assert not ok
    assert any("Status" in m for m in issues)


def test_validate_empty():
    ok, issues = validate_features(pd.DataFrame(columns=S.REQUIRED_SOURCE_COLUMNS))
    assert not ok and any("empty" in m for m in issues)


def test_feature_columns_contract(raw):
    fs = build_opp_features(raw, asof=ASOF)
    assert list(fs.X.columns) == S.FEATURE_COLUMNS
    assert len(fs.X) == len(raw)


def test_label_uses_status_not_iswon(raw):
    fs = build_opp_features(raw, asof=ASOF)
    # Won=1, Lost=0, Open=NaN
    labels = dict(zip(fs.ids, fs.y))
    assert labels["OPP-W1"] == 1.0
    assert labels["OPP-L1"] == 0.0
    assert np.isnan(labels["OPP-O1"])
    assert fs.is_closed.sum() == 4  # 2 Won + 2 Lost


def test_amount_log_and_flag_hot_excluded(raw):
    fs = build_opp_features(raw, asof=ASOF)
    row = fs.X[fs.ids.values == "OPP-W1"].iloc[0]
    assert row["amount_log"] == pytest.approx(np.log1p(3500000), rel=1e-6)
    # flag_hot removed 2026-06-05 (train/serving skew — constant 0 in closed deals)
    # → no longer a model feature, must not appear in the feature matrix.
    assert "flag_hot" not in fs.X.columns


def test_aging_derived_when_missing(raw):
    fs = build_opp_features(raw, asof=ASOF)
    # OPP-O2 has blank Aging Days → derive from Create Date 2025-03-15 to asof.
    aging = fs.X[fs.ids.values == "OPP-O2"].iloc[0]["aging_days"]
    expected = (pd.Timestamp(ASOF) - pd.Timestamp("2025-03-15")).days
    assert aging == expected


def test_categorical_dtype(raw):
    fs = build_opp_features(raw, asof=ASOF)
    for col in S.CATEGORICAL_FEATURES:
        assert str(fs.X[col].dtype) == "category"


def test_maturity_default_includes_all(raw):
    # default maturity_days=0 → every deal counts as mature (backward compatible).
    fs = build_opp_features(raw, asof=ASOF)
    assert fs.is_mature.all()


def test_is_mature_by_create_age(raw):
    # OPP-W1 created 2024-09-01, OPP-O2 created 2025-03-15; asof 2025-06-01.
    fs = build_opp_features(raw, asof=ASOF, maturity_days=100)
    m = dict(zip(fs.ids, fs.is_mature))
    assert m["OPP-W1"] is True or m["OPP-W1"] == True  # ~273d ≥ 100
    assert m["OPP-O2"] == False                         # ~78d  < 100


def test_mature_training_labels_excludes_immature_closed(raw):
    from predictive.features import mature_training_labels

    # asof shortly after W1's create → W1 (closed) is immature, L1 is mature.
    fs = build_opp_features(raw, asof="2024-09-15", maturity_days=200)
    yl = dict(zip(fs.ids, mature_training_labels(fs)))
    assert yl["OPP-L1"] == 0.0        # Lost, created 2024-01-10 → mature → kept
    assert np.isnan(yl["OPP-W1"])     # Won, created 2024-09-01 → immature → dropped
