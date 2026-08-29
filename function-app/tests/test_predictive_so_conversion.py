"""Tests for the new SO predictive — F1 aggregate_so_conversions + F3
build_so_conversion_label (offline, synthetic). The conversion label is grounded in
the real Fact_SalesOrder ledger, not Status=Won."""

import numpy as np
import pandas as pd
import pytest

from predictive import schema as S
from predictive.ingest import aggregate_so_conversions
from predictive.features import build_so_conversion_label

ASOF = "2026-06-01"


# --- F1: aggregate the per-order SO ledger to opportunity grain ---

def test_aggregate_so_conversions_collapses_multi_order_opp():
    df = pd.DataFrame([
        {"opp_id": "A", "grand_total": 100.0, "invoiced": 40.0, "created": "2025-01-10"},
        {"opp_id": "A", "grand_total": 60.0, "invoiced": 60.0, "created": "2025-03-02"},
        {"opp_id": "B", "grand_total": 50.0, "invoiced": 0.0, "created": "2026-02-01"},
    ])
    out = aggregate_so_conversions(df).set_index("opp_id")
    assert out.loc["A", "so_count"] == 2
    assert out.loc["A", "so_total"] == pytest.approx(160.0)
    assert out.loc["A", "so_invoiced"] == pytest.approx(100.0)
    assert pd.Timestamp(out.loc["A", "so_first_date"]) == pd.Timestamp("2025-01-10")  # earliest
    assert out.loc["B", "so_count"] == 1


def test_aggregate_so_conversions_empty():
    out = aggregate_so_conversions(pd.DataFrame())
    assert list(out.columns) == ["opp_id", "so_count", "so_total", "so_invoiced", "so_first_date"]
    assert out.empty


# --- F3: SO-conversion label ---

def _opps(rows):
    return pd.DataFrame(rows)


def test_label_open_deal_with_so_is_positive():
    """The crux: an Open deal that already booked a Sales Order = converted=1,
    even though Status is not Won (audit found 210 such deals)."""
    opps = _opps([
        {S.COL_OPP_ID: "OPEN_SO", S.COL_STATUS: "Open", S.COL_CREATE: "2026-05-01"},
    ])
    so = pd.DataFrame([{"opp_id": "OPEN_SO", "so_count": 1, "so_total": 100.0,
                        "so_invoiced": 0.0, "so_first_date": "2026-05-20"}])
    y = build_so_conversion_label(opps, so, ASOF)
    assert y.iloc[0] == 1.0


def test_label_lost_no_so_is_negative_regardless_of_maturity():
    opps = _opps([
        {S.COL_OPP_ID: "LOST_RECENT", S.COL_STATUS: "Lost", S.COL_CREATE: "2026-05-15"},
    ])
    y = build_so_conversion_label(opps, pd.DataFrame(columns=["opp_id"]), ASOF)
    assert y.iloc[0] == 0.0   # Lost is resolved even though immature


def test_label_mature_no_so_is_negative():
    opps = _opps([
        {S.COL_OPP_ID: "OLD_OPEN", S.COL_STATUS: "Open", S.COL_CREATE: "2023-01-01"},  # > 540d
    ])
    y = build_so_conversion_label(opps, pd.DataFrame(columns=["opp_id"]), ASOF)
    assert y.iloc[0] == 0.0


def test_label_immature_no_so_is_nan():
    opps = _opps([
        {S.COL_OPP_ID: "NEW_OPEN", S.COL_STATUS: "Open", S.COL_CREATE: "2026-05-20"},  # < 540d
    ])
    y = build_so_conversion_label(opps, pd.DataFrame(columns=["opp_id"]), ASOF)
    assert np.isnan(y.iloc[0])   # right-censored — excluded from training


def test_label_won_with_so_positive_and_index_aligned():
    opps = _opps([
        {S.COL_OPP_ID: "WON1", S.COL_STATUS: "Won", S.COL_CREATE: "2025-01-01"},
        {S.COL_OPP_ID: "NEW_OPEN", S.COL_STATUS: "Open", S.COL_CREATE: "2026-05-25"},
        {S.COL_OPP_ID: "OLD_OPEN", S.COL_STATUS: "Open", S.COL_CREATE: "2023-01-01"},
    ])
    so = pd.DataFrame([{"opp_id": "WON1", "so_count": 1, "so_total": 9.0,
                        "so_invoiced": 9.0, "so_first_date": "2025-02-01"}])
    y = build_so_conversion_label(opps, so, ASOF)
    assert list(y.index) == list(opps.index)          # aligned to opps
    assert y.iloc[0] == 1.0 and np.isnan(y.iloc[1]) and y.iloc[2] == 0.0
