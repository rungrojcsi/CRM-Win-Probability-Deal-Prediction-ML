"""Tests for predictive.forecast (F9) and predictive.attainment (F10/F11) — offline."""

import pandas as pd
import pytest

from predictive.attainment import compute_attainment
from predictive.forecast import aggregate_monthly, forecast_revenue


# ---------- forecast (F9) ----------

def _invoices_monthly(values, start="2025-01-01"):
    """One invoice mid-month for each value; gaps allowed via None."""
    rows = []
    months = pd.period_range(start, periods=len(values), freq="M")
    for m, v in zip(months, values):
        if v is not None:
            rows.append({"sales_id": "S1", "date": m.to_timestamp() + pd.Timedelta(days=14), "amount": v})
    return pd.DataFrame(rows)


def test_aggregate_fills_month_gaps():
    inv = _invoices_monthly([100, None, 300])  # Feb missing
    s = aggregate_monthly(inv)
    assert len(s) == 3
    assert s.iloc[1] == 0.0  # gap filled


def test_forecast_shape_and_band():
    inv = _invoices_monthly([50, 55, 45, 60, 52, 58, 54, 56], start="2025-01-01")
    out = forecast_revenue(inv, asof="2025-09-15", horizon=3)
    assert len(out["forecast"]) == 3
    for p in out["forecast"]:
        assert p["lower"] <= p["forecast"] <= p["upper"]
        assert p["lower"] >= 0
    # first forecast month is the month after last complete month (Aug) → Sep
    assert out["forecast"][0]["month"] == "2025-09"


def test_forecast_drops_partial_current_month():
    inv = _invoices_monthly([50, 55, 45, 60, 52, 999], start="2025-01-01")  # Jun partial/outlier
    out = forecast_revenue(inv, asof="2025-06-10", horizon=1)
    assert out["last_complete_month"] == "2025-05"  # Jun excluded


def test_forecast_needs_min_history():
    inv = _invoices_monthly([50, 55], start="2025-01-01")
    with pytest.raises(ValueError):
        forecast_revenue(inv, asof="2025-03-15")


# ---------- attainment (F10/F11) ----------

@pytest.fixture
def targets():
    return pd.DataFrame(
        [
            {"sales_id": "S1", "month": "2026-06-01", "target": 1_000_000},
            {"sales_id": "S2", "month": "2026-06-01", "target": 2_000_000},
            {"sales_id": "S3", "month": "2026-06-01", "target": 0},  # no target
        ]
    )


@pytest.fixture
def invoices():
    # June 2026, by mid-month
    return pd.DataFrame(
        [
            {"sales_id": "S1", "date": "2026-06-05", "amount": 600_000},
            {"sales_id": "S2", "date": "2026-06-10", "amount": 300_000},
            {"sales_id": "S3", "date": "2026-06-08", "amount": 100_000},
        ]
    )


def test_attainment_runrate_projection(targets, invoices):
    # asof Jun 15 → 15/30 = 0.5 elapsed
    out = compute_attainment(targets, invoices, asof="2026-06-15")
    assert out["month"] == "2026-06"
    assert out["month_fraction_elapsed"] == pytest.approx(0.5)
    by = {r["sales_id"]: r for r in out["by_sales"]}
    # S1: 600k MTD / 0.5 = 1.2M eom vs 1M target → 120%
    assert by["S1"]["predicted_eom"] == pytest.approx(1_200_000)
    assert by["S1"]["attainment_pct"] == pytest.approx(120.0)
    assert by["S1"]["status"] == "ahead"


def test_attainment_behind_and_no_target(targets, invoices):
    out = compute_attainment(targets, invoices, asof="2026-06-15")
    by = {r["sales_id"]: r for r in out["by_sales"]}
    # S2: 300k/0.5 = 600k vs 2M → 30% behind
    assert by["S2"]["status"] == "behind"
    # S3: no target → attainment None, status no_target
    assert by["S3"]["attainment_pct"] is None
    assert by["S3"]["status"] == "no_target"


def test_team_rollup(targets, invoices):
    out = compute_attainment(targets, invoices, asof="2026-06-15")
    assert out["team"]["target"] == pytest.approx(3_000_000)
    assert out["team"]["predicted_eom"] == pytest.approx(2_000_000)  # (600k+300k+100k)/0.5


def test_low_confidence_guard_early_month(targets, invoices):
    # Jun 3 → ~10% elapsed → below default 25% threshold
    out = compute_attainment(targets, invoices, asof="2026-06-03")
    assert out["low_confidence"] is True
    assert out["note"] and "elapsed" in out["note"]


def test_confident_after_threshold(targets, invoices):
    out = compute_attainment(targets, invoices, asof="2026-06-20")  # ~67% elapsed
    assert out["low_confidence"] is False
    assert out["note"] is None
