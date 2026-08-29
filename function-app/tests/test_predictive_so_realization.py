"""Tests for so_realization Stage B — F8 build_realization_curve + F9
assemble_so_forecast (offline, synthetic)."""

import pandas as pd
import pytest

from predictive.so_realization import (
    build_realization_curve,
    assemble_so_forecast,
    RealizationCurve,
    _ym_to_int,
    _int_to_ym,
)

ASOF = "2026-06-15"   # current month June; Jan–May elapsed; resolved years < 2026


def test_ym_roundtrip_and_lag():
    assert _int_to_ym(_ym_to_int("2026-06")) == "2026-06"
    assert _ym_to_int("2026-07") - _ym_to_int("2026-06") == 1
    assert _ym_to_int("2026-01") - _ym_to_int("2025-12") == 1   # year boundary


def test_realization_rate_dollar_weighted_on_resolved_years():
    lines = pd.DataFrame([
        # resolved (2025): 100 planned, 45 realized → rate 0.45
        {"opp_id": "A", "plan_ym": "2025-03", "plan_amount": 100.0,
         "actual_ym": "2025-03", "actual_amount": 45.0},
        # resolved (2024): 100 planned, 50 realized → year rate 0.50
        {"opp_id": "B", "plan_ym": "2024-08", "plan_amount": 100.0,
         "actual_ym": "2024-08", "actual_amount": 50.0},
        # current year (2026) → NOT resolved, excluded from rate
        {"opp_id": "C", "plan_ym": "2026-02", "plan_amount": 999.0,
         "actual_ym": "", "actual_amount": 0.0},
    ])
    curve = build_realization_curve(lines, ASOF)
    assert curve.rate == pytest.approx(0.475)        # (45+50)/(100+100)
    assert curve.rate_low == pytest.approx(0.45)     # worst resolved year (2025)
    assert curve.n_plan == 2                          # 2026 line excluded


def test_conditional_rate_excludes_non_converted_opps():
    """With converted_opps, rate = realized/plan over CONVERTED opps only (the factor
    the assembler multiplies by P(convert)) — higher than the unconditional rate that
    drags in never-converting opps' planned $."""
    lines = pd.DataFrame([
        # converted opp C: 100 planned, 80 realized
        {"opp_id": "C", "plan_ym": "2025-01", "plan_amount": 100.0,
         "actual_ym": "2025-01", "actual_amount": 80.0},
        # never-converted opp N: 100 planned, 0 realized (drags unconditional down)
        {"opp_id": "N", "plan_ym": "2025-02", "plan_amount": 100.0,
         "actual_ym": "", "actual_amount": 0.0},
    ])
    cond = build_realization_curve(lines, ASOF, converted_opps={"C"})
    uncond = build_realization_curve(lines, ASOF)
    assert cond.rate == pytest.approx(0.80)        # 80/100 over converted opp C only
    assert cond.rate_basis == "conditional"
    assert uncond.rate == pytest.approx(0.40)      # 80/200 over all
    assert cond.unconditional_rate == pytest.approx(0.40)
    assert cond.rate > uncond.rate                 # conditional must be higher


def test_lag_distribution_normalized():
    lines = pd.DataFrame([
        {"opp_id": "A", "plan_ym": "2024-01", "plan_amount": 80.0,
         "actual_ym": "2024-01", "actual_amount": 80.0},   # lag 0
        {"opp_id": "B", "plan_ym": "2024-01", "plan_amount": 20.0,
         "actual_ym": "2024-02", "actual_amount": 20.0},   # lag 1
    ])
    curve = build_realization_curve(lines, ASOF)
    assert curve.lag_dist[0] == pytest.approx(0.8)   # $-weighted
    assert curve.lag_dist[1] == pytest.approx(0.2)
    assert sum(curve.lag_dist.values()) == pytest.approx(1.0)


def test_assemble_elapsed_actual_vs_predicted():
    curve = RealizationCurve(rate=0.5, rate_low=0.4, lag_dist={0: 1.0})
    plan = pd.DataFrame([
        {"opp_id": "X", "ym": "2026-06", "amount": 100.0},   # current month
        {"opp_id": "Y", "ym": "2026-07", "amount": 40.0},    # future
        {"opp_id": "Z", "ym": "2026-03", "amount": 999.0},   # past plan → ignored (elapsed=actual)
    ])
    conv = {"X": 0.8, "Y": 0.5}                              # Z has no conv → 0
    actual = {"2026-03": 30.0, "2026-04": 20.0}
    fc = assemble_so_forecast(plan, conv, curve, actual, ASOF)
    m = {r["month"]: r for r in fc["months"]}
    assert len(fc["months"]) == 12
    assert m["2026-03"]["type"] == "actual" and m["2026-03"]["expected"] == 30.0
    # June expected = 100 * 0.8 * 0.5 = 40 ; conservative = 100*0.8*0.4 = 32
    assert m["2026-06"]["expected"] == pytest.approx(40.0)
    assert m["2026-06"]["conservative"] == pytest.approx(32.0)
    assert m["2026-07"]["expected"] == pytest.approx(40.0 * 0.5 * 0.5)   # amt*conv*rate = 10
    # yearly = elapsed actual (30+20) + predicted (40 + 10)
    assert fc["yearly_expected"] == pytest.approx(50.0 + 40.0 + 10.0)
    assert fc["elapsed_actual"] == pytest.approx(50.0)
    assert fc["pipeline_plan"] == pytest.approx(140.0)   # 100 + 40 (current+future only)


def test_assemble_lag_spreads_into_future_month():
    curve = RealizationCurve(rate=1.0, rate_low=1.0, lag_dist={0: 0.5, 1: 0.5})
    plan = pd.DataFrame([{"opp_id": "X", "ym": "2026-06", "amount": 100.0}])
    fc = assemble_so_forecast(plan, {"X": 1.0}, curve, {}, ASOF)
    m = {r["month"]: r for r in fc["months"]}
    assert m["2026-06"]["expected"] == pytest.approx(50.0)   # lag 0 half
    assert m["2026-07"]["expected"] == pytest.approx(50.0)   # lag 1 half


def test_empty_lines_safe():
    curve = build_realization_curve(pd.DataFrame(), ASOF)
    assert curve.rate == 0.0
    fc = assemble_so_forecast(pd.DataFrame(), {}, curve, {"2026-01": 5.0}, ASOF)
    assert fc["yearly_expected"] == pytest.approx(5.0)   # only elapsed actual
