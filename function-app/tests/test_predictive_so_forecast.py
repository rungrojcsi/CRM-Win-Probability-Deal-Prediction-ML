"""Tests for predictive.so_forecast — monthly + yearly SO predictive.

Past months use realized SO Actual; the current month and future months use
weighted SO Plan (amount x win_prob); past-due (at_risk) deals are excluded from
the monthly forecast and reported separately. Yearly = actual(elapsed) +
predicted(current..Dec).
"""

import pytest

from predictive.so_forecast import build_so_forecast, build_source_forecast

ASOF = "2026-06-05"  # current month = June (month 6); Jan–May elapsed

ACTUAL = {  # realized SO Actual by month (THB)
    "2026-01": 51_000_000,
    "2026-02": 36_000_000,
    "2026-03": 64_000_000,
    "2026-04": 61_000_000,
    "2026-05": 29_000_000,
    "2026-06": 15_000_000,  # partial — must be IGNORED (current month is predicted)
}

OPEN_ROWS = [
    # planned this month (June) — predicted
    {"so_plan_date": "2026-06-20", "amount": 10_000_000, "win_prob": 0.5, "at_risk": False},
    # planned July — predicted
    {"so_plan_date": "2026-07-10", "amount": 8_000_000, "win_prob": 0.25, "at_risk": False},
    # past-due (planned Feb, still open) — excluded from months, counted as at_risk
    {"so_plan_date": "2026-02-01", "amount": 4_000_000, "win_prob": 0.3, "at_risk": True},
    # next-year plan — out of scope
    {"so_plan_date": "2027-01-05", "amount": 9_000_000, "win_prob": 0.9, "at_risk": False},
]


@pytest.fixture
def fc():
    return build_so_forecast(ACTUAL, OPEN_ROWS, asof=ASOF)


def test_elapsed_months_use_actual(fc):
    m = {r["month"]: r for r in fc["months"]}
    assert m["2026-03"]["type"] == "actual"
    # elapsed: expected == conservative == optimistic == realized actual
    assert m["2026-03"]["expected"] == 64_000_000
    assert m["2026-03"]["conservative"] == 64_000_000 == m["2026-03"]["optimistic"]


def test_current_month_is_predicted_not_actual(fc):
    """June is the current month → expected (weighted), NOT the 15M partial actual."""
    jun = next(r for r in fc["months"] if r["month"] == "2026-06")
    assert jun["type"] == "predicted"
    assert jun["expected"] == pytest.approx(10_000_000 * 0.5)   # 5M expected
    assert jun["raw"] == pytest.approx(10_000_000)


def test_monte_carlo_band_brackets_expected(fc):
    """Single binary deal (p=0.5, 10M) → outcomes 0 or 10M, so the MC band must
    bracket expected: conservative ≤ expected ≤ optimistic ≤ raw."""
    jun = next(r for r in fc["months"] if r["month"] == "2026-06")
    assert jun["conservative"] <= jun["expected"] <= jun["optimistic"] <= jun["raw"]
    assert jun["optimistic"] == pytest.approx(10_000_000)   # P90 of a 0/10M coin = win value


def test_future_month_expected(fc):
    jul = next(r for r in fc["months"] if r["month"] == "2026-07")
    assert jul["expected"] == pytest.approx(8_000_000 * 0.25)   # 2M
    assert jul["raw"] == pytest.approx(8_000_000)


def test_past_due_excluded_and_reported(fc):
    # the Feb past-due deal must NOT inflate Feb (Feb is actual) nor any month
    assert fc["at_risk"]["weighted"] == pytest.approx(4_000_000 * 0.3)
    assert fc["at_risk"]["raw"] == pytest.approx(4_000_000)


def test_next_year_out_of_scope(fc):
    assert all(r["month"].startswith("2026-") for r in fc["months"])
    assert len(fc["months"]) == 12


def test_pipeline_plan_and_deal_count(fc):
    # two open future deals this year: Jun 10M + Jul 8M (past-due + next-year excluded)
    assert fc["n_deals"] == 2
    assert fc["pipeline_plan"] == pytest.approx(10_000_000 + 8_000_000)


def test_band_mix(fc):
    # Jun deal p=0.5 → Mid; Jul deal p=0.25 → Low
    assert fc["bands"] == {"High": 0, "Mid": 1, "Low": 1}


def test_yearly_expected_vs_conservative(fc):
    actual_elapsed = 51e6 + 36e6 + 64e6 + 61e6 + 29e6      # Jan–May
    predicted_rest = 10e6 * 0.5 + 8e6 * 0.25               # Jun + Jul expected
    assert fc["yearly_expected"] == pytest.approx(actual_elapsed + predicted_rest)
    # conservative (P10) sits between the realized floor and expected; optimistic above
    assert actual_elapsed <= fc["yearly_conservative"] <= fc["yearly_expected"]
    assert fc["yearly_optimistic"] >= fc["yearly_expected"]


# --- alternative-source re-base (Fact_IncomePlan / Fact_SOPlan) ---

def test_build_source_forecast_rebase():
    """Income-LINE re-base, structured like the deal-header forecast: elapsed months (< cur)
    show realized SO Actual; current+future show predicted Expected = Σ(line × win_prob) by
    each line's plan month. yearly = actual(elapsed) + expected(future). asof June → Jan–May
    elapsed."""
    lines = [
        {"opp_id": "A", "ym": "2026-06", "amount": 10_000_000},  # deal A split...
        {"opp_id": "A", "ym": "2026-07", "amount": 5_000_000},   # ...across Jun+Jul
        {"opp_id": "B", "ym": "2026-06", "amount": 2_000_000},
        {"opp_id": "C", "ym": "2025-12", "amount": 9_000_000},   # other year → ignored
        {"opp_id": "D", "ym": "2026-06", "amount": 1_000_000},   # opp not in win-prob map
        {"opp_id": "E", "ym": "2026-03", "amount": 8_000_000},   # elapsed (Mar) → actual used instead
    ]
    wp = {"A": 0.8, "B": 0.5, "C": 0.9, "E": 0.9}
    actual = {"2026-01": 4_000_000, "2026-03": 6_000_000}  # realized SO Actual (elapsed)
    fc = build_source_forecast(lines, wp, actual, "2026-06-06")
    mar = next(m for m in fc["months"] if m["month"] == "2026-03")
    jun = next(m for m in fc["months"] if m["month"] == "2026-06")
    jul = next(m for m in fc["months"] if m["month"] == "2026-07")
    assert mar["type"] == "actual" and mar["expected"] == pytest.approx(6_000_000)  # realized, not E's plan
    assert jun["type"] == "predicted"
    assert jun["raw"] == pytest.approx(13_000_000)              # 10 + 2 + 1
    assert jun["expected"] == pytest.approx(0.8 * 10e6 + 0.5 * 2e6 + 0.0 * 1e6)
    assert jul["expected"] == pytest.approx(0.8 * 5e6)
    assert fc["yearly_raw"] == pytest.approx(18_000_000)        # future prospect pipeline (Jun+Jul)
    assert fc["yearly_expected"] == pytest.approx((0.8 * 10e6 + 0.5 * 2e6) + (0.8 * 5e6) + 10_000_000)
    assert fc["elapsed_actual"] == pytest.approx(10_000_000)    # Jan 4M + Mar 6M
    assert fc["n_opps"] == 3                                    # A, B, D (future, distinct)
    assert fc["bands"]["High"] == 1                             # A (0.8 ≥ 0.70)
    assert len(fc["months"]) == 12


def test_enrich_source_lines():
    """Per-line rows for by-customer tables: keep only lines in the kept months, attach
    win-prob + names + band."""
    from predictive.so_forecast import enrich_source_lines
    lines = [
        {"opp_id": "A", "ym": "2026-06", "amount": 3_000_000},
        {"opp_id": "A", "ym": "2026-09", "amount": 1_000_000},   # outside keep window
        {"opp_id": "B", "ym": "2026-07", "amount": 2_000_000},
    ]
    wp = {"A": 0.8, "B": 0.3}
    names = {"A": {"account_name": "Acme", "opp_name": "MES"}, "B": {"account_name": "Beta", "opp_name": "ERP"}}
    keep = {"2026-06", "2026-07", "2026-08"}
    out = enrich_source_lines(lines, wp, names, keep)
    assert {r["ym"] for r in out} == {"2026-06", "2026-07"}    # Sep dropped
    a = next(r for r in out if r["opp_id"] == "A")
    assert a["win_pct"] == 80.0 and a["band"] == "High" and a["account_name"] == "Acme"


def test_delay_prospects_filter():
    """Delay Prospects = past-due income-lines (plan month < current) whose deal is still
    OPEN (in the win-prob store). Closed/absent opps are excluded."""
    from predictive.so_forecast import enrich_source_lines
    lines = [
        {"opp_id": "OPEN1", "ym": "2026-05", "amount": 62_000_000},  # past-due, open → delayed
        {"opp_id": "OPEN1", "ym": "2026-06", "amount": 1_000_000},   # current → not delayed
        {"opp_id": "CLOSED", "ym": "2026-04", "amount": 9_000_000},  # past-due but not in store → excluded
    ]
    wp = {"OPEN1": 0.83}                      # CLOSED absent → resolved, not "delayed"
    names = {"OPEN1": {"account_name": "Acme", "opp_name": "Auto Call"}}
    past = {"2026-01", "2026-02", "2026-03", "2026-04", "2026-05"}  # months < June
    delayed = [d for d in enrich_source_lines(lines, wp, names, past) if d["opp_id"] in wp]
    assert [d["opp_id"] for d in delayed] == ["OPEN1"]            # only the open past-due line
    assert delayed[0]["amount"] == pytest.approx(62_000_000)
    assert delayed[0]["ym"] == "2026-05" and delayed[0]["win_pct"] == 83.0


def test_name_fallback():
    """enrich falls back to the line's own opp_name so a GUID-only opp still shows a label."""
    from predictive.so_forecast import enrich_source_lines
    lines = [
        {"opp_id": "OPP1", "opp_name": "Web Auto Call 2026", "ym": "2026-06", "amount": 62_000_000},
    ]
    out = enrich_source_lines(lines, {"OPP1": 0.8}, {}, {"2026-06"})
    assert out[0]["opp_name"] == "Web Auto Call 2026" and out[0]["account_name"] is None
