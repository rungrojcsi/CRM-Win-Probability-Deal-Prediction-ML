"""Tests for predictive.store (F12/F13), predictive.api (F14/F17),
and predictive.pipeline.build_score_rows — all offline via InMemoryScoreStore."""

from datetime import datetime, timezone

import pytest

from predictive.api import deal_detail, list_deals
from predictive.pipeline import build_score_rows
from predictive.store import InMemoryScoreStore

SCORED_AT = datetime(2026, 6, 5, tzinfo=timezone.utc)


def _row(opp, prob, band, amount, status="Open", so_plan_date=None, at_risk=False):
    return {
        "opp_id": opp, "win_prob": prob, "band": band,
        "drivers": [{"feature": "aging_days", "value": 1, "impact": 0.3}],
        "amount": amount, "status": status, "scored_at": SCORED_AT,
        "model_run_id": "run-1",
        "so_plan_date": so_plan_date, "at_risk": at_risk,
    }


@pytest.fixture
def store():
    s = InMemoryScoreStore()
    s.insert_model_run("run-1", "winprob-hgb", {"auc_cv": 0.88})
    s.upsert_scores([
        _row("A", 0.92, "High", 4_000_000),
        _row("B", 0.55, "Mid", 2_000_000),
        _row("C", 0.20, "Low", 7_000_000),
        _row("D", 0.85, "High", 1_000_000, status="Won"),
    ])
    return s


# --- store (F12/F13) ---

def test_upsert_is_idempotent(store):
    n = store.upsert_scores([_row("A", 0.99, "High", 4_000_000)])
    assert n == 1
    assert store.get_deal_score("A")["win_prob"] == 0.99  # overwritten, not duplicated


def test_get_latest_sorted_desc(store):
    rows = store.get_latest_scores()
    probs = [r["win_prob"] for r in rows]
    assert probs == sorted(probs, reverse=True)


def test_filter_by_status_and_band(store):
    assert {r["opp_id"] for r in store.get_latest_scores(status="Open")} == {"A", "B", "C"}
    assert {r["opp_id"] for r in store.get_latest_scores(status="Open", band="High")} == {"A"}


def test_filter_min_prob(store):
    assert {r["opp_id"] for r in store.get_latest_scores(min_prob=0.5)} == {"A", "B", "D"}


def test_get_deal_missing(store):
    assert store.get_deal_score("ZZZ") is None


# --- api (F14/F17) ---

def test_list_deals_default_open_and_weighted(store):
    out = list_deals(store)
    assert out["count"] == 3  # Open only (excludes Won D)
    # weighted = 0.92*4M + 0.55*2M + 0.20*7M
    assert out["weighted_pipeline"] == pytest.approx(0.92 * 4e6 + 0.55 * 2e6 + 0.20 * 7e6)
    assert out["deals"][0]["opp_id"] == "A"  # ranked
    assert out["deals"][0]["win_pct"] == 92.0


def test_list_deals_band_filter(store):
    out = list_deals(store, band="High")
    assert [d["opp_id"] for d in out["deals"]] == ["A"]


def test_list_deals_aggregates_over_all_not_limit(store):
    """Pipeline KPIs (count, weighted, raw, bands) must cover EVERY open deal —
    not just the limited page returned for the table. Guards the bug where the
    'pipeline' was summed over only the top-50 returned rows."""
    store.upsert_scores([
        _row("E", 0.95, "High", 1_000_000),
        _row("F", 0.10, "Low", 500_000),
    ])
    out = list_deals(store, limit=2)
    assert out["count"] == 5            # all Open (A,B,C,E,F) — NOT capped at limit
    assert out["returned"] == 2         # table page is capped
    assert len(out["deals"]) == 2
    raw_all = 4e6 + 2e6 + 7e6 + 1e6 + 0.5e6
    assert out["raw_pipeline"] == pytest.approx(raw_all)
    weighted_all = 0.92 * 4e6 + 0.55 * 2e6 + 0.20 * 7e6 + 0.95 * 1e6 + 0.10 * 0.5e6
    assert out["weighted_pipeline"] == pytest.approx(weighted_all)
    assert out["bands"] == {"High": 2, "Mid": 1, "Low": 2}


def test_list_deals_order_by_amount_keeps_big_lowprob(store):
    """order='amount' must page by amount so a BIG low-win-prob deal isn't hidden by the
    win-prob top-N cap (the Opportunity-pipeline bug). C has the largest amount (7M) but
    the lowest win-prob (0.20) — it must be #1 when ordered by amount, even at limit=1."""
    out = list_deals(store, limit=1, order_by="amount")
    assert out["deals"][0]["opp_id"] == "C"          # 7M, win 0.20 — biggest by amount
    out_wp = list_deals(store, limit=1)               # default win_prob order
    assert out_wp["deals"][0]["opp_id"] == "A"        # 0.92 — biggest by win-prob
    # aggregates unchanged by ordering
    assert out["weighted_pipeline"] == pytest.approx(out_wp["weighted_pipeline"])


def test_store_summarize_respects_filters(store):
    s = store.summarize(status="Open", band="High")
    assert s["count"] == 1 and s["bands"]["High"] == 1   # only A
    assert s["raw"] == pytest.approx(4e6)


# --- SO Plan month scope + at-risk ---

@pytest.fixture
def plan_store():
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("THIS1", 0.7, "High", 1_000_000, so_plan_date="2026-06-10"),
        _row("THIS2", 0.5, "Mid", 2_000_000, so_plan_date="2026-06-28"),
        _row("FUT1", 0.9, "High", 5_000_000, so_plan_date="2026-11-01"),
        _row("PAST1", 0.3, "Low", 4_000_000, so_plan_date="2026-02-01", at_risk=True),
    ])
    return s


def test_filter_by_so_plan_month(plan_store):
    rows = plan_store.get_latest_scores(so_plan_month="2026-06")
    assert {r["opp_id"] for r in rows} == {"THIS1", "THIS2"}


def test_summarize_so_plan_month_and_at_risk(plan_store):
    s = plan_store.summarize(so_plan_month="2026-06")
    assert s["count"] == 2
    assert s["raw"] == pytest.approx(3e6)
    # at_risk count is reported across the (unfiltered) set
    full = plan_store.summarize()
    assert full["at_risk"] == 1


def test_summarize_excludes_at_risk_from_clean_aggregates():
    """Past-due (at_risk) deals must NOT inflate the clean pipeline count/weighted/
    raw/bands — they belong in the separate at_risk bucket so the Current-Month KPIs
    reconcile with the Annual SO-forecast (which also drops at_risk)."""
    s = InMemoryScoreStore()
    s.upsert_scores([
        _row("OK1", 0.6, "Mid", 10_000_000, so_plan_date="2026-06-20"),
        _row("LATE1", 0.5, "Mid", 8_000_000, so_plan_date="2026-06-03", at_risk=True),
    ])
    out = s.summarize(so_plan_month="2026-06")
    assert out["count"] == 1                          # only OK1 in the clean pipeline
    assert out["raw"] == pytest.approx(10e6)          # LATE1 excluded
    assert out["weighted"] == pytest.approx(0.6 * 10e6)
    assert out["bands"] == {"High": 0, "Mid": 1, "Low": 0}
    assert out["at_risk"] == 1                         # LATE1 reported separately


def test_at_risk_filter_and_totals(plan_store):
    """Delay Prospects (Opportunity tab) = at_risk deals listed via at_risk=True; summary
    also reports at_risk_raw / at_risk_weighted over the at_risk set."""
    out = list_deals(plan_store, status=None, at_risk=True)
    assert [d["opp_id"] for d in out["deals"]] == ["PAST1"]      # only the past-due deal
    # totals across the whole set (unfiltered call)
    full = list_deals(plan_store, status=None)
    assert full["at_risk"] == 1
    assert full["at_risk_raw"] == pytest.approx(4e6)
    assert full["at_risk_weighted"] == pytest.approx(4e6 * 0.3)
    assert full["count"] == 3                                    # clean excludes PAST1


def test_list_deals_so_plan_month(plan_store):
    out = list_deals(plan_store, status=None, so_plan_month="2026-06")
    assert out["count"] == 2
    assert {d["opp_id"] for d in out["deals"]} == {"THIS1", "THIS2"}
    assert all("at_risk" in d and "so_plan_date" in d for d in out["deals"])


def test_deal_detail_found(store):
    d = deal_detail(store, "B")
    assert d["opp_id"] == "B" and d["band"] == "Mid"
    assert d["drivers"] and "feature" in d["drivers"][0]
    assert d["scored_at"] == SCORED_AT.isoformat()


def test_deal_detail_missing(store):
    assert deal_detail(store, "NOPE") is None


# --- pipeline assembler (F25) ---

def test_build_score_rows_shapes():
    rows = build_score_rows(
        ids=["X", "Y"],
        probs=[0.8, 0.3],
        drivers=[[{"feature": "aging_days", "value": 10, "impact": 0.2}], []],
        amounts=[1000.0, None],
        statuses=["Open", "Open"],
        run_id="run-9",
        scored_at=SCORED_AT,
    )
    assert rows[0]["band"] == "High" and rows[1]["band"] == "Low"
    assert rows[1]["amount"] is None
    assert all(r["model_run_id"] == "run-9" for r in rows)
