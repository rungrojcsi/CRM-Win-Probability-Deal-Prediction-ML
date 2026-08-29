"""Unit tests for the 4 auxiliary win-prob feature groups (predictive.aux_features)
and their point-in-time wiring through build_opp_features.

Each group is tested for: correct values, strict point-in-time cutoff (no leakage
of post-reference-date records), and graceful default-fill when the source frame
is absent.
"""

import numpy as np
import pandas as pd
import pytest

from predictive import schema as S
from predictive import aux_features as AUX
from predictive.features import build_opp_features

ASOF = "2026-06-05"


def _ref(rows):
    """rows: list of (opp_id, acct_id, ref_date)."""
    return pd.DataFrame(rows, columns=["opp_id", "acct_id", "ref_date"]).assign(
        ref_date=lambda d: pd.to_datetime(d["ref_date"])
    )


# --- Group 1: ACTIVITY ENGAGEMENT -----------------------------------------

def test_activity_counts_point_in_time():
    ref = _ref([("O1", "A1", "2026-03-01")])
    activity = pd.DataFrame(
        {
            S.COL_ACT_OPP_ID: ["O1"] * 4,
            S.COL_ACT_DATE: ["2026-02-20", "2026-02-01", "2025-12-15", "2026-03-05"],
            S.COL_ACT_TYPE: ["Email", "Appointment", "Phone Call", "Email"],
            S.COL_ACT_DURATION: [10, 60, 15, 99],
        }
    )
    out = AUX.build_activity_features(ref, activity)
    r = out.loc["O1"]
    # 2026-03-05 is AFTER ref 2026-03-01 → must be excluded (no leakage)
    assert r["activity_count_total"] == 3
    # within 30d of 03-01 (i.e. on/after 01-30): 02-20 (age 9) + 02-01 (age 28) = 2
    assert r["activity_count_30d"] == 2
    assert r["activity_count_90d"] == 3          # all three within 90d
    assert r["total_duration_mins"] == 85        # 10+60+15, the 99 is excluded
    assert r["distinct_activity_types"] == 3
    assert r["meeting_count"] == 1               # the Appointment


def test_activity_missing_source_defaults_zero():
    ref = _ref([("O1", "A1", "2026-03-01")])
    out = AUX.build_activity_features(ref, None)
    assert (out.loc["O1", AUX.ACTIVITY_FEATURES] == 0).all()


def test_activity_unlinked_rows_ignored():
    ref = _ref([("O1", "A1", "2026-03-01")])
    activity = pd.DataFrame(
        {S.COL_ACT_OPP_ID: ["", "O1"], S.COL_ACT_DATE: ["2026-02-01", "2026-02-01"],
         S.COL_ACT_TYPE: ["Email", "Email"], S.COL_ACT_DURATION: [5, 5]}
    )
    out = AUX.build_activity_features(ref, activity)
    assert out.loc["O1", "activity_count_total"] == 1


# --- Group 2: ACCOUNT HISTORY ---------------------------------------------

def test_history_prior_deals_point_in_time():
    ref = _ref([("O3", "ACC", "2026-01-01")])
    opp_hist = pd.DataFrame(
        {
            "opp_id": ["O1", "O2", "O3", "O4"],
            "acct_id": ["ACC", "ACC", "ACC", "ACC"],
            "status": ["Won", "Lost", "Open", "Won"],
            "close_date": ["2025-06-01", "2025-09-01", None, "2026-05-01"],
            "amount": [1_000_000, 500_000, 0, 9_000_000],
        }
    )
    out = AUX.build_history_features(ref, opp_hist)
    r = out.loc["O3"]
    # prior closed deals for ACC before 2026-01-01: O1 (Won), O2 (Lost).
    # O4 closes AFTER ref → excluded (no leakage). O3 is self → excluded.
    assert r["prior_won_count"] == 1
    assert r["account_historical_win_rate"] == pytest.approx(0.5)
    assert r["is_repeat_buyer"] == 1.0
    assert r["avg_prior_deal_size"] == pytest.approx(1_000_000)
    assert r["days_since_last_purchase"] == (pd.Timestamp("2026-01-01") - pd.Timestamp("2025-06-01")).days


def test_history_no_prior_is_cold():
    ref = _ref([("O1", "NEWACC", "2026-01-01")])
    opp_hist = pd.DataFrame(
        {"opp_id": ["O1"], "acct_id": ["NEWACC"], "status": ["Open"],
         "close_date": [None], "amount": [0]}
    )
    out = AUX.build_history_features(ref, opp_hist)
    assert out.loc["O1", "prior_won_count"] == 0
    assert out.loc["O1", "is_repeat_buyer"] == 0.0
    assert out.loc["O1", "days_since_last_purchase"] == AUX._NO_PRIOR_PURCHASE


def test_history_invoice_lifetime_value():
    ref = _ref([("O2", "ACC", "2026-01-01")])
    opp_hist = pd.DataFrame(
        {"opp_id": ["O1", "O2"], "acct_id": ["ACC", "ACC"],
         "status": ["Won", "Open"], "close_date": ["2025-06-01", None],
         "amount": [1_000_000, 0]}
    )
    invoices = pd.DataFrame(
        {S.COL_ACT_OPP_ID: ["O1", "O1"], "Created On": ["2025-07-01", "2026-05-01"],
         "Grand Total": [300_000, 700_000]}
    )
    out = AUX.build_history_features(ref, opp_hist, invoices)
    # only the 2025-07-01 invoice is before ref 2026-01-01
    assert out.loc["O2", "prior_invoiced_total"] == pytest.approx(300_000)


# --- Group 3: BANT --------------------------------------------------------

def test_bant_latest_snapshot_at_or_before_ref():
    ref = _ref([("O1", "A1", "2026-06-01")])
    movement = pd.DataFrame(
        {
            S.COL_MOV_OPP_ID: ["O1", "O1", "O1"],
            S.COL_MOV_MODIFIED: ["2026-05-20", "2026-05-30", "2026-06-03"],
            S.COL_MOV_BUDGET: [1, 2, 9], S.COL_MOV_AUTHORITY: [1, 2, 9],
            S.COL_MOV_NEED: [1, 2, 9], S.COL_MOV_TIMING: [1, 2, 9],
            S.COL_MOV_COMPETE: [1, 5, 9],
        }
    )
    out = AUX.build_bant_features(ref, movement)
    r = out.loc["O1"]
    # latest at-or-before 2026-06-01 = the 2026-05-30 snapshot (2,2,2,2 / compete 5)
    assert r["bant_total"] == 8
    assert r["competitiveness_score"] == 5
    assert r["bant_has_data"] == 1.0


def test_bant_missing_defaults_and_flag_zero():
    ref = _ref([("O1", "A1", "2026-06-01")])
    out = AUX.build_bant_features(ref, None)
    assert out.loc["O1", "bant_has_data"] == 0.0
    assert out.loc["O1", "bant_total"] == 0.0


# --- Group 4: FIRMOGRAPHIC ------------------------------------------------

def test_firmographic_join_and_flags():
    ref = _ref([("O1", "ACC1", "2026-06-01"), ("O2", "MISSING", "2026-06-01")])
    accounts = pd.DataFrame(
        {
            S.COL_ACCT_KEY: ["ACC1"],
            S.COL_ACCT_INDUSTRY_L1: ["Manufacturing"],
            S.COL_ACCT_CUSTOMER_LEVEL: ["A"],
            S.COL_ACCT_PROVINCE: ["Bangkok"],
            S.COL_ACCT_BIZ_SECTOR: ["Auto"],
            S.COL_ACCT_PARENT: ["PARENT-X"],
        }
    )
    out = AUX.build_firmographic_features(ref, accounts)
    assert out.loc["O1", "industry_l1"] == "Manufacturing"
    assert out.loc["O1", "has_parent_account"] == 1.0
    # account not in Dim_Account → Unknown / no parent
    assert out.loc["O2", "industry_l1"] == "Unknown"
    assert out.loc["O2", "has_parent_account"] == 0.0


def test_firmographic_blank_becomes_unknown():
    ref = _ref([("O1", "ACC1", "2026-06-01")])
    accounts = pd.DataFrame(
        {S.COL_ACCT_KEY: ["ACC1"], S.COL_ACCT_INDUSTRY_L1: [""],
         S.COL_ACCT_CUSTOMER_LEVEL: ["nan"], S.COL_ACCT_PROVINCE: ["Bangkok"],
         S.COL_ACCT_BIZ_SECTOR: [""], S.COL_ACCT_PARENT: [""]}
    )
    out = AUX.build_firmographic_features(ref, accounts)
    assert out.loc["O1", "industry_l1"] == "Unknown"
    assert out.loc["O1", "customer_level"] == "Unknown"
    assert out.loc["O1", "province"] == "Bangkok"
    assert out.loc["O1", "has_parent_account"] == 0.0


# --- integration through build_opp_features --------------------------------

def test_build_opp_features_attaches_all_groups():
    df = pd.DataFrame(
        {
            S.COL_OPP_ID: ["O1", "O2"],
            S.COL_ACCOUNT_ID: ["ACC1", "ACC1"],
            S.COL_STATUS: ["Won", "Open"],
            S.COL_AMOUNT: [1_000_000, 2_000_000],
            S.COL_POSSIBILITY: [80, 50],
            S.COL_CREATE: ["2025-01-01", "2026-01-01"],
            S.COL_LAST_ACT: ["2025-06-01", "2026-05-01"],
            S.COL_SO_ACTUAL_DATE: ["2025-06-15", None],
        }
    )
    activity = pd.DataFrame(
        {S.COL_ACT_OPP_ID: ["O2"], S.COL_ACT_DATE: ["2026-03-01"],
         S.COL_ACT_TYPE: ["Appointment"], S.COL_ACT_DURATION: [60]}
    )
    accounts = pd.DataFrame(
        {S.COL_ACCT_KEY: ["ACC1"], S.COL_ACCT_INDUSTRY_L1: ["Manufacturing"],
         S.COL_ACCT_CUSTOMER_LEVEL: ["A"], S.COL_ACCT_PROVINCE: ["Bangkok"],
         S.COL_ACCT_BIZ_SECTOR: ["Auto"], S.COL_ACCT_PARENT: [""]}
    )
    fs = build_opp_features(df, asof=ASOF, activity=activity, accounts=accounts)
    assert list(fs.X.columns) == S.FEATURE_COLUMNS
    o2 = fs.X[fs.ids.values == "O2"].iloc[0]
    assert o2["meeting_count"] == 1
    assert o2["industry_l1"] == "Manufacturing"
    # O2 is the same account as the prior Won O1 (closed 2025-06-15 < O2 ref asof)
    assert o2["prior_won_count"] == 1
    assert o2["is_repeat_buyer"] == 1.0


def test_no_leakage_closed_deal_uses_close_ref(monkeypatch):
    """A closed deal must NOT see activity logged after it closed."""
    df = pd.DataFrame(
        {
            S.COL_OPP_ID: ["W1"], S.COL_ACCOUNT_ID: ["ACC1"], S.COL_STATUS: ["Won"],
            S.COL_AMOUNT: [1_000_000], S.COL_POSSIBILITY: [80],
            S.COL_CREATE: ["2025-01-01"], S.COL_SO_ACTUAL_DATE: ["2025-06-01"],
        }
    )
    activity = pd.DataFrame(
        {S.COL_ACT_OPP_ID: ["W1", "W1"], S.COL_ACT_DATE: ["2025-05-01", "2025-08-01"],
         S.COL_ACT_TYPE: ["Email", "Email"], S.COL_ACT_DURATION: [10, 10]}
    )
    fs = build_opp_features(df, asof=ASOF, activity=activity)
    # ref_date = SO Actual Date 2025-06-01 → only the 2025-05-01 activity counts.
    assert fs.X.iloc[0]["activity_count_total"] == 1
