"""aux_features.py — point-in-time auxiliary feature builders for win-prob.

Each builder takes:
  - a `ref` frame: one row per opportunity with columns ['opp_id', 'acct_id', 'ref_date']
    where ref_date is the per-deal point-in-time cutoff (close ref for closed deals,
    `asof` for open deals). Features use ONLY source records STRICTLY BEFORE ref_date,
    so nothing that happened at/after a deal closed can leak into its features.
  - the relevant raw source frame(s).

All builders return a DataFrame indexed by opp_id with the new numeric/categorical
columns, default-filled so every opportunity gets a value (no NaN explosions, no
join holes). Pure pandas — fully offline-testable.

POINT-IN-TIME is the project's #1 lesson: for a CLOSED training deal we must only
see what a salesperson could have seen while the deal was still Open. The strict
`< ref_date` cutoff enforces that; the banned static `Possibility` column never
enters here — only the BANT *trajectory* snapshot at-or-before the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S


# --- Group 1: ACTIVITY ENGAGEMENT (Fact_Activity) --------------------------

ACTIVITY_FEATURES = [
    "activity_count_30d",
    "activity_count_90d",
    "activity_count_total",
    "activity_trend",
    "total_duration_mins",
    "distinct_activity_types",
    "meeting_count",
]

_MEETING_TYPES = {"appointment", "meeting", "service activity"}


def _zero_frame(opp_ids: pd.Series, cols: list[str], fill=0.0) -> pd.DataFrame:
    """Default frame: every opportunity present, all features = fill."""
    return pd.DataFrame(fill, index=pd.Index(opp_ids.astype(str), name="opp_id"), columns=cols)


def build_activity_features(ref: pd.DataFrame, activity: pd.DataFrame | None) -> pd.DataFrame:
    """Per-opportunity engagement counts from activities dated strictly before
    the deal's reference date. Missing/empty source → all zeros (a deal with no
    logged activity genuinely has zero engagement signal)."""
    out = _zero_frame(ref["opp_id"], ACTIVITY_FEATURES)
    if activity is None or activity.empty:
        return out

    a = activity.copy()
    a["opp_id"] = a.get(S.COL_ACT_OPP_ID, "").astype(str).str.strip()
    a = a[a["opp_id"] != ""]
    if a.empty:
        return out
    a["date"] = pd.to_datetime(a.get(S.COL_ACT_DATE), errors="coerce")
    a["dur"] = pd.to_numeric(a.get(S.COL_ACT_DURATION), errors="coerce").fillna(0.0)
    a["type"] = a.get(S.COL_ACT_TYPE, "").astype(str).str.strip()
    a = a.dropna(subset=["date"])

    refmap = ref.set_index(ref["opp_id"].astype(str))["ref_date"]
    refmap = pd.to_datetime(refmap, errors="coerce")

    a = a.join(refmap.rename("ref_date"), on="opp_id")
    a = a[a["ref_date"].notna() & (a["date"] < a["ref_date"])]  # strict point-in-time
    if a.empty:
        return out

    a["age_days"] = (a["ref_date"] - a["date"]).dt.days

    a["_is_30d"] = (a["age_days"] <= 30).astype(int)
    a["_is_90d"] = (a["age_days"] <= 90).astype(int)
    a["_is_3190d"] = ((a["age_days"] > 30) & (a["age_days"] <= 90)).astype(int)
    a["_is_meeting"] = a["type"].str.lower().isin(_MEETING_TYPES).astype(int)

    grp = a.groupby("opp_id")
    feats = pd.DataFrame(index=grp.size().index)
    feats["activity_count_total"] = grp.size()
    feats["activity_count_30d"] = grp["_is_30d"].sum()
    feats["activity_count_90d"] = grp["_is_90d"].sum()
    feats["total_duration_mins"] = grp["dur"].sum()
    feats["distinct_activity_types"] = grp["type"].nunique()
    feats["meeting_count"] = grp["_is_meeting"].sum()
    # trend: recent (≤30d) rate per day vs earlier (31–90d) rate per day. >0 = accelerating.
    recent_rate = feats["activity_count_30d"] / 30.0
    earlier_rate = grp["_is_3190d"].sum() / 60.0
    feats["activity_trend"] = (recent_rate - earlier_rate).astype(float)

    out.loc[feats.index, ACTIVITY_FEATURES] = feats[ACTIVITY_FEATURES].to_numpy()
    return out


# --- Group 2: ACCOUNT HISTORY ---------------------------------------------
# Built from the opportunity table itself (prior CLOSED deals for the same
# account, strictly before this deal's reference date) + invoice lifetime value
# joined via Opportunity ID. Fact_Invoice/SalesOrder have NO Account ID and no
# usable Dim_Account relationship (verified live 2026-06-05), so account-level
# invoiced totals are aggregated through the opportunity→account map.

HISTORY_FEATURES = [
    "prior_won_count",
    "prior_invoiced_total",
    "is_repeat_buyer",
    "days_since_last_purchase",
    "account_historical_win_rate",
    "avg_prior_deal_size",
]

_NO_PRIOR_PURCHASE = 9999.0  # sentinel for "never purchased before" (large recency)


def build_history_features(
    ref: pd.DataFrame,
    opp_hist: pd.DataFrame,
    invoices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """For each opportunity, summarize the account's PRIOR closed deals.

    `opp_hist` is the full opportunity frame with columns:
      opp_id, acct_id, status (Won/Lost/Open), close_date, amount.
    A prior deal counts only if it belongs to the same account AND closed
    strictly before this deal's reference date (point-in-time, no leakage —
    the deal's own outcome is never counted toward its own history).
    """
    out = _zero_frame(ref["opp_id"], HISTORY_FEATURES)
    out["days_since_last_purchase"] = _NO_PRIOR_PURCHASE
    if opp_hist is None or opp_hist.empty:
        return out

    h = opp_hist.copy()
    h["acct_id"] = h["acct_id"].astype(str).str.strip()
    h["status"] = h["status"].astype(str).str.strip()
    h["close_date"] = pd.to_datetime(h["close_date"], errors="coerce")
    h["amount"] = pd.to_numeric(h["amount"], errors="coerce").fillna(0.0)
    h = h[h["status"].isin([S.STATUS_WON, S.STATUS_LOST]) & h["close_date"].notna()]

    # invoice lifetime value per account (via opp→account map), point-in-time below.
    inv = None
    if invoices is not None and not invoices.empty:
        inv = invoices.copy()
        inv["opp_id"] = inv.get(S.COL_ACT_OPP_ID, "").astype(str).str.strip()
        inv["date"] = pd.to_datetime(inv.get("Created On"), errors="coerce")
        inv["amt"] = pd.to_numeric(inv.get("Grand Total"), errors="coerce").fillna(0.0)
        opp2acct = h.drop_duplicates("opp_id").set_index("opp_id")["acct_id"]
        inv = inv.join(opp2acct.rename("acct_id"), on="opp_id")
        inv = inv[inv["acct_id"].notna() & inv["date"].notna()]

    refmap = ref.set_index(ref["opp_id"].astype(str))
    refmap["ref_date"] = pd.to_datetime(refmap["ref_date"], errors="coerce")
    refmap["acct_id"] = refmap["acct_id"].astype(str).str.strip()

    # group prior deals by account once for speed
    by_acct = {k: v for k, v in h.groupby("acct_id")}
    inv_by_acct = {k: v for k, v in inv.groupby("acct_id")} if inv is not None else {}

    for opp_id, row in refmap.iterrows():
        acct = row["acct_id"]
        rd = row["ref_date"]
        if pd.isna(rd) or acct not in by_acct:
            continue
        prior = by_acct[acct]
        prior = prior[(prior["close_date"] < rd) & (prior["opp_id"].astype(str) != str(opp_id))]
        if prior.empty:
            continue
        won = prior[prior["status"] == S.STATUS_WON]
        n = len(prior)
        out.at[opp_id, "prior_won_count"] = float(len(won))
        out.at[opp_id, "account_historical_win_rate"] = float(len(won)) / n
        out.at[opp_id, "is_repeat_buyer"] = 1.0 if len(won) > 0 else 0.0
        if len(won) > 0:
            out.at[opp_id, "avg_prior_deal_size"] = float(won["amount"].mean())
            last = won["close_date"].max()
            out.at[opp_id, "days_since_last_purchase"] = float((rd - last).days)
        if acct in inv_by_acct:
            pi = inv_by_acct[acct]
            pi = pi[pi["date"] < rd]
            out.at[opp_id, "prior_invoiced_total"] = float(pi["amt"].sum())

    return out


# --- Group 3: BANT + COMPETITIVENESS (Fact_OpportunityMovement) ------------
# LOW-COVERAGE (verified live 2026-06-05): snapshots span only ~8 days
# (2026-05-28 → 2026-06-05) and just 671/3229 opps (20.8%) carry nonzero BANT.
# We implement the feature but default-fill missing deals and emit a
# `bant_has_data` flag so the model can learn to ignore the zero-filled mass.

BANT_FEATURES = ["bant_total", "competitiveness_score", "bant_has_data"]


def build_bant_features(ref: pd.DataFrame, movement: pd.DataFrame | None) -> pd.DataFrame:
    """Latest BANT snapshot at-or-before the reference date per opportunity.

    Uses Modified On to pick the most recent snapshot not after ref_date. The
    static `Possibility` column is NOT used (banned leakage); only the BANT
    trajectory scores, which are point-in-time by construction."""
    out = _zero_frame(ref["opp_id"], BANT_FEATURES)
    if movement is None or movement.empty:
        return out

    m = movement.copy()
    m["opp_id"] = m.get(S.COL_MOV_OPP_ID, "").astype(str).str.strip()
    m = m[m["opp_id"] != ""]
    if m.empty:
        return out
    m["mod"] = pd.to_datetime(m.get(S.COL_MOV_MODIFIED), errors="coerce")
    for c in (S.COL_MOV_BUDGET, S.COL_MOV_AUTHORITY, S.COL_MOV_NEED,
              S.COL_MOV_TIMING, S.COL_MOV_COMPETE):
        m[c] = pd.to_numeric(m.get(c), errors="coerce").fillna(0.0)
    m = m.dropna(subset=["mod"])

    refmap = pd.to_datetime(
        ref.set_index(ref["opp_id"].astype(str))["ref_date"], errors="coerce"
    )
    m = m.join(refmap.rename("ref_date"), on="opp_id")
    m = m[m["ref_date"].notna() & (m["mod"] <= m["ref_date"])]  # at-or-before cutoff
    if m.empty:
        return out

    # latest snapshot per opportunity
    m = m.sort_values("mod").groupby("opp_id").tail(1).set_index("opp_id")
    bant_total = (
        m[S.COL_MOV_BUDGET] + m[S.COL_MOV_AUTHORITY]
        + m[S.COL_MOV_NEED] + m[S.COL_MOV_TIMING]
    )
    out.loc[m.index, "bant_total"] = bant_total.to_numpy()
    out.loc[m.index, "competitiveness_score"] = m[S.COL_MOV_COMPETE].to_numpy()
    out.loc[m.index, "bant_has_data"] = 1.0
    return out


# --- Group 5: INCOME-LINE STRUCTURE (Fact_IncomePlan) ----------------------
# Deal-level summary of the SO-Plan income LINES (the project price spread across
# planned delivery months). Used ONLY by the CRM_PDT_MIX model.
#
# CRITICAL LEAKAGE RULE: only the SO-PLAN side + line structure is consumed here.
# Fact_IncomePlan ALSO carries realized columns (SO Actual Amount, SO Actual Date,
# Invoice fields) that are populated post-outcome and would leak the Won/Lost
# label. Those are NEVER read — the income source frame is pre-filtered upstream
# (ingest.fetch_income_plan_so_lines: Income Type Label = "SO Plan", Code = "P")
# and this builder touches only [opp_id, ym (plan month), amount (SO Plan (P))].
# Every feature is computable while the deal is still OPEN (point-in-time).

INCOME_FEATURES = [
    "income_line_count",
    "income_total_p_log",
    "income_line_month_spread",
    "income_has_multi_line",
]


def build_income_plan_features(ref: pd.DataFrame, income: pd.DataFrame | None) -> pd.DataFrame:
    """Per-opportunity SO-Plan income-line structure.

    `income` columns (plan side only): [opp_id, ym ('YYYY-MM' plan month), amount]
    and OPTIONALLY `created` (the line's Created On). A deal with no income lines
    genuinely has zero structure signal → default 0. `income_line_month_spread` =
    (max − min) plan-month index across the deal's lines = how spread its delivery
    schedule is (0 if single month / single line).

    POINT-IN-TIME (verified necessary, 2026-06-06): if `created` is provided we keep
    ONLY lines created STRICTLY BEFORE the deal's ref_date. Verified on live data:
    62.6% of SO-Plan (P) lines on LOST deals are created AFTER the deal's close
    reference (vs 8.4% on Won), so counting post-close lines would let the *outcome*
    drive the feature (a leakage path — Won deals get their plan lines ~41d before
    close, losers get them logged at/after the deal dies). The strict cutoff makes
    line PRESENCE a genuine leading signal (a salesperson committing a SO plan while
    the deal is still Open) rather than a post-mortem artifact. When `created` is
    absent (e.g. open-deal scoring where ref_date = asof = today), all lines pass.
    """
    out = _zero_frame(ref["opp_id"], INCOME_FEATURES)
    if income is None or income.empty:
        return out

    inc = income.copy()
    inc["opp_id"] = inc.get("opp_id", "").astype(str).str.strip()
    inc = inc[inc["opp_id"] != ""]
    if inc.empty:
        return out
    inc["amount"] = pd.to_numeric(inc.get("amount"), errors="coerce").fillna(0.0)

    # strict point-in-time cutoff on line creation (only if Created On is available)
    if "created" in inc.columns:
        inc["created"] = pd.to_datetime(inc["created"], errors="coerce")
        refmap = pd.to_datetime(
            ref.set_index(ref["opp_id"].astype(str))["ref_date"], errors="coerce"
        )
        inc = inc.join(refmap.rename("_ref"), on="opp_id")
        # keep a line if its ref is unknown (open/asof) OR it was created before ref
        inc = inc[inc["_ref"].isna() | inc["created"].isna() | (inc["created"] < inc["_ref"])]
        if inc.empty:
            return out
    # plan month → integer index (year*12+month) for a spread distance.
    ym = inc.get("ym", "").astype(str).str.strip()
    parts = ym.str.split("-", n=1, expand=True)
    yr = pd.to_numeric(parts[0], errors="coerce")
    mo = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
    inc["month_idx"] = (yr * 12 + mo)

    grp = inc.groupby("opp_id")
    feats = pd.DataFrame(index=grp.size().index)
    feats["income_line_count"] = grp.size().astype(float)
    feats["income_total_p_log"] = np.log1p(grp["amount"].sum().clip(lower=0))
    spread = (grp["month_idx"].max() - grp["month_idx"].min())
    feats["income_line_month_spread"] = spread.fillna(0.0).astype(float)
    feats["income_has_multi_line"] = (feats["income_line_count"] > 1).astype(float)

    # income lines can reference opps not in this feature frame (closed/other-year);
    # only assign rows that exist in `out` (the rest stay default 0).
    feats = feats.loc[feats.index.intersection(out.index)]
    keep = [c for c in INCOME_FEATURES if c in feats.columns]
    out.loc[feats.index, keep] = feats[keep].to_numpy()
    return out


# --- Group 4: ACCOUNT FIRMOGRAPHIC (Dim_Account) ---------------------------

FIRMO_CATEGORICALS = ["industry_l1", "customer_level", "province", "biz_sector"]
FIRMO_NUMERIC = ["has_parent_account"]


def build_firmographic_features(ref: pd.DataFrame, accounts: pd.DataFrame | None) -> pd.DataFrame:
    """Static firmographics joined opportunity→account. Categoricals default to
    'Unknown', has_parent_account to 0. Static (no time component) so no
    point-in-time concern beyond using the account's current attributes."""
    idx = pd.Index(ref["opp_id"].astype(str), name="opp_id")
    out = pd.DataFrame(index=idx)
    for c in FIRMO_CATEGORICALS:
        out[c] = "Unknown"
    out["has_parent_account"] = 0.0

    if accounts is None or accounts.empty:
        return out

    acc = accounts.copy()
    acc[S.COL_ACCT_KEY] = acc[S.COL_ACCT_KEY].astype(str).str.strip()
    acc = acc.drop_duplicates(S.COL_ACCT_KEY).set_index(S.COL_ACCT_KEY)

    def _clean_cat(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        return s.where((s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none"), "Unknown")

    colmap = {
        "industry_l1": S.COL_ACCT_INDUSTRY_L1,
        "customer_level": S.COL_ACCT_CUSTOMER_LEVEL,
        "province": S.COL_ACCT_PROVINCE,
        "biz_sector": S.COL_ACCT_BIZ_SECTOR,
    }
    acct_ids = ref["acct_id"].astype(str).str.strip().to_numpy()
    for feat, src in colmap.items():
        if src in acc.columns:
            vals = _clean_cat(acc[src]).reindex(acct_ids).to_numpy()
            out[feat] = pd.Series(vals, index=idx).fillna("Unknown")

    if S.COL_ACCT_PARENT in acc.columns:
        parent = acc[S.COL_ACCT_PARENT].astype(str).str.strip()
        has_parent = ((parent != "") & (parent.str.lower() != "nan") & (parent.str.lower() != "none")).astype(float)
        out["has_parent_account"] = pd.Series(
            has_parent.reindex(acct_ids).to_numpy(), index=idx
        ).fillna(0.0)

    return out
