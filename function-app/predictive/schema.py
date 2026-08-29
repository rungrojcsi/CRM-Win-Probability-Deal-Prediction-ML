"""schema.py — single source of truth for column names, label logic, feature spec.

Column names match the LIVE `SALES DATA MODEL` Fact_Opportunity (spaces preserved).
Label uses Status (Won/Lost) — NOT the `IsWon` column, which is a computed flag that
is set to 1 on ~400 Open deals (verified 2026-06-05) and would corrupt training.
"""

from __future__ import annotations

# --- source columns (live Fact_Opportunity) ---
COL_OPP_ID = "Opportunity ID"
COL_OPP_NAME = "Opportunity Name"
COL_ACCOUNT_NAME = "Account Name"
COL_STATUS = "Status"
# Use the (P) = project-price variant as the canonical deal size (Boss: "ใช้ P
# เป็นหลัก"; H = outsource, K = hardware/software). The unsuffixed "SO Plan
# Amount" = P+H+K combined, which over-states vs the CRM certified measures
# (verified 2026-06-05: base Jan–May SO Actual 244M vs certified (P) 197.9M; the
# (P) column raw sum 198.3M matches the certified measure).
COL_AMOUNT = "SO Plan Amount (P)"
COL_SO_ACTUAL_AMOUNT = "SO Actual Amount (P)"
COL_POSSIBILITY = "Possibility"
COL_PROGRESS = "Progress"
COL_AGING = "Aging Days"
COL_CYCLE = "Sales Cycle Days"
COL_FLAG_HOT = "Flag Hot"
COL_HOT_OPP = "Hot Opportunity"          # alternate hot column
COL_CREATE = "Create Date"
COL_LAST_ACT = "Last Activity Date"
COL_SO_PLAN_DATE = "SO Plan Date"   # planned month the deal converts to a sales order
COL_SOLUTION = "Solution Name"
COL_PROSPECT = "Prospect Category Name"
COL_ACCOUNT_ID = "Account ID"            # FK to Dim_Account, on Fact_Opportunity
COL_SO_ACTUAL_DATE = "SO Actual Date"    # realized close date (Won deals)

# --- Group 1: Activity engagement (Fact_Activity) ---
COL_ACT_DATE = "Activity Date"
COL_ACT_TYPE = "Activity Type"
COL_ACT_DURATION = "Duration (Mins)"
COL_ACT_OPP_ID = "Opportunity ID"        # link Fact_Activity → Fact_Opportunity

# --- Group 3: BANT + competitiveness (Fact_OpportunityMovement) ---
COL_MOV_OPP_ID = "Opportunity ID"
COL_MOV_MODIFIED = "Modified On"
COL_MOV_BUDGET = "Budget Score"
COL_MOV_AUTHORITY = "Authority Score"
COL_MOV_NEED = "Need Score"
COL_MOV_TIMING = "Timing Score"
COL_MOV_COMPETE = "Competitiveness Score"

# --- Group 5: Income-line structure (Fact_IncomePlan) ---
# Deal-level summary of the SO-Plan income LINES (project price spread across
# delivery months). LEAKAGE-SAFE by construction: only the SO-PLAN side + line
# structure is used. The SO Actual Amount / SO Actual Date / Invoice columns of
# Fact_IncomePlan are realized post-outcome and would leak the label — they are
# NEVER read here (see aux_features.build_income_plan_features).
COL_INC_OPP_ID = "Opportunity ID"
COL_INC_TYPE = "Income Type Label"        # "SO Plan" (used) vs realized variants (banned)
COL_INC_CODE = "SO Income Code Label"     # "P" = project price
COL_INC_PLAN_DATE = "SO Plan Date"        # planned delivery month of the line (plan side)
COL_INC_PLAN_AMOUNT = "SO Plan Amount"    # SO Plan (P) per line (plan side)

# --- Group 4: Account firmographics (Dim_Account) ---
COL_ACCT_KEY = "Account ID"
COL_ACCT_INDUSTRY_L1 = "Industry L1"
COL_ACCT_INDUSTRY_L2 = "Industry L2"
COL_ACCT_CUSTOMER_LEVEL = "Customer Level"
COL_ACCT_PROVINCE = "Province"
COL_ACCT_BIZ_SECTOR = "Biz Sector"
COL_ACCT_PARENT = "Parent Account ID"

# --- label definition (use Status, never IsWon) ---
LABEL_COL = COL_STATUS
STATUS_WON = "Won"
STATUS_LOST = "Lost"
LABEL_POSITIVE = {STATUS_WON}
LABEL_NEGATIVE = {STATUS_LOST}
# Statuses that are NOT closed → excluded from training, kept for inference.
STATUS_OPEN = "Open"

# --- feature spec (the model's input contract) ---
# NOTE: `Possibility` and `Progress` are intentionally EXCLUDED — verified label
# leakage (2026-06-05): Possibility == 100 for 100% of Won deals (set at close,
# not a leading signal); Progress is free-text and coerces to all-NaN. Including
# them gives a fake AUC ~0.99. Re-introduce only via an early-stage snapshot.
# `cycle_days` also EXCLUDED — partial leakage / distribution shift (2026-06-05):
# closed deals carry the FINAL realized cycle (Won 111 vs Lost 250 days) while
# Open deals show the still-running cycle (== aging, ~338) → not point-in-time
# correct. Re-introduce only as cycle-to-date from a snapshot.
# `flag_hot` REMOVED (2026-06-05): train/serving skew — verified constant 0 across
# ALL 1174 closed training deals (sales only flag Open deals), so it carried zero
# learnable signal yet SHAP attributed +1.5 to it on every Open deal, polluting the
# driver display, and the 72 Open deals with flag_hot=1 were scored on a value the
# model never saw (extrapolation). Re-introduce only if it is also populated on
# closed deals. The feature builder still derives it; it is simply not selected.
NUMERIC_FEATURES = [
    "amount_log",
    "aging_days",
    "days_since_last_activity",
    # Group 1: activity engagement (point-in-time, before reference date)
    "activity_count_30d",
    "activity_count_90d",
    "activity_count_total",
    "activity_trend",
    "total_duration_mins",
    "distinct_activity_types",
    "meeting_count",
    # Group 2: account history (prior closed deals before reference date).
    # TRIMMED 2026-06-06: only prior_won_count + is_repeat_buyer are selected.
    # account_historical_win_rate / days_since_last_purchase / avg_prior_deal_size /
    # prior_invoiced_total overfit and HURT out-of-sample AUC (−0.025 alone) — the
    # builder still derives them, they are simply not fed to the model.
    "prior_won_count",
    "is_repeat_buyer",
    # Group 3: BANT + competitiveness (latest snapshot at-or-before reference)
    "bant_total",
    "competitiveness_score",
    "bant_has_data",          # 1 if a real BANT snapshot was found, else 0
    # Group 4: firmographic flags (numeric)
    "has_parent_account",
]
CATEGORICAL_FEATURES = [
    "solution",
    "prospect_category",
    # Group 4: firmographic categoricals
    "industry_l1",
    "customer_level",
    "province",
    "biz_sector",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Group 5 income-line numeric features (INC model only) ---
# All point-in-time: a deal's SO-Plan income lines (project price + planned
# delivery schedule) are known while the deal is still Open. NO realized
# (SO Actual / Invoice) fields — see schema.py Group 5 + aux_features.
INCOME_FEATURES = [
    "income_line_count",       # number of SO-Plan (P) income lines for the deal
    "income_total_p_log",      # log1p(Σ SO Plan Amount (P) across lines)
    "income_line_month_spread",  # max−min plan-month distance (delivery schedule spread)
    "income_has_multi_line",   # 1 if >1 income line, else 0
]

# --- Model registry (serving contract). Two named models, selectable at serve
# time. CRM_PDT_BASE = the existing Fact_Opportunity + 4-aux model (unchanged).
# CRM_PDT_MIX = CRM_PDT_BASE features PLUS the Fact_IncomePlan income-line group.
MODEL_BASE = "CRM_PDT_BASE"
MODEL_MIX = "CRM_PDT_MIX"
# CRM_PDT_AZ = the new Sales-Order predictive. Stage A = SO-conversion (label from the
# real Fact_SalesOrder ledger, not Status=Won). Shares the CRM_PDT_BASE feature contract;
# the stored win_prob for this model = P(opp → a real Sales Order).
MODEL_AZ = "CRM_PDT_AZ"
MODEL_IDS = (MODEL_BASE, MODEL_MIX, MODEL_AZ)
DEFAULT_MODEL_ID = MODEL_BASE


def normalize_model_id(model_id: str | None) -> str:
    """Whitelist the serving `model` param → a known model id (default OPP)."""
    return model_id if model_id in MODEL_IDS else DEFAULT_MODEL_ID


# Income-line point-in-time window (days). For CLOSED training deals the income
# builder must use a cutoff that is SYMMETRIC between Won and Lost (the close
# reference is not: Won deals carry a real SO Actual Date, Lost deals fall back to
# Last Activity/Create, so a close-ref cutoff lets line PRESENCE proxy the label —
# verified 2026-06-06: AUC inflates to 0.92). Instead the income cutoff = the deal's
# Create Date + this window, identical for both classes and outcome-independent.
# Verified that with a symmetric Create+window cutoff, an early SO-Plan line still
# carries genuine signal (~67% win with vs ~26% without, stable 30–180d).
INCOME_CUTOFF_DAYS = 90


def feature_columns(model_id: str | None = None) -> list[str]:
    """The model's input contract. CRM_PDT_BASE excludes the income group;
    CRM_PDT_MIX appends it (numeric, after the base columns)."""
    if normalize_model_id(model_id) == MODEL_MIX:
        return NUMERIC_FEATURES + INCOME_FEATURES + CATEGORICAL_FEATURES
    return FEATURE_COLUMNS

# Minimum columns required to BUILD features (validation contract, F2).
REQUIRED_SOURCE_COLUMNS = [COL_OPP_ID, COL_STATUS, COL_AMOUNT, COL_POSSIBILITY]

# Cohort maturity window (days). Deals created more recently than this are
# RIGHT-CENSORED: Won deals resolve fast (cycle P50 63d) but Lost deals resolve
# slow (P50 188d, P90 539d), so a recent cohort's still-Open deals are
# disproportionately future-Lost. Training/eval on immature cohorts inflates the
# win-rate (~60% observed vs ~40% on fully-resolved cohorts) and drives the
# high-band overconfidence. Default = Lost P90 cycle (~539d) so even slow losses
# have had time to land. Verified against live SALES DATA MODEL 2026-06-05.
MATURITY_DAYS = 540

# Probability → band thresholds (G5 display).
BAND_HIGH = 0.70
BAND_LOW = 0.40


def band(prob: float) -> str:
    """Map a win probability to a display band."""
    if prob >= BAND_HIGH:
        return "High"
    if prob < BAND_LOW:
        return "Low"
    return "Mid"
