"""features.py — F2 validate_features + F3 build_opp_features.

Pure pandas, no live calls — fully offline-testable with synthetic fixtures.
Turns a raw Fact_Opportunity frame into the model's feature contract (schema.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from . import schema as S
from . import aux_features as AUX


@dataclass
class FeatureSet:
    """Output of build_opp_features."""
    X: pd.DataFrame          # FEATURE_COLUMNS, ready for the model
    y: pd.Series             # label 1/0, NaN for non-closed (inference) rows
    ids: pd.Series           # Opportunity ID aligned to X/y
    is_closed: pd.Series     # True where row is Won/Lost (usable for training)
    is_mature: pd.Series     # True where cohort old enough to be fully resolved


def mature_training_labels(fs: "FeatureSet") -> pd.Series:
    """Training labels = closed AND mature deals only. Immature closed deals are
    masked to NaN (excluded): their slow-resolving Lost peers haven't landed yet,
    so keeping them would right-censor the labels and inflate the win-rate."""
    return fs.y.where(fs.is_mature, other=np.nan)


def build_so_conversion_label(
    opps: pd.DataFrame,
    so_conversions: pd.DataFrame,
    asof: str | datetime,
    maturity_days: int = S.MATURITY_DAYS,
) -> pd.Series:
    """F3 — SO-conversion label for the CRM_PDT_AZ model, indexed like `opps`.

    converted = 1.0 if the opp produced a real Sales Order (opp_id present in the
        Fact_SalesOrder ledger) — this is the HARD truth signal and INCLUDES deals
        still flagged Status=Open that already booked an order (audit: 210 of them).
    converted = 0.0 if NO Sales Order AND the deal is resolved-negative: either
        Status=Lost, or the cohort is mature (created ≥ maturity_days before asof, so
        a Sales Order would have landed by now).
    converted = NaN  otherwise (no SO yet but the cohort is still immature →
        right-censored, could still convert) → excluded from training.

    The label is "ever converted" (presence of any SO); point-in-time discipline is
    enforced on FEATURES, not on this outcome label. maturity_days mirrors
    mature_training_labels so SO-less immature cohorts are not mislabeled negative."""
    asof_ts = pd.Timestamp(asof)
    opp_ids = opps[S.COL_OPP_ID].astype(str).str.strip()

    so_opps: set[str] = set()
    if so_conversions is not None and not so_conversions.empty:
        so_opps = set(so_conversions["opp_id"].astype(str).str.strip())
    has_so = opp_ids.isin(so_opps)

    status = (
        opps[S.COL_STATUS].astype(str).str.strip()
        if S.COL_STATUS in opps.columns
        else pd.Series("", index=opps.index)
    )
    is_lost = status.isin(S.LABEL_NEGATIVE)

    create_age = (
        _days_between(asof_ts, opps[S.COL_CREATE])
        if S.COL_CREATE in opps.columns
        else pd.Series(np.nan, index=opps.index)
    )
    is_mature = (create_age >= maturity_days).fillna(False)

    y = pd.Series(np.nan, index=opps.index, dtype="float64")
    y[~has_so & (is_lost | is_mature)] = 0.0
    y[has_so] = 1.0          # known conversion overrides — applied last
    return y


def validate_features(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """F2 — check the raw frame has what we need. Returns (ok, issues)."""
    issues: list[str] = []
    for col in S.REQUIRED_SOURCE_COLUMNS:
        if col not in df.columns:
            issues.append(f"missing required column: {col!r}")
    if df.empty:
        issues.append("frame is empty")
    return (len(issues) == 0, issues)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool_int(series: pd.Series) -> pd.Series:
    """Coerce Yes/No/True/False/1/0 → 1/0."""
    truthy = {"true", "yes", "y", "1", "hot", "true ", "1.0"}
    return (
        series.astype(str).str.strip().str.lower().isin(truthy).astype("int64")
    )


def _days_between(asof: pd.Timestamp, dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    return (asof - d).dt.days


def _label_from_status(status: pd.Series) -> pd.Series:
    s = status.astype(str).str.strip()
    y = pd.Series(np.nan, index=status.index, dtype="float64")
    y[s.isin(S.LABEL_POSITIVE)] = 1.0
    y[s.isin(S.LABEL_NEGATIVE)] = 0.0
    return y


def _reference_dates(df: pd.DataFrame, asof_ts: pd.Timestamp) -> pd.Series:
    """Per-deal point-in-time cutoff for auxiliary features.

    CLOSED deals: the date the deal resolved — prefer SO Actual Date, else Last
    Activity Date, else Create Date. OPEN deals: `asof` (we score them as they
    stand today). This is the line auxiliary features must not cross: only
    source records strictly before it can be seen, so a closed deal's features
    reflect only what was knowable while it was still in play (no leakage)."""
    status = df[S.COL_STATUS].astype(str).str.strip()
    is_closed = status.isin(S.LABEL_POSITIVE | S.LABEL_NEGATIVE)

    def _col(name):
        return pd.to_datetime(df[name], errors="coerce") if name in df.columns else pd.Series(pd.NaT, index=df.index)

    so_actual = _col(S.COL_SO_ACTUAL_DATE)
    last_act = _col(S.COL_LAST_ACT)
    create = _col(S.COL_CREATE)
    closed_ref = so_actual.fillna(last_act).fillna(create)

    ref = pd.Series(asof_ts, index=df.index)
    ref[is_closed] = closed_ref[is_closed]
    # any deal with no usable date falls back to asof
    return ref.fillna(asof_ts)


def build_opp_features(
    df: pd.DataFrame,
    asof: str | datetime,
    maturity_days: int = 0,
    activity: pd.DataFrame | None = None,
    movement: pd.DataFrame | None = None,
    accounts: pd.DataFrame | None = None,
    invoices: pd.DataFrame | None = None,
    income: pd.DataFrame | None = None,
    model_id: str | None = None,
) -> FeatureSet:
    """F3 — derive the model feature contract from a raw Fact_Opportunity frame.

    asof: reference date for recency/aging derivations (deterministic, testable).
    maturity_days: a deal is "mature" if created at least this many days before
        asof. Default 0 → all deals mature (backward compatible); the production
        pipeline passes S.MATURITY_DAYS to exclude right-censored cohorts.
    activity/movement/accounts/invoices: optional raw source frames for the four
        auxiliary feature groups. When omitted, those features default-fill (0 /
        'Unknown') so the contract and tests stay stable offline.
    income: optional Fact_IncomePlan SO-Plan income-line frame ([opp_id, ym,
        amount]). Consumed only by the CRM_PDT_MIX model (Group 5).
    model_id: selects the feature contract — CRM_PDT_BASE (default) excludes the
        income group; CRM_PDT_MIX includes it. See schema.feature_columns.
    """
    model_id = S.normalize_model_id(model_id)
    ok, issues = validate_features(df)
    if not ok:
        raise ValueError(f"feature validation failed: {issues}")

    asof_ts = pd.Timestamp(asof)
    out = pd.DataFrame(index=df.index)

    amount = _to_num(df.get(S.COL_AMOUNT, np.nan)).fillna(0.0).clip(lower=0)
    out["amount_log"] = np.log1p(amount)

    # aging/cycle: prefer explicit column, fill per-row gaps from Create Date.
    create_age = (
        _days_between(asof_ts, df[S.COL_CREATE])
        if S.COL_CREATE in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    out["aging_days"] = (
        _to_num(df[S.COL_AGING]).fillna(create_age)
        if S.COL_AGING in df.columns
        else create_age
    )

    if S.COL_LAST_ACT in df.columns:
        out["days_since_last_activity"] = _days_between(asof_ts, df[S.COL_LAST_ACT])
    else:
        out["days_since_last_activity"] = np.nan

    # hot flag: accept either column name.
    hot_col = S.COL_FLAG_HOT if S.COL_FLAG_HOT in df.columns else S.COL_HOT_OPP
    out["flag_hot"] = _to_bool_int(df[hot_col]) if hot_col in df.columns else 0

    out["solution"] = (
        df.get(S.COL_SOLUTION, pd.Series("Unknown", index=df.index))
        .fillna("Unknown").astype("category")
    )
    out["prospect_category"] = (
        df.get(S.COL_PROSPECT, pd.Series("Unknown", index=df.index))
        .fillna("Unknown").astype("category")
    )

    # --- auxiliary feature groups (point-in-time, default-filled) ---
    opp_ids = df[S.COL_OPP_ID].astype(str)
    acct_ids = (
        df.get(S.COL_ACCOUNT_ID, pd.Series("", index=df.index)).astype(str).str.strip()
    )
    ref_dates = _reference_dates(df, asof_ts)
    ref = pd.DataFrame(
        {"opp_id": opp_ids.to_numpy(), "acct_id": acct_ids.to_numpy(),
         "ref_date": ref_dates.to_numpy()}
    )
    def _attach(aux_df: pd.DataFrame, cols: list[str], default):
        # aux_df indexed by opp_id → align to df's row index.
        aligned = aux_df.reindex(opp_ids.to_numpy())
        for c in cols:
            vals = aligned[c].to_numpy() if c in aligned.columns else default
            out[c] = pd.Series(vals, index=out.index)

    act_df = AUX.build_activity_features(ref, activity)
    _attach(act_df, AUX.ACTIVITY_FEATURES, 0.0)

    # account history needs the full opp frame as its own source
    opp_hist = pd.DataFrame(
        {
            "opp_id": opp_ids.to_numpy(),
            "acct_id": acct_ids.to_numpy(),
            "status": df[S.COL_STATUS].astype(str).to_numpy(),
            "close_date": _reference_dates(df, asof_ts).to_numpy(),
            "amount": pd.to_numeric(df.get(S.COL_SO_ACTUAL_AMOUNT, df.get(S.COL_AMOUNT)),
                                    errors="coerce").to_numpy(),
        }
    )
    hist_df = AUX.build_history_features(ref, opp_hist, invoices)
    _attach(hist_df, AUX.HISTORY_FEATURES, 0.0)

    bant_df = AUX.build_bant_features(ref, movement)
    _attach(bant_df, AUX.BANT_FEATURES, 0.0)

    firmo_df = AUX.build_firmographic_features(ref, accounts)
    for c in AUX.FIRMO_CATEGORICALS:
        vals = firmo_df.reindex(opp_ids.to_numpy())[c].to_numpy()
        out[c] = pd.Series(vals, index=out.index).fillna("Unknown").astype("category")
    _attach(firmo_df, AUX.FIRMO_NUMERIC, 0.0)

    # Group 5: income-line structure (CRM_PDT_MIX only). Always derived so the
    # builder stays uniform; only selected into X for the INC model. The income
    # cutoff is SYMMETRIC across Won/Lost (Create Date + INCOME_CUTOFF_DAYS) — NOT
    # the close ref — so line PRESENCE can't proxy the label via an asymmetric
    # reference date (see schema.INCOME_CUTOFF_DAYS). Open deals → asof.
    if model_id == S.MODEL_MIX:
        status = df[S.COL_STATUS].astype(str).str.strip()
        is_closed = status.isin(S.LABEL_POSITIVE | S.LABEL_NEGATIVE)
        create = (
            pd.to_datetime(df[S.COL_CREATE], errors="coerce")
            if S.COL_CREATE in df.columns else pd.Series(pd.NaT, index=df.index)
        )
        income_cut = pd.Series(asof_ts, index=df.index)
        income_cut[is_closed] = (
            create[is_closed] + pd.Timedelta(days=S.INCOME_CUTOFF_DAYS)
        )
        income_cut = income_cut.fillna(asof_ts)
        income_ref = pd.DataFrame(
            {"opp_id": opp_ids.to_numpy(), "acct_id": acct_ids.to_numpy(),
             "ref_date": income_cut.to_numpy()}
        )
        income_df = AUX.build_income_plan_features(income_ref, income)
        _attach(income_df, AUX.INCOME_FEATURES, 0.0)

    X = out[S.feature_columns(model_id)].copy()
    y = _label_from_status(df[S.COL_STATUS])
    ids = df[S.COL_OPP_ID].astype(str)
    is_closed = y.notna()

    if maturity_days and maturity_days > 0:
        # NaN create_age (missing date) → not provably mature → treat as immature.
        is_mature = (create_age >= maturity_days).fillna(False)
    else:
        is_mature = pd.Series(True, index=df.index)

    return FeatureSet(X=X, y=y, ids=ids, is_closed=is_closed, is_mature=is_mature)
