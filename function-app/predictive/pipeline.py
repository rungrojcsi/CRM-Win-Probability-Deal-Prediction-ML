"""pipeline.py — F25 run_scoring (daily orchestration).

ingest → features → train → score → explain → persist. Writes a model-run row
(F24a) and upserts per-deal scores for OPEN deals into the store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from . import schema as S
from .features import build_opp_features, mature_training_labels
from .ingest import (
    fetch_accounts,
    fetch_activities,
    fetch_income_plan_so_lines,
    fetch_incomeplan_realization,
    fetch_invoice_history,
    fetch_movements,
    fetch_opportunities,
    fetch_so_actual_by_month,
    fetch_so_conversions,
)
from .winprob import backtest_winprob, explain_winprob, score_winprob, train_winprob

MODEL_TYPE = "winprob-hgb"


def _fetch_aux_sources(dataset_id, asof: str | None = None, model_id: str | None = None):
    """Best-effort pull of the auxiliary feature sources. Any source that fails
    to load returns None so build_opp_features default-fills its group — scoring
    must never crash because one optional source is unavailable.

    For the CRM_PDT_MIX model, also pull the Fact_IncomePlan SO-Plan income lines
    (Group 5). Income lines are date-keyed per year, so we pull the asof year plus
    the prior year to cover deals whose lines straddle a year boundary."""
    sources = {}
    for name, fn in (
        ("activity", fetch_activities),
        ("movement", fetch_movements),
        ("accounts", fetch_accounts),
        ("invoices", fetch_invoice_history),
    ):
        try:
            sources[name] = fn(dataset_id)
        except Exception:
            sources[name] = None
    if model_id == S.MODEL_MIX:
        sources["income"] = _fetch_income_lines(dataset_id, asof)
    return sources


def _fetch_income_lines(dataset_id, asof: str | None):
    """Group 5 source: SO-Plan (P) income lines across the asof year and the prior
    year (deals' delivery schedules can span years). Plan side only — leakage-safe."""
    import pandas as pd

    year = pd.Timestamp(asof).year if asof else pd.Timestamp.utcnow().year
    frames = []
    for yr in (year - 1, year):
        try:
            df = fetch_income_plan_so_lines(yr, dataset_id)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _clean_name(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def build_score_rows(
    ids, probs, drivers, amounts, statuses, run_id, scored_at,
    opp_names=None, account_names=None, so_plan_dates=None, at_risks=None,
    model_id=S.DEFAULT_MODEL_ID,
) -> list[dict]:
    """Pure assembler — turns parallel arrays into store rows. Unit-testable."""
    rows = []
    for i in range(len(ids)):
        p = float(probs[i])
        rows.append(
            {
                "opp_id": str(ids[i]),
                "opp_name": _clean_name(opp_names[i]) if opp_names is not None else None,
                "account_name": _clean_name(account_names[i]) if account_names is not None else None,
                "win_prob": p,
                "band": S.band(p),
                "drivers": drivers[i],
                "amount": None if amounts[i] is None or pd.isna(amounts[i]) else float(amounts[i]),
                "status": statuses[i],
                "so_plan_date": so_plan_dates[i] if so_plan_dates is not None else None,
                "at_risk": bool(at_risks[i]) if at_risks is not None else False,
                "scored_at": scored_at,
                "model_run_id": run_id,
                "model_id": S.normalize_model_id(model_id),
            }
        )
    return rows


def run_backtest(
    store,
    asof: str,
    test_frac: float = 0.30,
    maturity_days: int = S.MATURITY_DAYS,
    dataset_id: str | None = None,
    model_id: str | None = None,
) -> dict:
    """Temporal holdout backtest on historical CLOSED deals, maturity-filtered.

    Restricts to MATURE create-cohorts (created ≥ maturity_days before asof) so
    right-censored recent deals don't inflate the win-rate, then orders by Create
    Date and trains on the earliest (1 - test_frac), evaluating the most recent
    `test_frac`. Pass maturity_days=0 to evaluate every closed deal. `store` is
    unused (kept for route-signature parity).

    NOTE: features are still built at a single `asof`, so aging-based inputs are
    NOT fully point-in-time per deal — same caveat in schema.py. The temporal
    split removes label look-ahead; per-deal snapshots would be the full fix.
    """
    model_id = S.normalize_model_id(model_id)
    df = fetch_opportunities(dataset_id)
    aux = _fetch_aux_sources(dataset_id, asof=asof, model_id=model_id)
    fs = build_opp_features(df, asof=asof, maturity_days=maturity_days,
                           model_id=model_id, **aux)

    create = pd.to_datetime(df.get(S.COL_CREATE), errors="coerce")
    create.index = fs.X.index
    mask = fs.is_mature.to_numpy()

    res = backtest_winprob(
        fs.X[mask], fs.y[mask], order=create[mask], test_frac=test_frac
    )
    res["order_by"] = S.COL_CREATE
    res["asof"] = asof
    res["test_frac"] = test_frac
    res["maturity_days"] = maturity_days
    res["model_id"] = model_id
    res["n_mature_closed"] = int((fs.is_mature & fs.is_closed).sum())
    return res


def run_scoring(
    store,
    asof: str,
    maturity_days: int = S.MATURITY_DAYS,
    dataset_id: str | None = None,
    model_id: str | None = None,
) -> dict:
    """F25 — full daily run. Returns a summary dict.

    Trains on MATURE closed deals only (maturity_days, see schema.py) to avoid
    right-censored win-rate inflation, then scores ALL open deals. `model_id`
    selects the feature contract (CRM_PDT_BASE default | CRM_PDT_MIX) and tags
    every persisted score + the model_run with that id so both coexist.
    """
    model_id = S.normalize_model_id(model_id)
    df = fetch_opportunities(dataset_id)
    aux = _fetch_aux_sources(dataset_id, asof=asof, model_id=model_id)
    fs = build_opp_features(df, asof=asof, maturity_days=maturity_days,
                           model_id=model_id, **aux)

    wm = train_winprob(fs.X, mature_training_labels(fs))
    run_id = f"{model_id}-{asof}-{uuid4().hex[:8]}"
    scored_at = datetime.now(timezone.utc)

    store.ensure_schema()
    store.insert_model_run(run_id, model_id, wm.metrics)  # F24a log_metrics

    open_mask = (~fs.is_closed).to_numpy()
    X_open = fs.X[open_mask]
    probs = score_winprob(wm, X_open)
    drivers = explain_winprob(wm, X_open, top_n=None)
    ids = fs.ids[open_mask].to_numpy()
    amounts = pd.to_numeric(df[S.COL_AMOUNT], errors="coerce").to_numpy()[open_mask]
    statuses = df[S.COL_STATUS].astype(str).to_numpy()[open_mask]
    opp_names = df.get(S.COL_OPP_NAME, pd.Series(index=df.index, dtype=object)).to_numpy()[open_mask]
    account_names = df.get(S.COL_ACCOUNT_NAME, pd.Series(index=df.index, dtype=object)).to_numpy()[open_mask]

    # SO Plan timing: the month the deal is planned to convert. at_risk = an open
    # deal whose planned month is already in the past (slipping / likely to fade).
    so_plan = pd.to_datetime(df.get(S.COL_SO_PLAN_DATE), errors="coerce")
    month_start = pd.Timestamp(asof).to_period("M").to_timestamp()
    so_plan_open = so_plan.to_numpy()[open_mask]
    so_plan_dates = [
        None if pd.isna(d) else pd.Timestamp(d).date().isoformat() for d in so_plan_open
    ]
    at_risks = [(not pd.isna(d)) and pd.Timestamp(d) < month_start for d in so_plan_open]

    rows = build_score_rows(
        ids, probs, drivers, amounts, statuses, run_id, scored_at,
        opp_names=opp_names, account_names=account_names,
        so_plan_dates=so_plan_dates, at_risks=at_risks, model_id=model_id,
    )
    written = store.upsert_scores(rows)

    bands = {b: int(np.sum([r["band"] == b for r in rows])) for b in ("High", "Mid", "Low")}
    return {
        "run_id": run_id,
        "model_id": model_id,
        "metrics": wm.metrics,
        "scored": written,
        "bands": bands,
        "scored_at": scored_at.isoformat(),
        "maturity_days": maturity_days,
        "n_mature_closed": int((fs.is_mature & fs.is_closed).sum()),
    }


def run_so_scoring(
    store,
    asof: str,
    maturity_days: int = S.MATURITY_DAYS,
    dataset_id: str | None = None,
) -> dict:
    """F10 — score + persist the SO-conversion model (model_id=CRM_PDT_AZ).

    Same orchestration as run_scoring but the label comes from the real
    Fact_SalesOrder ledger (build_so_conversion_label) instead of Status=Won, so it
    learns from the Open deals that already booked an order. Feature contract =
    CRM_PDT_BASE; the stored win_prob = P(opp → a Sales Order). Scores OPEN deals."""
    from .features import build_so_conversion_label

    df = fetch_opportunities(dataset_id)
    so_conv = fetch_so_conversions(dataset_id)
    aux = _fetch_aux_sources(dataset_id, asof=asof, model_id=S.MODEL_BASE)
    fs = build_opp_features(df, asof=asof, maturity_days=maturity_days,
                            model_id=S.MODEL_BASE, **aux)
    y = build_so_conversion_label(df, so_conv, asof, maturity_days=maturity_days)

    wm = train_winprob(fs.X, y)
    run_id = f"{S.MODEL_AZ}-{asof}-{uuid4().hex[:8]}"
    scored_at = datetime.now(timezone.utc)

    store.ensure_schema()
    store.insert_model_run(run_id, S.MODEL_AZ, wm.metrics)

    open_mask = (~fs.is_closed).to_numpy()
    X_open = fs.X[open_mask]
    probs = score_winprob(wm, X_open)
    drivers = explain_winprob(wm, X_open, top_n=None)
    ids = fs.ids[open_mask].to_numpy()
    amounts = pd.to_numeric(df[S.COL_AMOUNT], errors="coerce").to_numpy()[open_mask]
    statuses = df[S.COL_STATUS].astype(str).to_numpy()[open_mask]
    opp_names = df.get(S.COL_OPP_NAME, pd.Series(index=df.index, dtype=object)).to_numpy()[open_mask]
    account_names = df.get(S.COL_ACCOUNT_NAME, pd.Series(index=df.index, dtype=object)).to_numpy()[open_mask]

    so_plan = pd.to_datetime(df.get(S.COL_SO_PLAN_DATE), errors="coerce")
    month_start = pd.Timestamp(asof).to_period("M").to_timestamp()
    so_plan_open = so_plan.to_numpy()[open_mask]
    so_plan_dates = [
        None if pd.isna(d) else pd.Timestamp(d).date().isoformat() for d in so_plan_open
    ]
    at_risks = [(not pd.isna(d)) and pd.Timestamp(d) < month_start for d in so_plan_open]

    rows = build_score_rows(
        ids, probs, drivers, amounts, statuses, run_id, scored_at,
        opp_names=opp_names, account_names=account_names,
        so_plan_dates=so_plan_dates, at_risks=at_risks, model_id=S.MODEL_AZ,
    )
    written = store.upsert_scores(rows)
    bands = {b: int(np.sum([r["band"] == b for r in rows])) for b in ("High", "Mid", "Low")}
    return {
        "run_id": run_id,
        "model_id": S.MODEL_AZ,
        "metrics": wm.metrics,
        "scored": written,
        "bands": bands,
        "scored_at": scored_at.isoformat(),
        "maturity_days": maturity_days,
        "n_converted_train": int((y == 1).sum()),
    }


# F13 run_so_conversion_forecast lives in so_realization.py (serving-safe — pipeline
# imports sklearn at module top, which the lean serving runtime can't load).
