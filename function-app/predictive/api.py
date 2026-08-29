"""api.py — F14 list_deals + F17 deal_detail.

Pure functions: take a store + filters, return JSON-able dicts. The HTTP binding
(function_app.py) is a thin wrapper over these so they stay unit-testable.
"""

from __future__ import annotations

from typing import Any

from . import schema as S


def _fmt(row: dict) -> dict:
    """Shape a stored score row for API output."""
    return {
        "opp_id": row["opp_id"],
        "opp_name": row.get("opp_name"),
        "account_name": row.get("account_name"),
        "win_prob": round(float(row["win_prob"]), 4),
        "win_pct": round(float(row["win_prob"]) * 100, 1),
        "band": row["band"],
        "amount": row.get("amount"),
        "status": row.get("status"),
        "so_plan_date": _iso(row.get("so_plan_date")),
        "at_risk": bool(row.get("at_risk", False)),
        "drivers": row.get("drivers", []),
        "scored_at": _iso(row.get("scored_at")),
    }


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


def list_deals(
    store: Any,
    status: str | None = "Open",
    band: str | None = None,
    min_prob: float | None = None,
    limit: int = 50,
    so_plan_month: str | None = None,
    order_by: str = "win_prob",
    at_risk: bool | None = None,
    model_id: str | None = None,
) -> dict:
    """F14 — ranked deals (default: Open deals, highest win-prob first).

    Clean pipeline aggregates (count, weighted, raw, bands) cover every matching
    deal EXCEPT past-due `at_risk` ones (those are counted separately in `at_risk`),
    so the Current-Month KPIs reconcile with the Annual SO-forecast. `deals` is the
    top-`limit` page for the table; `count` is the clean total (not the page size)
    and `returned` is the page size. `so_plan_month` ("YYYY-MM") narrows to deals
    planned to convert that month.
    """
    model_id = S.normalize_model_id(model_id)
    rows = store.get_latest_scores(
        status=status, band=band, min_prob=min_prob, limit=limit,
        so_plan_month=so_plan_month, order_by=order_by, at_risk=at_risk,
        model_id=model_id,
    )
    deals = [_fmt(r) for r in rows]
    summary = store.summarize(
        status=status, band=band, min_prob=min_prob, so_plan_month=so_plan_month,
        model_id=model_id,
    )
    return {
        "model_id": model_id,
        "count": summary["count"],          # total matching deals (clean, excl at_risk)
        "returned": len(deals),             # rows in this page (capped by limit)
        "at_risk": summary.get("at_risk", 0),
        "at_risk_raw": round(summary.get("at_risk_raw", 0), 2),        # Σ amount of at_risk
        "at_risk_weighted": round(summary.get("at_risk_weighted", 0), 2),
        "filters": {
            "status": status, "band": band, "min_prob": min_prob,
            "limit": limit, "so_plan_month": so_plan_month, "at_risk": at_risk,
        },
        "weighted_pipeline": round(summary["weighted"], 2),
        "raw_pipeline": round(summary["raw"], 2),
        "bands": summary["bands"],
        "deals": deals,
    }


def deal_detail(store: Any, opp_id: str, model_id: str | None = None) -> dict | None:
    """F17 — single deal score + drivers, or None if unscored."""
    row = store.get_deal_score(opp_id, model_id=S.normalize_model_id(model_id))
    return _fmt(row) if row else None
