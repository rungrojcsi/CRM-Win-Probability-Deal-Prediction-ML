"""attainment.py — F10 compute_pacing + F11 predict_eom.

For the current month, per salesperson: compare actual MTD bookings to the monthly
target (Fact_GoalMonth), project end-of-month via run-rate, and classify pacing.

ACTUALS SOURCE = Fact_SalesOrder (bookings). Verified: SalesOrder salesperson IDs
overlap 15/15 with GoalMonth targets, vs Invoice 2/15 — so the quota is a booking
target, not a billing one. `actuals` is still injectable for swapping.

Run-rate guard: early in the month a single order makes the linear EOM projection
explode, so attainment is flagged low_confidence until `min_fraction` of the month
has elapsed.
"""

from __future__ import annotations

import calendar

import pandas as pd

# pacing tolerance around the linear target pace
AHEAD = 1.05
BEHIND = 0.95
# below this fraction of the month elapsed, run-rate EOM projection is unreliable
MIN_CONFIDENT_FRACTION = 0.25


def _month_fraction(asof: pd.Timestamp) -> float:
    """Fraction of the current month elapsed (day d of D), in (0, 1]."""
    days_in_month = calendar.monthrange(asof.year, asof.month)[1]
    return asof.day / days_in_month


def _monthly_actuals(actuals: pd.DataFrame, month_start: pd.Timestamp) -> pd.Series:
    df = actuals.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    nxt = (month_start.to_period("M") + 1).to_timestamp()
    cur = df[(df["date"] >= month_start) & (df["date"] < nxt)]
    return cur.groupby("sales_id")["amount"].sum()


def _targets_for_month(targets: pd.DataFrame, month_start: pd.Timestamp) -> pd.Series:
    df = targets.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M")
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0.0)
    cur = df[df["month"] == month_start.to_period("M")]
    return cur.groupby("sales_id")["target"].sum()


def _status(actual_mtd: float, target: float, frac: float) -> str:
    if target <= 0:
        return "no_target"
    expected = target * frac
    if actual_mtd >= expected * AHEAD:
        return "ahead"
    if actual_mtd < expected * BEHIND:
        return "behind"
    return "on_track"


def compute_attainment(
    targets: pd.DataFrame,
    actuals: pd.DataFrame,
    asof: str,
    min_fraction: float = MIN_CONFIDENT_FRACTION,
    names: dict[str, str] | None = None,
) -> dict:
    """F10+F11 — per-salesperson pacing + end-of-month projection for asof's month."""
    asof_ts = pd.Timestamp(asof)
    month_start = asof_ts.to_period("M").to_timestamp()
    frac = _month_fraction(asof_ts)
    low_confidence = frac < min_fraction

    target_s = _targets_for_month(targets, month_start)
    actual_s = _monthly_actuals(actuals, month_start)
    sales_ids = sorted(set(target_s.index) | set(actual_s.index))

    names = names or {}
    rows = []
    for sid in sales_ids:
        target = float(target_s.get(sid, 0.0))
        actual_mtd = float(actual_s.get(sid, 0.0))
        predicted_eom = actual_mtd / frac if frac > 0 else actual_mtd
        attainment = (predicted_eom / target) if target > 0 else None
        rows.append(
            {
                "sales_id": sid,
                "sales_name": names.get(sid),
                "target": round(target, 2),
                "actual_mtd": round(actual_mtd, 2),
                "predicted_eom": round(predicted_eom, 2),
                "attainment_pct": round(attainment * 100, 1) if attainment is not None else None,
                "status": _status(actual_mtd, target, frac),
            }
        )

    team_target = float(target_s.sum())
    team_pred = float(sum(r["predicted_eom"] for r in rows))
    return {
        "month": month_start.strftime("%Y-%m"),
        "month_fraction_elapsed": round(frac, 3),
        "low_confidence": low_confidence,
        "note": (
            "EOM projection unreliable — less than "
            f"{int(min_fraction * 100)}% of the month elapsed"
            if low_confidence
            else None
        ),
        "team": {
            "target": round(team_target, 2),
            "actual_mtd": round(float(actual_s.sum()), 2),
            "predicted_eom": round(team_pred, 2),
            "attainment_pct": round(team_pred / team_target * 100, 1) if team_target > 0 else None,
        },
        "by_sales": sorted(
            rows, key=lambda r: (r["attainment_pct"] is None, -(r["attainment_pct"] or 0))
        ),
    }
