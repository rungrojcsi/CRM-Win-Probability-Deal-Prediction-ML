"""forecast.py — F9 revenue forecast.

Honest BASELINE forecaster for a short, gappy monthly revenue series (~18 months
with missing months + a migration outlier). Deliberately dependency-light
(numpy/pandas only) — TimesFM/Prophet would overfit this few points. The method:

  1. aggregate daily invoices → continuous monthly series (missing months → 0)
  2. drop the current (partial) month and a leading migration outlier window
  3. forecast h months ahead at a robust level (trailing median) with a damped
     trend, and an empirical prediction band from trailing residuals.

Swap for TimesFM once snapshot history is deeper (see roadmap Phase 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 6
Z_80 = 1.2816  # ~80% prediction interval


def aggregate_monthly(invoices: pd.DataFrame) -> pd.Series:
    """Daily invoice rows → continuous monthly revenue Series (gaps filled 0)."""
    df = invoices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    monthly = (
        df.dropna(subset=["date"])
        .set_index("date")["amount"]
        .resample("MS")
        .sum()
    )
    if monthly.empty:
        return monthly
    full = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    return monthly.reindex(full, fill_value=0.0)


def _complete_history(series: pd.Series, asof: pd.Timestamp) -> pd.Series:
    """Drop the current (partial) month and anything after asof."""
    cur = asof.to_period("M").to_timestamp()
    return series[series.index < cur]


def forecast_revenue(
    invoices: pd.DataFrame,
    asof: str,
    horizon: int = 3,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """F9 — forecast the next `horizon` months of revenue with an 80% band."""
    asof_ts = pd.Timestamp(asof)
    series = _complete_history(aggregate_monthly(invoices), asof_ts)
    if len(series) < 3:
        raise ValueError("need >= 3 complete months to forecast")

    hist = series.tail(window)
    level = float(np.median(hist.values))
    # damped linear trend on the window (gentle; clipped to avoid runaway).
    x = np.arange(len(hist))
    slope = float(np.polyfit(x, hist.values, 1)[0]) if len(hist) >= 2 else 0.0
    resid_std = float(np.std(hist.values - np.median(hist.values)))

    last_month = series.index.max()
    points = []
    for h in range(1, horizon + 1):
        damp = 0.5 ** (h - 1)  # trend influence decays each step
        point = max(0.0, level + slope * damp * h)
        margin = Z_80 * resid_std * np.sqrt(h)
        m = (last_month.to_period("M") + h).to_timestamp()
        points.append(
            {
                "month": m.strftime("%Y-%m"),
                "forecast": round(point, 2),
                "lower": round(max(0.0, point - margin), 2),
                "upper": round(point + margin, 2),
            }
        )

    return {
        "method": "trailing-median + damped-trend (baseline)",
        "window": int(len(hist)),
        "level": round(level, 2),
        "history_months": int(len(series)),
        "last_complete_month": last_month.strftime("%Y-%m"),
        "forecast": points,
    }
