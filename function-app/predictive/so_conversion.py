"""so_conversion.py — Stage A of the new SO predictive (model_id CRM_PDT_AZ).

Predicts P(opportunity → a real Sales Order). The model algorithm is identical to
winprob (regularized HistGradientBoosting + sigmoid calibration + SHAP), so it is
REUSED here verbatim — the only difference is the LABEL: build_so_conversion_label
grounds it in the Fact_SalesOrder ledger (incl the 210 Open deals that already
booked an order) instead of Status=Won.

Feature contract = CRM_PDT_BASE (the opportunity feature set). CRM_PDT_AZ is its own
model_id only for SCORE STORAGE (F10), not a new feature group.

Pure composition over ingest/features/winprob — offline-testable with synthetic frames.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import schema as S
from . import winprob as W
from .features import (
    build_opp_features,
    build_so_conversion_label,
    _reference_dates,
)

# Stage A learns from opportunity features; the conversion label is the new target.
_FEATURE_MODEL = S.MODEL_BASE


def _aux_kwargs(activity, movement, accounts, invoices):
    return dict(activity=activity, movement=movement, accounts=accounts, invoices=invoices)


def train_so_conversion(
    opps: pd.DataFrame,
    so_conversions: pd.DataFrame,
    asof: str | datetime,
    maturity_days: int = S.MATURITY_DAYS,
    activity: pd.DataFrame | None = None,
    movement: pd.DataFrame | None = None,
    accounts: pd.DataFrame | None = None,
    invoices: pd.DataFrame | None = None,
) -> W.WinProbModel:
    """F5 — train the SO-conversion model on resolved (mature/Lost or converted) deals.
    Label = build_so_conversion_label (Fact_SalesOrder ledger). Immature SO-less
    cohorts are NaN → train_winprob drops them."""
    fs = build_opp_features(
        opps, asof=asof, maturity_days=maturity_days, model_id=_FEATURE_MODEL,
        **_aux_kwargs(activity, movement, accounts, invoices),
    )
    y = build_so_conversion_label(opps, so_conversions, asof, maturity_days=maturity_days)
    return W.train_winprob(fs.X, y)


def score_so_conversion(
    model: W.WinProbModel,
    opps: pd.DataFrame,
    asof: str | datetime,
    activity: pd.DataFrame | None = None,
    movement: pd.DataFrame | None = None,
    accounts: pd.DataFrame | None = None,
    invoices: pd.DataFrame | None = None,
):
    """F6 — P(SO conversion) per opp (calibrated). Scores ALL rows (open or closed)."""
    fs = build_opp_features(
        opps, asof=asof, model_id=_FEATURE_MODEL,
        **_aux_kwargs(activity, movement, accounts, invoices),
    )
    return W.score_winprob(model, fs.X)


def explain_so_conversion(
    model: W.WinProbModel,
    opps: pd.DataFrame,
    asof: str | datetime,
    top_n: int = 3,
    activity: pd.DataFrame | None = None,
    movement: pd.DataFrame | None = None,
    accounts: pd.DataFrame | None = None,
    invoices: pd.DataFrame | None = None,
) -> list[list[dict]]:
    """F6 — per-opp top SHAP drivers of the conversion score."""
    fs = build_opp_features(
        opps, asof=asof, model_id=_FEATURE_MODEL,
        **_aux_kwargs(activity, movement, accounts, invoices),
    )
    return W.explain_winprob(model, fs.X, top_n=top_n)


def backtest_so_conversion(
    opps: pd.DataFrame,
    so_conversions: pd.DataFrame,
    asof: str | datetime,
    test_frac: float = 0.30,
    maturity_days: int = S.MATURITY_DAYS,
    n_bins: int = 10,
    activity: pd.DataFrame | None = None,
    movement: pd.DataFrame | None = None,
    accounts: pd.DataFrame | None = None,
    invoices: pd.DataFrame | None = None,
) -> dict:
    """F7 — temporal holdout backtest of the conversion model. Orders resolved deals
    by their point-in-time reference date (close ref → asof) so the test set is the
    most recent cohort; reports OOS AUC/Brier/lift/calibration vs base rate."""
    fs = build_opp_features(
        opps, asof=asof, maturity_days=maturity_days, model_id=_FEATURE_MODEL,
        **_aux_kwargs(activity, movement, accounts, invoices),
    )
    y = build_so_conversion_label(opps, so_conversions, asof, maturity_days=maturity_days)
    order = _reference_dates(opps, pd.Timestamp(asof))
    out = W.backtest_winprob(fs.X, y, order=order, test_frac=test_frac, n_bins=n_bins)
    # lift over the majority-class baseline (accuracy alone misleads under imbalance)
    base = max(out["test_base_rate"], 1.0 - out["test_base_rate"])
    out["baseline_accuracy"] = round(float(base), 4)
    out["lift"] = round(float(out["accuracy"] - base), 4)
    return out
