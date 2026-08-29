"""so_realization.py — Stage B of the new SO predictive.

Stage A (so_conversion.py) gives P(opp → a Sales Order). Stage B turns the SO-Plan
income-line pipeline into a calibrated SO forecast using the EMPIRICAL realization
of Fact_IncomePlan (how much planned SO-Plan actually lands, and when), instead of
the old Σ(plan × win_prob) which overstates.

Two pure functions, no live calls:
- F8 build_realization_curve: realization rate ($ landed / $ planned) on resolved
  past cohorts + plan→actual month-lag distribution (audit: rate ~45%, lag ≈ 0).
- F9 assemble_so_forecast: elapsed months = realized actual; current+future months =
  Σ(plan_line × conversion_prob(opp) × realization_rate) spread by the lag curve,
  with a conservative band from the low realization rate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class RealizationCurve:
    rate: float                       # the rate the assembler multiplies by conv_prob.
                                      # CONDITIONAL (realized/plan | converted) when
                                      # converted_opps is supplied — else unconditional.
    rate_low: float                   # conservative (worst resolved-year rate, same basis)
    lag_dist: dict[int, float] = field(default_factory=lambda: {0: 1.0})  # month-lag → weight
    n_plan: int = 0
    n_realized: int = 0
    rate_basis: str = "unconditional"  # "conditional" once converted_opps is known
    unconditional_rate: float = 0.0   # Σ actual / Σ plan over ALL resolved lines (reference)


def _ym_to_int(ym: str) -> int | None:
    """'YYYY-MM' → ordinal months, 0-indexed (year*12 + month-1), for lag math."""
    if not isinstance(ym, str) or len(ym) < 7:
        return None
    try:
        return int(ym[:4]) * 12 + (int(ym[5:7]) - 1)
    except ValueError:
        return None


def _int_to_ym(t: int) -> str:
    """Inverse of _ym_to_int → 'YYYY-MM'."""
    return f"{t // 12:04d}-{(t % 12) + 1:02d}"


def build_realization_curve(
    lines: pd.DataFrame, asof: str | datetime, converted_opps: set | None = None
) -> RealizationCurve:
    """F8 — realization rate + plan→actual month-lag from IncomePlan plan/actual lines.

    lines: [opp_id, plan_ym, plan_amount, actual_ym, actual_amount]. "Resolved" =
    plan year strictly before asof's year (fully past cohorts, not right-censored).

    `rate` is the factor the assembler multiplies by P(convert), so it must NOT
    re-embed conversion:
      - converted_opps given → CONDITIONAL rate = Σ actual / Σ plan over resolved lines
        whose opp converted (in the Fact_SalesOrder ledger). E[realized | converted].
        Backtest-validated (-2.9% on Jan–May vs the unconditional rate's double-discount
        -42.5%). The assembler's plan × conv_prob × this_rate is the correct two-stage
        decomposition.
      - converted_opps None → falls back to the unconditional rate (legacy / offline).
    rate_low = worst resolved-year rate on the SAME basis. lag = max(0, actual_month −
    plan_month) over realized lines, $-weighted then normalised."""
    asof_year = pd.Timestamp(asof).year
    if lines is None or lines.empty:
        return RealizationCurve(rate=0.0, rate_low=0.0)

    df = lines.copy()
    df["opp_id"] = df["opp_id"].astype(str).str.strip()
    df["plan_amount"] = pd.to_numeric(df["plan_amount"], errors="coerce").fillna(0.0)
    df["actual_amount"] = pd.to_numeric(df.get("actual_amount"), errors="coerce").fillna(0.0)
    df["plan_yr"] = df["plan_ym"].astype(str).str[:4]

    resolved = df[pd.to_numeric(df["plan_yr"], errors="coerce") < asof_year]
    uncond = (float(resolved["actual_amount"].sum()) / float(resolved["plan_amount"].sum())
              if resolved["plan_amount"].sum() > 0 else 0.0)

    # CONDITIONAL on conversion when we know which opps converted; else unconditional.
    if converted_opps is not None:
        basis = "conditional"
        scope = resolved[resolved["opp_id"].isin({str(o).strip() for o in converted_opps})]
    else:
        basis = "unconditional"
        scope = resolved
    plan_sum = float(scope["plan_amount"].sum())
    act_sum = float(scope["actual_amount"].sum())
    rate = (act_sum / plan_sum) if plan_sum > 0 else 0.0

    # worst resolved-year rate (same basis) as the conservative floor
    rate_low = rate
    by_yr = scope.groupby("plan_yr").agg(p=("plan_amount", "sum"), a=("actual_amount", "sum"))
    yr_rates = [(r.a / r.p) for r in by_yr.itertuples() if r.p > 0]
    if yr_rates:
        rate_low = min(yr_rates)

    # plan→actual month-lag distribution ($-weighted) over realized lines
    realized = resolved[(resolved["actual_ym"].astype(str).str.len() >= 7)
                        & (resolved["actual_amount"] > 0)]
    lag_w: dict[int, float] = defaultdict(float)
    for r in realized.itertuples():
        pm, am = _ym_to_int(r.plan_ym), _ym_to_int(r.actual_ym)
        if pm is None or am is None:
            continue
        lag = max(0, am - pm)
        lag_w[lag] += float(r.plan_amount)
    total_w = sum(lag_w.values())
    lag_dist = {k: v / total_w for k, v in sorted(lag_w.items())} if total_w > 0 else {0: 1.0}

    return RealizationCurve(
        rate=round(rate, 4), rate_low=round(rate_low, 4), lag_dist=lag_dist,
        n_plan=int(len(scope)), n_realized=int(len(realized)),
        rate_basis=basis, unconditional_rate=round(uncond, 4),
    )


def _spread(amount: float, plan_ym: str, lag_dist: dict[int, float]) -> dict[str, float]:
    """Distribute an expected amount from its plan month across target months by lag."""
    base = _ym_to_int(plan_ym)
    if base is None:
        return {}
    out: dict[str, float] = {}
    for lag, w in lag_dist.items():
        ym = _int_to_ym(base + lag)
        out[ym] = out.get(ym, 0.0) + amount * w
    return out


def assemble_so_forecast(
    plan_lines: pd.DataFrame,
    conversion_prob_by_opp: dict[str, float],
    curve: RealizationCurve,
    actual_by_month: dict[str, float],
    asof: str | datetime,
) -> dict:
    """F9 — combine Stage-A conversion × Stage-B realization into a monthly + yearly
    SO forecast for `asof`'s year.

    plan_lines: future SO-Plan income lines [opp_id, ym, amount].
    conversion_prob_by_opp: opp_id → P(SO) from so_conversion (Stage A).
    Elapsed months (< current) report realized actual_by_month. Current..Dec report
    Σ(amount × conv_prob × rate) spread by the lag curve; conservative uses rate_low.
    """
    asof_ts = pd.Timestamp(asof)
    year, cur = asof_ts.year, asof_ts.month

    exp_by_m: dict[str, float] = defaultdict(float)
    con_by_m: dict[str, float] = defaultdict(float)
    raw_by_m: dict[str, float] = defaultdict(float)
    pipeline_plan = 0.0
    n_lines = 0

    if plan_lines is not None and not plan_lines.empty:
        for r in plan_lines.itertuples():
            ym = str(getattr(r, "ym", "") or "")
            if len(ym) < 7 or int(ym[:4]) != year or int(ym[5:7]) < cur:
                continue   # this year, current month onward only
            amt = float(getattr(r, "amount", 0.0) or 0.0)
            p = float(conversion_prob_by_opp.get(str(getattr(r, "opp_id", "")), 0.0))
            pipeline_plan += amt
            n_lines += 1
            for tym, exp_amt in _spread(amt * p * curve.rate, ym, curve.lag_dist).items():
                if tym[:4] == str(year):
                    exp_by_m[tym] += exp_amt
            for tym, con_amt in _spread(amt * p * curve.rate_low, ym, curve.lag_dist).items():
                if tym[:4] == str(year):
                    con_by_m[tym] += con_amt
            raw_by_m[ym] += amt

    months = []
    yearly_expected = yearly_conservative = elapsed_actual = 0.0
    for m in range(1, 13):
        ym = f"{year}-{m:02d}"
        if m < cur:
            actual = float(actual_by_month.get(ym, 0.0))
            months.append({"month": ym, "type": "actual", "expected": actual,
                           "conservative": actual, "raw": actual})
            yearly_expected += actual
            yearly_conservative += actual
            elapsed_actual += actual
        else:
            exp = exp_by_m.get(ym, 0.0)
            con = con_by_m.get(ym, 0.0)
            months.append({"month": ym, "type": "predicted", "expected": exp,
                           "conservative": con, "raw": raw_by_m.get(ym, 0.0)})
            yearly_expected += exp
            yearly_conservative += con

    return {
        "asof": str(asof),
        "months": months,
        "yearly_expected": yearly_expected,
        "yearly_conservative": yearly_conservative,
        "elapsed_actual": elapsed_actual,
        "pipeline_plan": pipeline_plan,
        "n_lines": n_lines,
        "realization_rate": curve.rate,
        "realization_rate_low": curve.rate_low,
    }


def run_so_conversion_forecast(store, asof: str, dataset_id: str | None = None) -> dict:
    """F13 — assemble the 2-stage SO forecast for SERVING. Lives here (not pipeline.py)
    so the import chain stays pandas-only: pipeline imports winprob/sklearn at module
    top, which the lean serving runtime can't load.

    Stage A conversion probabilities are read from the store (model_id=CRM_PDT_AZ,
    persisted by pipeline.run_so_scoring offline). Stage B realization curve + plan
    lines + elapsed actuals are pulled live and combined here."""
    from . import schema as S
    from .ingest import (
        fetch_so_conversions, fetch_incomeplan_realization,
        fetch_income_plan_so_lines, fetch_so_actual_by_month,
    )

    scores = store.get_latest_scores(status="Open", limit=200_000, model_id=S.MODEL_AZ)
    conv = {r["opp_id"]: float(r.get("win_prob") or 0.0) for r in scores}

    year = pd.Timestamp(asof).year
    # CONDITIONAL realization: the curve's rate must be E[realized | converted] because
    # the assembler already multiplies by P(convert) — the unconditional rate double-
    # discounts (backtest -42.5%). converted_opps = the Fact_SalesOrder ledger.
    so_ledger = fetch_so_conversions(dataset_id)
    converted_opps = set(so_ledger["opp_id"].astype(str)) if not so_ledger.empty else set()
    curve = build_realization_curve(
        fetch_incomeplan_realization(dataset_id), asof, converted_opps=converted_opps)
    plan = fetch_income_plan_so_lines(year, dataset_id)
    plan_lines = plan[["opp_id", "ym", "amount"]] if not plan.empty else plan
    actual_by_month = fetch_so_actual_by_month(year, dataset_id)

    out = assemble_so_forecast(plan_lines, conv, curve, actual_by_month, asof)
    out["model_id"] = S.MODEL_AZ
    out["n_scored_opps"] = len(conv)
    return out
