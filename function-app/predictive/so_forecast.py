"""so_forecast.py — monthly + yearly SO predictive.

Splits SO by SO Plan month. Elapsed months use realized SO Actual; the current
month and future months use weighted SO Plan (Σ amount × win_prob) plus a raw
(unweighted) variant. Past-due deals (at_risk) are excluded from the monthly
forecast and reported separately. Yearly = Σ actual(elapsed) + Σ predicted(now..Dec).

No ML deps — reads win-probabilities already persisted in the score store, so it
runs inside the lean serving function runtime.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import schema as S


def build_so_forecast(
    actual_by_month: dict, open_rows: list[dict], asof: str, n_sims: int = 10_000
) -> dict:
    """Pure assembler — realized actuals + EXPECTED prediction + Monte-Carlo band.

    actual_by_month: {"YYYY-MM": realized SO Actual amount}.
    open_rows: dicts with so_plan_date, amount (SO Plan Amount), win_prob, at_risk.
    asof: reference date; its month is the first "predicted" month.
    n_sims: Monte-Carlo draws (each future deal is Bernoulli(win_prob), win-all/lose-all).
    """
    asof_ts = pd.Timestamp(asof)
    year, cur = asof_ts.year, asof_ts.month

    # Each open deal is binary (win the whole project or nothing). The EXPECTED
    # value Σ(amount × win_prob) is the correct portfolio mean (linearity of
    # expectation) but a single point hides the variance. So we ALSO Monte-Carlo
    # each deal as a Bernoulli(win_prob) draw and report a CONSERVATIVE (P10) and
    # OPTIMISTIC (P90) band per month + for the year.
    exp_w: dict = defaultdict(float)    # expected = Σ amount × win_prob
    raw_m: dict = defaultdict(float)    # raw plan (all win)
    fp, famt, fmon = [], [], []         # future deals for simulation
    at_w = at_r = 0.0
    for r in open_rows:
        amt = float(r.get("amount") or 0.0)
        wp = float(r.get("win_prob") or 0.0)
        if r.get("at_risk"):            # past-due → out of the clean forecast
            at_w += amt * wp
            at_r += amt
            continue
        spd = r.get("so_plan_date")
        if not spd:
            continue
        spd = spd if isinstance(spd, str) else (
            spd.isoformat() if hasattr(spd, "isoformat") else str(spd)
        )
        if spd[:4] != str(year) or int(spd[5:7]) < cur:   # this year, current month onward
            continue
        m = spd[:7]
        exp_w[m] += amt * wp
        raw_m[m] += amt
        fp.append(wp); famt.append(amt); fmon.append(m)

    fp = np.array(fp); famt = np.array(famt); fmon = np.array(fmon)
    rng = np.random.default_rng(0)      # fixed seed → deterministic/reproducible
    draws = (rng.random((n_sims, len(fp))) < fp) * famt if len(fp) else np.zeros((n_sims, 0))

    def band(mask):
        if not mask.any():
            return 0.0, 0.0
        s = draws[:, mask].sum(axis=1)
        return float(np.percentile(s, 10)), float(np.percentile(s, 90))

    months = []
    yearly_exp = 0.0
    for mm in range(1, 13):
        m = f"{year}-{mm:02d}"
        if mm < cur:                    # elapsed → realized actual (deterministic)
            a = float(actual_by_month.get(m, 0.0))
            months.append({"month": m, "type": "actual",
                           "expected": round(a, 2), "conservative": round(a, 2),
                           "optimistic": round(a, 2), "raw": round(a, 2)})
            yearly_exp += a
        else:                           # current + future → expected + MC band
            e = exp_w.get(m, 0.0)
            p10, p90 = band(fmon == m)
            months.append({"month": m, "type": "predicted",
                           "expected": round(e, 2), "conservative": round(p10, 2),
                           "optimistic": round(p90, 2), "raw": round(raw_m.get(m, 0.0), 2)})
            yearly_exp += e

    elapsed = sum(float(actual_by_month.get(f"{year}-{mm:02d}", 0.0)) for mm in range(1, cur))
    if len(fp):
        ysum = draws.sum(axis=1) + elapsed
        y_p10, y_p90 = float(np.percentile(ysum, 10)), float(np.percentile(ysum, 90))
    else:
        y_p10 = y_p90 = elapsed
    return {
        "asof": asof,
        "year": year,
        "current_month": f"{year}-{cur:02d}",
        "n_sims": n_sims,
        "months": months,
        "yearly_expected": round(yearly_exp, 2),
        "yearly_conservative": round(y_p10, 2),   # P10 — 90% chance of exceeding
        "yearly_optimistic": round(y_p90, 2),     # P90
        "pipeline_plan": round(float(sum(raw_m.values())), 2),  # raw SO Plan, open deals now..Dec
        "n_deals": int(len(fp)),                  # open deals planned now..Dec (the "Deal Plan")
        "bands": _band_counts(fp),                # win-prob mix of those deals (High/Mid/Low)
        "elapsed_actual": round(elapsed, 2),
        "at_risk": {"weighted": round(at_w, 2), "raw": round(at_r, 2)},
    }


def _band_counts(probs) -> dict:
    """High/Mid/Low deal counts (same thresholds as the per-deal band) — drives the
    Annual power bar so it shows the win-prob mix (incl. red for Low)."""
    out = {"High": 0, "Mid": 0, "Low": 0}
    for p in probs:
        out[S.band(float(p))] += 1
    return out


def build_source_forecast(lines: list[dict], win_prob_by_opp: dict,
                          actual_by_month: dict, asof: str) -> dict:
    """Monthly SO from an income-line source, structured like build_so_forecast:
    elapsed months (< current) show realized SO Actual; current+future show predicted
    Expected = Σ(line amount × win_prob(opp)) by each line's plan month. `lines`: dicts
    with opp_id, ym ('YYYY-MM'), amount. yearly = Σ actual(elapsed) + Σ expected(future)."""
    asof_ts = pd.Timestamp(asof)
    year, cur = asof_ts.year, asof_ts.month
    actual_by_month = actual_by_month or {}

    raw: dict = defaultdict(float)   # future prospect pipeline (raw line P) by month
    exp: dict = defaultdict(float)   # future expected (× win_prob) by month
    opp_wp: dict = {}
    for ln in lines:
        ym = ln.get("ym")
        if not ym or not str(ym).startswith(str(year)):
            continue
        if int(str(ym)[5:7]) < cur:          # elapsed handled by realized actual
            continue
        amt = float(ln.get("amount") or 0.0)
        opp = ln.get("opp_id")
        wp = float(win_prob_by_opp.get(opp, 0.0))
        raw[ym] += amt
        exp[ym] += amt * wp
        if opp is not None:
            opp_wp[opp] = wp

    months = []
    yearly_exp = yearly_raw = 0.0
    for mm in range(1, 13):
        m = f"{year}-{mm:02d}"
        if mm < cur:                          # elapsed → realized SO Actual
            a = round(float(actual_by_month.get(m, 0.0)), 2)
            months.append({"month": m, "type": "actual", "raw": a, "expected": a})
            yearly_exp += a
        else:                                 # current + future → predicted
            r = round(raw.get(m, 0.0), 2)
            e = round(exp.get(m, 0.0), 2)
            months.append({"month": m, "type": "predicted", "raw": r, "expected": e})
            yearly_exp += e
            yearly_raw += r

    bands = {"High": 0, "Mid": 0, "Low": 0}
    for wp in opp_wp.values():
        bands[S.band(wp)] += 1
    elapsed = sum(float(actual_by_month.get(f"{year}-{mm:02d}", 0.0)) for mm in range(1, cur))
    return {
        "year": year,
        "current_month": f"{year}-{cur:02d}",
        "months": months,
        "yearly_expected": round(yearly_exp, 2),   # actual(elapsed) + expected(future)
        "yearly_raw": round(yearly_raw, 2),         # future prospect pipeline only
        "elapsed_actual": round(elapsed, 2),
        "bands": bands,
        "n_opps": len(opp_wp),
        "n_lines": sum(
            1 for ln in lines
            if str(ln.get("ym") or "").startswith(str(year))
            and int((str(ln.get("ym")) + "-00")[5:7] or 0) >= cur
        ),
    }


def enrich_source_lines(lines: list[dict], win_prob_by_opp: dict, name_by_opp: dict,
                        months_keep: set, drivers_by_opp: dict | None = None) -> list[dict]:
    """Per-line rows for the Monthly / Next-3 by-customer tables: keep only lines whose
    plan month is in `months_keep`, attach win-prob + account/opp names + band + drivers."""
    if drivers_by_opp is None:
        drivers_by_opp = {}
    out = []
    for ln in lines:
        ym = ln.get("ym")
        if ym not in months_keep:
            continue
        opp = ln.get("opp_id")
        wp = float(win_prob_by_opp.get(opp, 0.0))
        nm = name_by_opp.get(opp, {})
        out.append({
            "opp_id": opp, "ym": ym, "amount": float(ln.get("amount") or 0.0),
            "win_prob": wp, "win_pct": round(wp * 100, 1), "band": S.band(wp),
            "account_name": nm.get("account_name"),
            # fall back to the line's own Opportunity Name so the table never shows a bare GUID
            "opp_name": nm.get("opp_name") or ln.get("opp_name"),
            "drivers": drivers_by_opp.get(opp, []),
        })
    return out


def run_source_forecast(store, source: str, asof: str, dataset_id: str | None = None,
                        model_id: str | None = None) -> dict:
    """Wire an alternative SO Plan source ('incomeplan' | 'soplan') to win-probs from the
    store. IncomePlan uses each line's own plan month; SOPlan (no date) borrows each opp's
    SO Plan Date from the store. Both run in the lean runtime (PBI read + persisted scores).
    `model_id` selects which model's persisted win-probs to join (default CRM_PDT_BASE)."""
    from .ingest import (
        fetch_income_plan_so_lines, fetch_soplan_so_amounts, fetch_so_actual_by_month,
        fetch_opportunities,
    )

    model_id = S.normalize_model_id(model_id)
    asof_ts = pd.Timestamp(asof)
    year, cur = asof_ts.year, asof_ts.month
    scores = store.get_latest_scores(status="Open", limit=200_000, model_id=model_id)
    win_prob_by_opp = {r["opp_id"]: float(r.get("win_prob") or 0.0) for r in scores}
    drivers_by_opp = {r["opp_id"]: r.get("drivers", []) for r in scores}
    # Names from the FULL opportunity frame (open + closed) so income-line opps that aren't
    # in the open-scored store still resolve to an account name (else the table shows a GUID).
    try:
        opp_df = fetch_opportunities(dataset_id)
        name_by_opp = {
            str(r.get("Opportunity ID")): {
                "account_name": r.get("Account Name"), "opp_name": r.get("Opportunity Name"),
            }
            for r in opp_df.to_dict("records")
        }
    except Exception:  # fall back to store names if the opp pull fails
        name_by_opp = {
            r["opp_id"]: {"account_name": r.get("account_name"), "opp_name": r.get("opp_name")}
            for r in scores
        }

    if source == "incomeplan":
        df = fetch_income_plan_so_lines(year, dataset_id)
        lines = df.to_dict("records") if not df.empty else []
    elif source == "soplan":
        def _ym(v):
            if not v:
                return None
            return (v if isinstance(v, str) else (v.isoformat() if hasattr(v, "isoformat") else str(v)))[:7]
        plan_month = {r["opp_id"]: _ym(r.get("so_plan_date")) for r in scores}
        df = fetch_soplan_so_amounts(dataset_id)
        lines = [
            {"opp_id": r["opp_id"], "ym": plan_month.get(r["opp_id"]), "amount": r["amount"]}
            for r in (df.to_dict("records") if not df.empty else [])
        ]
    else:
        raise ValueError(f"unknown source: {source!r} (expected 'incomeplan' or 'soplan')")

    actual_by_month = fetch_so_actual_by_month(year, dataset_id)
    out = build_source_forecast(lines, win_prob_by_opp, actual_by_month, asof)
    out["asof"] = asof
    out["source"] = source
    out["model_id"] = model_id
    # line-level rows for the Monthly + Next-3 by-customer tables (current..+3 only)
    keep = {f"{year if cur + i <= 12 else year + 1}-{((cur - 1 + i) % 12) + 1:02d}" for i in range(4)}
    out["lines"] = enrich_source_lines(lines, win_prob_by_opp, name_by_opp, keep,
                                         drivers_by_opp=drivers_by_opp)
    # "Delay Prospects" = income-lines whose plan month is PAST (< current) but the deal is
    # still OPEN (present in the win-prob store) → slipped/past-due revenue, not yet realized.
    past = {f"{year}-{mm:02d}" for mm in range(1, cur)}
    delayed = [
        d for d in enrich_source_lines(lines, win_prob_by_opp, name_by_opp, past,
                                        drivers_by_opp=drivers_by_opp)
        if d["opp_id"] in win_prob_by_opp
    ]
    out["delayed"] = delayed
    out["delayed_total"] = round(sum(d["amount"] for d in delayed), 2)
    return out


def run_so_forecast(store, asof: str, dataset_id: str | None = None,
                    model_id: str | None = None) -> dict:
    """Wire live data: realized actuals (PBI) + persisted win-prob scores (store).
    `model_id` selects which model's scores to read (default CRM_PDT_BASE)."""
    from .ingest import fetch_so_actual_by_month

    model_id = S.normalize_model_id(model_id)
    actual = fetch_so_actual_by_month(pd.Timestamp(asof).year, dataset_id)
    rows = store.get_latest_scores(status="Open", limit=100_000, model_id=model_id)
    open_rows = [
        {
            "so_plan_date": r.get("so_plan_date"),
            "amount": r.get("amount"),
            "win_prob": r.get("win_prob"),
            "at_risk": r.get("at_risk"),
        }
        for r in rows
    ]
    out = build_so_forecast(actual, open_rows, asof)
    out["model_id"] = model_id
    return out
