"""Offline backtest — IncomePlan Model Expected vs Actual, Jan–May 2026.

Rigorous point-in-time (leave-the-cohort-out) backtest of the income-line SO forecast:
- TEST cohort  = CLOSED (Won/Lost) deals that have an income-line SO Plan (P) dated
  Jan–May 2026.
- TRAIN        = all OTHER mature closed deals (the test cohort is EXCLUDED → no label
  look-ahead). Features for closed deals are built from each deal's own close reference
  (point-in-time), not `asof`.
- For each plan month: Expected = Σ line_P × predicted_win_prob(deal);
  Actual = Σ line_P where the deal ACTUALLY Won. Error = Expected − Actual.

Needs sklearn/shap (requirements-ml.txt) → run with .venv + a PBI token. Read-only (PBI).
"""
import numpy as np
import pandas as pd

from predictive import schema as S
from predictive.features import build_opp_features
from predictive.ingest import fetch_income_plan_so_lines
from predictive.pipeline import _fetch_aux_sources
from predictive.ingest import fetch_opportunities
from predictive.winprob import train_winprob, score_winprob

ASOF = "2026-06-06"
MONTHS = [f"2026-{m:02d}" for m in range(1, 6)]  # Jan–May

print("fetching opportunities + aux + income-lines ...")
df = fetch_opportunities()
aux = _fetch_aux_sources(None)
fs = build_opp_features(df, asof=ASOF, **aux)
opp_ids = fs.ids.astype(str)
won = fs.y  # 1=Won, 0=Lost, NaN=open

lines = fetch_income_plan_so_lines(2026)
lines = lines[lines["ym"].isin(MONTHS)].copy()
line_opps = set(lines["opp_id"].astype(str))

# TEST cohort = closed deals that appear in Jan–May income-lines
is_closed = fs.y.notna().to_numpy()
in_cohort = opp_ids.isin(line_opps).to_numpy()
test_mask = is_closed & in_cohort
# TRAIN = mature closed deals NOT in the test cohort (leave-cohort-out → no leakage)
train_mask = fs.is_mature.to_numpy() & is_closed & ~in_cohort

print(f"train deals: {train_mask.sum()} | test closed deals in cohort: {test_mask.sum()}")

wm = train_winprob(fs.X[train_mask], fs.y[train_mask].astype(int))
proba = score_winprob(wm, fs.X[test_mask])
test_opp = opp_ids[test_mask].to_numpy()
test_won = fs.y[test_mask].astype(int).to_numpy()
prob_by_opp = dict(zip(test_opp, proba))
won_by_opp = dict(zip(test_opp, test_won))

# aggregate income-lines (closed only) by plan month
rows = {m: {"plan": 0.0, "expected": 0.0, "actual": 0.0, "n": 0} for m in MONTHS}
for _, ln in lines.iterrows():
    opp = str(ln["opp_id"]); m = ln["ym"]; amt = float(ln["amount"] or 0)
    if opp not in prob_by_opp:      # open / not-in-model deals: not part of resolved backtest
        continue
    rows[m]["plan"] += amt
    rows[m]["expected"] += amt * prob_by_opp[opp]
    rows[m]["actual"] += amt * won_by_opp[opp]
    rows[m]["n"] += 1

print("\n=== IncomePlan backtest — Expected (model) vs Actual (Won), by plan month ===")
print(f"{'Month':>8} {'Plan(P)':>12} {'Expected':>12} {'Actual':>12} {'Err':>10} {'Err%':>7} {'n':>4}")
tot = {"plan": 0.0, "expected": 0.0, "actual": 0.0, "n": 0}
for m in MONTHS:
    r = rows[m]
    err = r["expected"] - r["actual"]
    errp = (err / r["actual"] * 100) if r["actual"] else float("nan")
    for k in tot:
        tot[k] += r[k]
    print(f"{m:>8} {r['plan']/1e6:>11.1f}M {r['expected']/1e6:>11.1f}M {r['actual']/1e6:>11.1f}M "
          f"{err/1e6:>9.1f}M {errp:>6.0f}% {r['n']:>4}")
terr = tot["expected"] - tot["actual"]
terrp = (terr / tot["actual"] * 100) if tot["actual"] else float("nan")
print(f"{'TOTAL':>8} {tot['plan']/1e6:>11.1f}M {tot['expected']/1e6:>11.1f}M {tot['actual']/1e6:>11.1f}M "
      f"{terr/1e6:>9.1f}M {terrp:>6.0f}% {tot['n']:>4}")

# accuracy framing
mae = np.mean([abs(rows[m]["expected"] - rows[m]["actual"]) for m in MONTHS]) / 1e6
print(f"\nMAE per month: {mae:.1f}M | overall Expected {tot['expected']/1e6:.1f}M vs "
      f"Actual {tot['actual']/1e6:.1f}M ({terrp:+.0f}%)")
print(f"deal-level: test n={test_mask.sum()}, train n={train_mask.sum()}, "
      f"observed win-rate(test)={test_won.mean():.0%}, mean pred={proba.mean():.0%}")
