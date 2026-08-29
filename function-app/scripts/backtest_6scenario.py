"""backtest_6scenario.py — 3 models x 2 pipelines = 6 scenarios (2026-06-20).

Models (feature/label contract, schema.py registry):
  BASE = CRM_PDT_BASE  Fact_Opportunity + 4 aux         label = Status (Won/Lost)
  MIX  = CRM_PDT_MIX   BASE + Fact_IncomePlan Group-5    label = Status (Won/Lost)
  AZ   = CRM_PDT_AZ    BASE features                     label = Fact_SalesOrder ledger

Pipelines (= metric VIEW over the SAME per-model temporal-holdout test split):
  P-OPP (Fact_Opportunity) = UNWEIGHTED classification  -> AUC / Brier / lift / calib
  P-INC (Fact_IncomePlan)  = ฿-AMOUNT-WEIGHTED           -> wAUC / wBrier / gains-lift
                             + money accuracy: Expected฿ vs Actual฿ vs %err

Both pipelines of one model reuse ONE train+score (identical y,p on the test set);
they differ only by sample-weighting on SO Plan Amount (P). This makes the 6 cells
directly comparable while still surfacing the ฿-accuracy that AUC alone hides.

Shared params: temporal holdout 70/30, MATURITY_DAYS=540, asof 2026-06-20, live PBI.
BASE/MIX order by Create Date (Status label); AZ orders by point-in-time reference
date (SO-conversion label) — mirrors pipeline.run_backtest / so_conversion.backtest.

Run:
  export PBI_ACCESS_TOKEN="$(az account get-access-token \
    --resource https://analysis.windows.net/powerbi/api --query accessToken -o tsv)"
  export PBI_CLIENT_SECRET=""; export PYTHONPATH=$PWD
  .venv/bin/python scripts/backtest_6scenario.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from predictive import schema as S
from predictive.features import (
    _reference_dates,
    build_opp_features,
    build_so_conversion_label,
)
from predictive.ingest import (
    fetch_accounts,
    fetch_activities,
    fetch_income_plan_so_lines,
    fetch_invoice_history,
    fetch_movements,
    fetch_opportunities,
    fetch_so_conversions,
)
from predictive.winprob import score_winprob, train_winprob

ASOF = "2026-06-20"
TEST_FRAC = 0.30
MATURITY = S.MATURITY_DAYS


def _income_lines(dataset_id, asof):
    year = pd.Timestamp(asof).year
    frames = []
    for yr in (year - 1, year):
        try:
            df = fetch_income_plan_so_lines(yr, dataset_id)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None


def _aux(dataset_id, model_id, asof):
    src = {}
    for name, fn in (
        ("activity", fetch_activities),
        ("movement", fetch_movements),
        ("accounts", fetch_accounts),
        ("invoices", fetch_invoice_history),
    ):
        try:
            src[name] = fn(dataset_id)
        except Exception:
            src[name] = None
    if model_id == S.MODEL_MIX:
        src["income"] = _income_lines(dataset_id, asof)
    return src


def _temporal_split(X, y, order, amount, test_frac):
    """Return (X_tr, y_tr, X_te, y_te, amt_te) — most-recent test_frac held out."""
    m = y.notna().to_numpy()
    Xc, yc, amt = X[m], y[m].astype(int), amount[m]
    ordv = order[m].to_numpy()
    idx = np.argsort(ordv, kind="stable")
    Xc, yc, amt = Xc.iloc[idx], yc.iloc[idx], amt.iloc[idx]
    n = len(yc)
    n_test = int(round(n * test_frac))
    n_train = n - n_test
    return (
        Xc.iloc[:n_train], yc.iloc[:n_train],
        Xc.iloc[n_train:], yc.iloc[n_train:], amt.iloc[n_train:],
        n_train, n_test,
    )


def _opp_view(y, p):
    """P-OPP: unweighted classification metrics."""
    yt = y.to_numpy()
    auc = None if len(np.unique(yt)) < 2 else round(float(roc_auc_score(yt, p)), 4)
    brier = round(float(np.mean((p - yt) ** 2)), 4)
    acc = float(((p >= 0.5).astype(int) == yt).mean())
    base = max(yt.mean(), 1 - yt.mean())
    return {
        "auc": auc,
        "brier": brier,
        "acc": round(acc, 4),
        "base": round(float(base), 4),
        "lift_pp": round((acc - base) * 100, 1),
    }


def _inc_view(y, p, amt):
    """P-INC: ฿-amount-weighted classification + money accuracy."""
    yt = y.to_numpy()
    w = pd.to_numeric(amt, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    wauc = (
        None if len(np.unique(yt)) < 2 or w.sum() == 0
        else round(float(roc_auc_score(yt, p, sample_weight=w)), 4)
    )
    wbrier = round(float(np.sum(w * (p - yt) ** 2) / w.sum()), 4) if w.sum() else None
    # gains lift: share of ACTUAL ฿ captured by the top-30% deals (by prob) / 0.30
    k = max(1, int(round(0.30 * len(p))))
    top = np.argsort(p)[::-1][:k]
    actual_amt = w * yt
    tot_actual = actual_amt.sum()
    gains = (actual_amt[top].sum() / tot_actual) if tot_actual else 0.0
    expected = float(np.sum(p * w))           # Σ prob × ฿
    actual = float(tot_actual)                # Σ ฿ where outcome=1
    err = round((expected - actual) / actual * 100, 1) if actual else None
    mae_deal = round(float(np.mean(np.abs(p - yt) * w)) / 1e6, 3)  # ฿M avg |err| per deal
    return {
        "wauc": wauc,
        "wbrier": wbrier,
        "gains_lift": round(float(gains / 0.30), 2),
        "exp_m": round(expected / 1e6, 1),
        "act_m": round(actual / 1e6, 1),
        "err_pct": err,
        "mae_m": mae_deal,
    }


def _run_model(model_id, df, dataset_id):
    aux = _aux(dataset_id, model_id, ASOF)
    fs = build_opp_features(df, asof=ASOF, maturity_days=MATURITY,
                            model_id=(S.MODEL_BASE if model_id == S.MODEL_AZ else model_id),
                            **{k: v for k, v in aux.items()})
    amount = pd.to_numeric(df[S.COL_AMOUNT], errors="coerce")
    amount.index = fs.X.index

    if model_id == S.MODEL_AZ:
        so_conv = fetch_so_conversions(dataset_id)
        y = build_so_conversion_label(df, so_conv, ASOF, maturity_days=MATURITY)
        y.index = fs.X.index
        order = _reference_dates(df, pd.Timestamp(ASOF))
        order.index = fs.X.index
        X, yv, amt, ordv = fs.X, y, amount, order
    else:
        # Status label, mature cohorts only, ordered by Create Date (run_backtest parity)
        mask = fs.is_mature.to_numpy()
        order = pd.to_datetime(df[S.COL_CREATE], errors="coerce")
        order.index = fs.X.index
        X = fs.X[mask]
        yv = fs.y[mask]
        amt = amount[mask]
        ordv = order[mask]

    X_tr, y_tr, X_te, y_te, amt_te, n_tr, n_te = _temporal_split(
        X, yv, ordv, amt, TEST_FRAC
    )
    wm = train_winprob(X_tr, y_tr)
    p = score_winprob(wm, X_te)
    return n_tr, n_te, _opp_view(y_te, p), _inc_view(y_te, p, amt_te)


def main():
    dataset_id = None
    df = fetch_opportunities(dataset_id)
    print(f"asof={ASOF}  test_frac={TEST_FRAC}  maturity_days={MATURITY}  "
          f"opps={len(df)}\n")

    results = {}
    for mid in (S.MODEL_BASE, S.MODEL_MIX, S.MODEL_AZ):
        print(f"=== {mid} ===")
        n_tr, n_te, opp, inc = _run_model(mid, df, dataset_id)
        results[mid] = (n_tr, n_te, opp, inc)
        print(f"  n_train={n_tr} n_test={n_te}")
        print(f"  P-OPP  AUC={opp['auc']} Brier={opp['brier']} acc={opp['acc']} "
              f"base={opp['base']} lift={opp['lift_pp']}pp")
        print(f"  P-INC  wAUC={inc['wauc']} wBrier={inc['wbrier']} "
              f"gains_lift={inc['gains_lift']}x | Exp={inc['exp_m']}M "
              f"Act={inc['act_m']}M err={inc['err_pct']}%\n")

    print("=== SUMMARY: P-OPP (unweighted classification) ===")
    print(f"{'model':<14}{'AUC':>8}{'Brier':>9}{'acc':>8}{'base':>8}{'lift_pp':>9}")
    for mid in (S.MODEL_BASE, S.MODEL_MIX, S.MODEL_AZ):
        _, _, o, _ = results[mid]
        print(f"{mid:<14}{str(o['auc']):>8}{o['brier']:>9}{o['acc']:>8}"
              f"{o['base']:>8}{o['lift_pp']:>9}")

    print("\n=== SUMMARY: P-INC (฿-amount-weighted + money) ===")
    print(f"{'model':<14}{'wAUC':>8}{'wBrier':>9}{'gainsX':>8}{'Exp_M':>9}"
          f"{'Act_M':>9}{'err%':>8}")
    for mid in (S.MODEL_BASE, S.MODEL_MIX, S.MODEL_AZ):
        _, _, _, c = results[mid]
        print(f"{mid:<14}{str(c['wauc']):>8}{str(c['wbrier']):>9}"
              f"{c['gains_lift']:>8}{c['exp_m']:>9}{c['act_m']:>9}"
              f"{str(c['err_pct']):>8}")


if __name__ == "__main__":
    main()
