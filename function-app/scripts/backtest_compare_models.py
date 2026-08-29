"""Offline temporal-holdout comparison: CRM_PDT_OPP vs CRM_PDT_INC.

Same maturity-filtered, Create-Date-ordered temporal split for BOTH models (train on
the earliest 70%, evaluate the most recent 30%). The only difference is the feature
contract: CRM_PDT_INC adds the Fact_IncomePlan SO-Plan income-line group (Group 5).

Reports AUC / Brier / accuracy + lift over the test-cohort base rate for each model so
we can see honestly whether the income features help. Needs sklearn/shap + a PBI token.

LEAKAGE NOTE (verified 2026-06-06): the income builder uses a SYMMETRIC cutoff
(Create Date + INCOME_CUTOFF_DAYS), identical for Won and Lost, NOT the deal's close
reference. A close-ref cutoff is asymmetric — Won deals carry a real SO Actual Date,
Lost deals fall back to Last Activity/Create — which let income-line PRESENCE proxy the
label and inflated AUC to ~0.92. Under the symmetric cutoff the gain is honest and modest
(see in-file results below). Even then, an early SO-Plan line still carries genuine signal
(~71% win-rate with vs ~18% without), so the lift is real, not a reference-date artifact.

Last offline run (2026-06-06, mature closed cohort, 70/30 temporal split):
  CRM_PDT_OPP   AUC 0.810  Brier 0.203  acc 0.657  lift +7.4pp
  CRM_PDT_INC   AUC 0.854  Brier 0.174  acc 0.704  lift +12.0pp   (ΔAUC +0.044, ΔBrier −0.030)

Run:
  export PBI_ACCESS_TOKEN="$(az account get-access-token \
    --resource https://analysis.windows.net/powerbi/api --query accessToken -o tsv)"
  export PBI_CLIENT_SECRET=""; export PYTHONPATH=$PWD
  .venv/bin/python scripts/backtest_compare_models.py
"""
from predictive import schema as S
from predictive.pipeline import run_backtest

ASOF = "2026-06-06"
TEST_FRAC = 0.30


def _lift(res):
    """Accuracy lift in pp over always-predicting the majority test base rate."""
    base = max(res["test_base_rate"], 1 - res["test_base_rate"])
    return round((res["accuracy"] - base) * 100, 1)


def main():
    rows = []
    for mid in (S.MODEL_OPP, S.MODEL_INC):
        print(f"\n=== {mid} ===")
        res = run_backtest(None, asof=ASOF, test_frac=TEST_FRAC, model_id=mid)
        print(f"  n_train={res['n_train']} n_test={res['n_test']} "
              f"test_base_rate={res['test_base_rate']}")
        print(f"  AUC={res['auc']}  Brier={res['brier']}  acc={res['accuracy']}  "
              f"lift={_lift(res)}pp")
        print("  calibration (mean_pred vs actual):")
        for c in res["calibration"]:
            print(f"    {c['bin']}: n={c['n']:>4}  pred={c['mean_pred']:.3f}  "
                  f"actual={c['actual_rate']:.3f}")
        rows.append((mid, res))

    print("\n=== SUMMARY ===")
    print(f"{'model':<14}{'AUC':>8}{'Brier':>9}{'acc':>8}{'lift_pp':>9}")
    for mid, res in rows:
        print(f"{mid:<14}{res['auc']:>8}{res['brier']:>9}{res['accuracy']:>8}"
              f"{_lift(res):>9}")
    opp, inc = rows[0][1], rows[1][1]
    if opp["auc"] is not None and inc["auc"] is not None:
        print(f"\nΔAUC (INC − OPP) = {round(inc['auc'] - opp['auc'], 4):+}")
        print(f"ΔBrier (INC − OPP) = {round(inc['brier'] - opp['brier'], 4):+} "
              f"(negative = better)")


if __name__ == "__main__":
    main()
