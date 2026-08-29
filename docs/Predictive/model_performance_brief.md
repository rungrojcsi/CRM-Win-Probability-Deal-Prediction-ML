# Win-Probability Model — Performance Brief (for slide deck)

> Source brief for a stakeholder slide deck. Audience: business + technical
> leadership. Goal: explain WHICH metrics we use, HOW each is used, WHERE each is
> computed, the ACTUAL values from our backtest, and a final PRODUCTION-GRADE
> readiness assessment. All examples are anonymized (no client names).

## 1. What the model is

- **Purpose:** score every OPEN sales opportunity with a probability of winning
  (0–100%), plus a plain-language reason, so sales leadership can prioritize.
- **Algorithm:** Gradient-Boosted Decision Trees (`HistGradientBoostingClassifier`).
- **Trained on:** historical CLOSED deals (Won/Lost) from the live CRM semantic
  model (SALES DATA MODEL). Label = deal Status (Won=1, Lost=0).
- **Features (5):** deal amount (log), aging days, days since last activity,
  solution type, prospect category. (A 6th feature, "hot flag", was removed —
  see §5.)
- **Data scale:** 2,569 opportunities total — 1,174 closed (709 Won + 465 Lost),
  1,395 open.

## 2. The metrics we use — what, how, where, value

Every value below is from the temporal backtest on the live model
(asof 2026-06-05, maturity-filtered; see §5).

### 2.1 Win Probability (the score itself)
- **What:** P(win) per deal, 0–100%.
- **How used:** rank Open deals; weight the pipeline (Σ amount × P).
- **Computed:** `score_winprob()` → `estimator.predict_proba()`.
- **Value range observed (open deals):** 0.1% … 99.8%, median ~53–64%.

### 2.2 Band (High / Mid / Low)
- **What:** human-friendly bucket of the score.
- **How used:** dashboard grouping / filtering.
- **Computed:** `band()` — High ≥ 0.70, Low < 0.40, else Mid.
- **Value (open deals, maturity-trained):** High 545 · Mid 249 · Low 601.

### 2.3 AUC — ROC Area Under Curve  ★ headline metric
- **What:** ability to RANK a random Won above a random Lost.
- **How used:** the single most trustworthy quality number — unaffected by class
  imbalance, so it survives a skewed win-rate.
- **Computed:** `roc_auc_score(y_test, p_test)` on the temporal hold-out.
- **Scale:** 0.50 = random, 0.80 = good, 1.00 = perfect.
- **Value: 0.735** (out-of-sample).

### 2.4 Brier Score
- **What:** mean squared error of the probability vs the 0/1 outcome.
- **How used:** overall probability quality (accuracy + calibration combined).
- **Computed:** mean((p − y)²).
- **Scale:** lower is better; 0 = perfect.
- **Value: 0.236.**

### 2.5 Accuracy @ 0.5
- **What:** % of test deals classified correctly at a 0.5 threshold.
- **How used:** intuitive but MISLEADING when classes are imbalanced — always
  read next to the baseline.
- **Computed:** mean((p ≥ 0.5) == y).
- **Value: 66.3%.**

### 2.6 Majority-Class Baseline
- **What:** accuracy of the dumbest model — always predict the majority class.
- **How used:** the bar the model must beat to add any value.
- **Computed:** max(win-rate, 1 − win-rate) on the test set.
- **Value: 58.7%.**

### 2.7 Lift over Baseline  ★ "did it add value?"
- **What:** accuracy minus baseline.
- **How used:** the honest "is the model worth it" signal.
- **Computed:** accuracy − baseline.
- **Value: +7.7 percentage points** (66.3% − 58.7%). Positive = real value.

### 2.8 Calibration table — "does the % mean what it says?"
- **What:** for each probability bin, predicted vs actual win-rate.
- **How used:** tells you whether "90%" really wins 90% of the time.
- **Computed:** `_calibration()` — bin predictions, compare mean predicted to
  actual outcome rate.
- **Value (key bins):**
  - 0.80–0.90 (n=20): predicted 87% → actual 80% (close)
  - 0.90–1.00 (n=46): predicted 96% → **actual 72%** (still overconfident ~24pp)
  - lower bins have too few samples (n=2–9) to read.

### 2.9 SHAP drivers — "why this score"
- **What:** per-deal contribution of each feature (log-odds), toward Win (+) or
  Lost (−).
- **How used:** the "Why this score" panel; auditability (no score without a
  reason).
- **Computed:** `explain_winprob()` → SHAP TreeExplainer, top-3 by magnitude.
- **Note:** values are log-odds contributions, NOT percentages, and do not sum
  to the displayed probability.

## 3. How the performance was measured (so the numbers are trustworthy)

- **Temporal hold-out (no look-ahead):** sort deals by time, train on the oldest
  70%, test on the newest 30%. A random split would leak future information.
- **Split:** train 242 → test 104 (mature pool 346 closed deals).
- **Reported together:** AUC + Brier + calibration + lift — never accuracy alone.

## 4. Headline result table

| Metric | Value | Reading |
|---|---|---|
| AUC (out-of-sample) | 0.735 | above random, below "good" 0.80 |
| Brier | 0.236 | moderate |
| Accuracy @0.5 | 66.3% | — |
| Baseline (majority) | 58.7% | the bar to beat |
| Lift over baseline | +7.7 pp | real value added |
| Win-rate train → test | 29.8% → 58.7% | residual upward trend |
| Top-bin calibration | 96% → 72% actual | overconfident at the top |

## 5. Why these numbers are honest (two data fixes)

1. **Removed a dead feature ("hot flag").** It was constant across 100% of
   training deals (only set on open deals), so it carried no signal — yet it was
   shown as the #1 reason on every deal. Removing it left AUC unchanged (proof it
   was dead) and cleaned up the explanations.

2. **Maturity cutoff (the big one).** Won deals close fast (~63 days), Lost deals
   close slowly (~188 days median, up to ~539). So recent deal-cohorts are
   "right-censored": their slow future losses are still open and unlabeled,
   making the win-rate look like 60% when fully-resolved cohorts win only ~40%.
   This censoring was the root cause of the 99%+ overconfidence. We now train and
   evaluate on MATURE cohorts only (created ≥ 540 days ago). Result: the honest
   AUC of 0.735 and +7.7pp lift — versus the censored model which sat at or below
   the baseline.

   (Probability calibration was also attempted and reverted — on in-distribution
   data it made overconfidence worse. The issue is data/label timing, not a
   probability-scale problem.)

## 6. Production-grade readiness assessment

Scored 1 (prototype) to 5 (production-grade).

| Dimension | Score | Rationale |
|---|---|---|
| Methodology rigor | 4 / 5 | temporal hold-out, leakage controls, censoring handled, test-first |
| Data quality | 3 / 5 | censoring found & mitigated, but small mature pool (346) and single-asof features (not point-in-time) |
| Discrimination (AUC 0.735) | 3 / 5 | beats chance and baseline, but below 0.80 |
| Calibration | 2 / 5 | top band overconfident (96%→72%); raw % not yet trustworthy |
| Explainability | 4 / 5 | per-deal SHAP, every score has a reason |
| MLOps / robustness | 2 / 5 | no prediction-log, no monitoring/drift, no point-in-time snapshots, small test set |

**Overall: ~2.8 / 5 — "Pilot / ranking-grade, NOT yet decision-grade."**

- ✅ **Safe to use** to PRIORITIZE and RANK leads (AUC 0.735, +7.7pp lift).
- ⚠️ **Do NOT use** the literal % for revenue commitments or money decisions —
  the high band is overconfident and the test set is small.
- **Path to production-grade:** (1) engagement features from the Activity log to
  lift AUC past 0.80; (2) prediction-log for true point-in-time backtesting and
  drift monitoring; (3) flag stale open deals (likely losses) as at-risk;
  (4) re-calibrate only AFTER the data/label timing is fully corrected.

## 7. One-line takeaway

A model that is now HONEST (it beats the baseline by +7.7pp with a clean AUC of
0.735) and well-understood — strong enough to rank and prioritize leads today,
but one calibration tier and one feature-engineering pass away from being
trusted for money decisions.
