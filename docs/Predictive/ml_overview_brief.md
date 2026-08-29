# CRM Predictive ML — Overview & Accuracy Assessment

**Audience:** Business and technical leadership
**Scope:** Predictive machine-learning layer built on the CRM sales data lake
**Note on confidentiality:** All client and account names are anonymized. Only aggregate facts and counts are shown.

---

## 1. ML Models — Built and Buildable

### Models already built (in production / pilot)

| Model | Status | Technique | What it produces |
|---|---|---|---|
| **Win-Probability (deal-level)** | BUILT | HistGradientBoostingClassifier (gradient-boosted trees) | Scores each open opportunity 0–100% chance of winning, plus a SHAP reason explaining the score |
| **SO Monthly→Yearly Forecast** | BUILT | Probability-weighted pipeline roll-up | For each future month: Σ(SO Plan Amount (P) × win_prob). For elapsed months: realized SO Actual. Combined into a yearly projection |
| **Revenue Forecast** | BUILT | Trailing-median + damped-trend baseline (time series) | Forward revenue projection from historical trend |
| **Target Attainment** | BUILT | Pacing vs target | Per-salesperson pace against the Fact_GoalMonth target, with an end-of-month projection |

### Models that COULD be built next (not yet built)

- **Account churn / at-risk scoring** — strong signal already present: 847 deals with past-due SO Plan.
- **Lead / next-best-action scoring** — prioritize which lead or action to pursue.
- **Calibration model** — to correct overconfident probabilities (build after the data fixes below).
- **Deal-velocity / time-to-close regression** — predict how long a deal will take to close.

---

## 2. Feature Taxonomy

Features fall into three buckets: what is used today, what is planned, and what is deliberately excluded.

### Bucket A — USED NOW (5 features)

1. **amount_log** — log of SO Plan Amount (P), i.e. the project price.
2. **aging_days** — how long the deal has been open.
3. **days_since_last_activity** — recency of the last logged activity.
4. **solution** — Solution Name (26 distinct types).
5. **prospect_category** — Existing Customer (New / Stock) vs New Customer.

### Bucket B — PLANNED / RECOMMENDED (being added)

- **Activity engagement** (Fact_Activity, 96,753 rows): activity count over 30/90 days, recency, trend, total duration, meeting count.
- **Account purchase history** (Fact_Invoice 4,687 + Fact_SalesOrder): prior won count, lifetime value, repeat-buyer flag, last-purchase recency, historical account win-rate.
- **BANT + Competitiveness scores** (Fact_OpportunityMovement, 3,229 rows): Budget, Authority, Need, Timing, and Competitiveness Score.
- **Account firmographic** (Dim_Account, 11,503 rows): Industry L1/L2, Customer Level, Province, Business Sector, Parent Account.
- **Possibility TRAJECTORY** — rising vs falling, captured point-in-time.
- **Salesperson track record.**
- **Lead source / campaign.**
- **Seasonality.**

### Bucket C — SHOULD-USE-BUT-EXCLUDED (leakage or dead)

| Field | Why excluded |
|---|---|
| **Possibility** | Static; equals 100 for 100% of Won deals at close → label leakage → produces a fake AUC of ~0.99 |
| **Progress** | Free-text → all NaN, unusable |
| **Sales Cycle Days** | Won 63d vs Lost 188d, but value is not known point-in-time (leakage) |
| **Flag Hot** | Constant 0 across all training deals → dead feature, removed |

---

## 3. Data Available + Limitations

### Data available

Source: 31 tables in the semantic model **"SALES DATA MODEL"**. Key tables:

| Table | Rows | Notes |
|---|---|---|
| Fact_Opportunity | 2,569 | 709 Won / 465 Lost / 1,395 Open |
| Fact_Activity | 96,753 | Engagement signal |
| Fact_OpportunityMovement | 3,229 | Contains BANT scores |
| Dim_Account | 11,503 | Firmographics |
| Fact_Invoice | 4,687 | Purchase history |
| Fact_SalesOrder | — | Plan/actual amounts |
| Fact_GoalMonth | — | Targets |
| Dim_SalesPerson | — | Salesperson dimension |

### Limitations (critical)

1. **Maturation / right-censoring.** Won deals close fast (cycle P50 = 63 days); Lost deals close slowly (P50 = 188 days, P90 = 539 days). Recent cohorts therefore under-count losses, inflating the observed win-rate to ~60% when the real (mature-cohort) rate is ~40%. Mitigated with a **MATURITY_DAYS = 540** cutoff.
2. **Amount has three variants** — P (project) / H (outsource) / K (hardware-software). Use **P as primary** (matches the CRM certified measures).
3. **BANT data is shallow** — only ~8 days of history as of mid-2026. Verify depth before relying on it.
4. **No point-in-time snapshots yet** — aging-based features are not fully per-deal-historical.
5. **Small mature training set** — only 346 deals remain after the maturity cutoff.

---

## 4. Measurements — Meaning + Expected Values

| Metric | What it means | Good value | Current |
|---|---|---|---|
| **AUC (ROC)** | Ranking ability. 0.50 = random, 0.80 = good, 1.00 = perfect. The only metric unaffected by class imbalance | ≥ 0.80 | **0.735** (honest, maturity-filtered) |
| **Brier score** | Mean squared error of the probability. Lower is better; 0 = perfect | low | **0.236** |
| **Accuracy@0.5** | % of predictions correct. Misleading under class imbalance — must be compared to baseline | beat baseline | **66.3%** |
| **Majority-class baseline** | Accuracy of always predicting the majority class — the bar to beat | n/a | **58.7%** |
| **Lift over baseline** | Accuracy minus baseline = real value added | positive | **+7.7 pp** |
| **Calibration** | Predicted vs actual win-rate per bin | aligned | Top bin predicted 96% → actual 72% (**overconfident at the top**) |

**Expected after the 4 new feature groups are added:** AUC target **0.82–0.85**.

---

## 5. Overall Accuracy Assessment

Production-grade scorecard (1 = prototype … 5 = production):

| Dimension | Score | Rationale |
|---|---|---|
| **Methodology** | 4/5 | Temporal holdout, leakage controls, maturity cutoff, test-first discipline |
| **Data quality** | 3/5 | Censoring handled, but small mature set and no point-in-time snapshots |
| **Discrimination** | 3/5 | AUC 0.735 — beats baseline, but below the 0.80 "good" line |
| **Calibration** | 2/5 | Top probability band is overconfident |
| **Explainability** | 4/5 | Per-deal SHAP reasons |
| **MLOps** | 2/5 | No prediction-log or monitoring yet |

**Overall ≈ 2.8 / 5 — "pilot / ranking-grade, NOT yet decision-grade."**

- **Safe to use** for ranking and prioritizing leads.
- **Do NOT use** the literal percentage for money decisions.
- **Path to production:** add the 4 feature groups + a prediction-log + recalibration.
