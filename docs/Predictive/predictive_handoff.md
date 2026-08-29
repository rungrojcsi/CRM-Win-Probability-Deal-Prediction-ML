# CRM Predictive — Engineering Handoff & Knowledge Base

> Canonical handoff for the CRM Predictive system (win-probability + SO forecast +
> dashboard). Read this first when resuming. Anonymized — no real client names.
> Last updated 2026-06-06. Companion memory:
> `~/.claude/projects/-Users-rojios-Documents-Claude-Projects-RAG-Azure/memory/`
> (`project-so-plan-predictive.md`, `project-winprob-backtest.md`).

## 1. Objective
Give sales leadership a **credible, explainable forecast of Sales Orders (SO)**:
- Per-deal **win probability** (0–100%) with a plain-language reason (SHAP).
- **SO Monthly → Yearly forecast** (project-price, "P"): realized actuals for
  elapsed months + probability-weighted prediction for the rest of the year, with
  a risk band (Expected vs Conservative P10).
- Supporting: revenue forecast, target attainment.
Primary user surface: a static web dashboard. Decision use today = **rank/prioritize
leads + set credible SO expectations**, NOT literal-% financial commitments.

## 2. Data sources & limitations
- **Source:** live Power BI semantic model **"SALES DATA MODEL"** (31 tables) via
  PBI REST `executeQueries` (DAX). Key tables: Fact_Opportunity (2,569: 709 Won /
  465 Lost / 1,395 Open), Fact_Activity (96,753), Fact_OpportunityMovement (3,229,
  has BANT), Dim_Account (11,503), Fact_Invoice (4,687), Fact_SalesOrder,
  Fact_GoalMonth, Dim_SalesPerson.
- **Limitations (critical):**
  1. **Maturation / right-censoring** — Won deals close fast (cycle P50 63d), Lost
     slow (P50 188d, P90 539d). Recent cohorts under-count losses → observed
     win-rate ~60% but the *resolved-cohort* reality is ~40%. Mitigated by a
     maturity cutoff (`MATURITY_DAYS=540`, train/eval on resolved cohorts only).
  2. **Amount variants P/H/K** — use **(P) = project price** (canonical, matches
     the CRM certified measures). The unsuffixed column = P+H+K (over-states).
     H = outsource, K = hardware/software. Verify aggregates with
     `EVALUATE INFO.VIEW.MEASURES()` (`INFO.MEASURES()` is blocked).
  3. **BANT shallow** — Fact_OpportunityMovement snapshots only ~8 days deep;
     0 of 1,174 closed deals have a pre-close BANT snapshot → BANT useless for
     training today, value accrues as snapshots accumulate.
  4. **Small mature training set** (~360 deals) → the overfit/accuracy ceiling.
  5. **No point-in-time snapshots** — aging-type features are built at a single
     `asof`, not per-deal-historical (documented caveat).

## 3. Models (`function-app/predictive/`)
- **Win-Probability** (`winprob.py`) — HistGradientBoostingClassifier (gradient-
  boosted trees), regularized + sigmoid-calibrated. Label = `Status` (Won=1/Lost=0).
  Trains on MATURE closed deals only. Per-deal SHAP drivers (`explain_winprob`).
- **SO Monthly→Yearly Forecast** (`so_forecast.py`) — elapsed months = realized SO
  Actual (P); current+future = Monte-Carlo over each open deal as Bernoulli(win_prob)
  win-all/lose-all (n_sims=10000) → Expected (mean) + Conservative (P10) + Optimistic
  (P90) per month and for the year; also pipeline_plan (raw), n_deals, bands.
- **Revenue Forecast** (`forecast.py`) — trailing-median + damped-trend (time series).
- **Target Attainment** (`attainment.py`) — per-salesperson pacing vs Fact_GoalMonth.
- Candidates not built: churn/at-risk scoring, lead/next-best-action, deal-velocity.

## 4. Features (`schema.py` FEATURE_COLUMNS — 22 total)
- **USED (numeric):** amount_log (= log SO Plan Amount **(P)**), aging_days,
  days_since_last_activity, prior_won_count, is_repeat_buyer, has_parent_account,
  + Group-1 activity (count 30/90/total, trend, duration, distinct types, meetings)
  + Group-3 BANT (bant_total, competitiveness_score, bant_has_data).
- **USED (categorical):** solution, prospect_category, industry_l1, customer_level,
  province, biz_sector.
- **Feature groups added** (`aux_features.py`, point-in-time): G1 Activity
  (Fact_Activity), G2 Account history TRIMMED to prior_won_count + is_repeat_buyer
  (the other 4 overfit — dropped), G3 BANT+Competitiveness (low coverage today),
  G4 Firmographic (the clearest signal). Empirically: G4 is the main lift; G1/G3
  add ~0 now (low coverage) but grow with data.
- **EXCLUDED — leakage/dead (do NOT re-add naively):** `Possibility` (static =100
  for all Won at close → label leakage, fake AUC ~0.99), `Progress` (free-text→NaN),
  `Sales Cycle Days` (not point-in-time), `Flag Hot` (constant 0 in training = dead).
  Possibility allowed only as a point-in-time TRAJECTORY.
- **Hard rule:** every feature must be computable while the deal is still Open
  (point-in-time); closed-deal features use the deal's close reference
  (SO Actual Date → Last Activity → Create), open deals use `asof`.

## 5. Methodology fixes (chronological — the "why" behind the numbers)
1. **Temporal backtest** (`backtest_winprob`) — train oldest, test newest 30%;
   report AUC + Brier + accuracy + calibration + **lift over majority baseline**
   (accuracy alone is misleading under class imbalance).
2. **Removed `flag_hot`** — constant across all training deals (dead); had been
   shown as the #1 driver on every deal. AUC unchanged → confirmed inert.
3. **Maturity cutoff (540d)** — fixed the censoring; AUC 0.68(censored, meaningless)
   → honest 0.735; revealed real win-rate ~40% (not 60%).
4. **(P) project-price amounts** — switched from base (P+H+K) to (P); raw SUM of (P)
   matches the CRM certified measure (Jan–May actual 198.3M).
5. **Regularization** — depth 4→3, l2 1→5, iter 300→200, min_samples_leaf→20;
   beat both calibration (which traded AUC) and feature pruning (G1/G3 inert) →
   AUC 0.757→0.796, fixed most overconfidence.
6. **Sigmoid (Platt) calibration** — prefit on a 25% holdout (monotonic → SHAP
   intact); pulled probabilities toward observed win-rates (top bands now well-
   calibrated). Accepted AUC trade for credibility (Boss's priority).
7. **Monte Carlo SO forecast** — each deal is binary (win-all/lose-all), so the
   point Expected Σ(p×amt) is the portfolio mean but hides variance → report a P10
   conservative band alongside.

## 6. Dashboard (static; 3 deal sections share one pattern: KPIs + win-prob band
   bar green/amber/red + sortable table)
- **Annual Predictive SO** — KPIs: Yearly Expected / Conservative (P10) / Actual
  (elapsed) / Yearly Pipeline SO Plan / Deal Plan. Power bar = High/Mid/Low mix of
  open deals now..Dec. Monthly tables: Actual (elapsed) + Prediction (SO Plan ·
  Expected · Conservative).
- **Current Month Predictive SO by Customer** — KPIs: Month Expected / Conservative
  / Monthly SO Plan / Deal Plan. Sortable deal table (row selector 25/50/100/200),
  cols Opportunity(+project) / Amount / Win% / Band / drivers.
- **Next 3 Month Predictive SO by Customer** — KPIs: 3-Mo Expected / Conservative /
  SO Plan / Deal Plan. Sortable table cols Account(+project) / Plan Month / SO Plan
  (P) / Win% / Band.
- Files: `predictive/dashboard/{index.html,app.js,styles.css}` (+ `mock_server.py`
  for offline preview via `.claude/launch.json` "predictive-dashboard-mock").
  Cache-bust assets with `?v=` on each deploy.

## 7. Measurement / accuracy (latest backtest, 2026-06-06; current model)
Temporal holdout, maturity-filtered, 22-feat + regularized + sigmoid-calibrated;
train 253 / test 108.
- **AUC (OOS) = 0.810** (good, ≥0.80) · **Brier 0.203** · Accuracy 65.7% vs
  baseline 58.3% → **lift +7.4 pp**.
- **Calibration:** high bands well-calibrated (pred 75%→actual 81%, 86%→88%);
  mid-low bands still read a bit high (small n).
- Live (open deals): Current Month Expected ฿110.3M, Annual Expected ฿505.1M.
- Production-grade ≈ **3.5/5** — "near decision-grade for High-band; ranking-grade
  for mid". Caveats: small test set (108) → wide margin; mid-band optimism.

## 8. Deploy / ops
- **Function App:** `function-app` (RG **RESOURCE_GROUP**). Routes:
  `/api/predictive/{deals,deal/{id},run,backtest,so-forecast,forecast,attainment}`.
  Deploy: `func azure functionapp publish function-app --build remote`
  (⚠ sometimes needs a 2nd publish to pick up edited modules — verify the response
  shape after deploy).
- **Dashboard:** Azure Storage static website **`crmstorage` $web** →
  https://crmstorage.z23.web.core.windows.net/ . Deploy = `az storage blob upload
  --account-name crmstorage --container-name '$web' --auth-mode login --overwrite
  --name <f> --file <f> --content-type <ct> --content-cache-control "no-cache,
  must-revalidate"`.
- **Serving runtime has pandas only** (no sklearn). ML training (`run_scoring`)
  runs LOCALLY: `.venv/bin/python` (sklearn/shap in `requirements-ml.txt`).
- **Re-score recipe** (writes to prod Postgres `pg-crm-app`, flexible-server):
  1. open a TEMP firewall rule for your public IP,
  2. fetch real `POSTGRES_CONN_STR` from `az functionapp config appsettings list`
     (local.settings.json secret is a placeholder),
  3. PBI token: `az account get-access-token --resource
     https://analysis.windows.net/powerbi/api` → `PBI_ACCESS_TOKEN`, set
     `PBI_CLIENT_SECRET=""`,
  4. `run_scoring(default_store(), asof=...)`,
  5. DELETE the firewall rule.
- **Tests:** `.venv/bin/python -m pytest tests/ -q` (129 passing). Test-first,
  synthetic fixtures only — never live data in tests.
- **Git:** branch off `main`, ff-merge back. Generated decks (`docs/Predictive/
  *.pdf,*.pptx`) are gitignored.

## 9. Known gaps / roadmap (prioritized)
1. **No automated retrain** — model is static until `run_scoring` is run manually
   (locally; the cloud runtime can't train). New monthly outcomes do NOT feed back
   on their own. → set up a scheduled monthly retrain in an ML-enabled environment.
2. **No prediction-log** — predictions vs realized outcomes aren't logged, so
   accuracy drift / true forward-validation can't be measured. → add a prediction
   log (opp_id, asof, prob, band) and join on close.
3. **Small mature training set (~360)** — the overfit/accuracy ceiling; improves as
   deals close and cohorts mature (≥540d). Expect AUC/calibration to improve with
   data, especially early.
4. **BANT (G3) + Activity (G1) low coverage** — wired but near-zero training value
   now; revisit in 2–3 months as snapshots/activity accumulate.
5. **Maturity lag** — a newly-created deal's outcome enters training only after its
   cohort matures (~1.5y); recently-closed OLD-cohort deals feed in fast.

## 10. Code map
`predictive/`: schema.py (columns, labels, FEATURE_COLUMNS, MATURITY_DAYS, band) ·
ingest.py (DAX pulls) · features.py (build_opp_features, is_mature,
mature_training_labels) · aux_features.py (4 feature-group builders) · winprob.py
(train/score/explain/backtest + calibrator) · so_forecast.py (Monte-Carlo
monthly→yearly) · forecast.py · attainment.py · pipeline.py (run_scoring,
run_backtest, _fetch_aux_sources) · store.py (Postgres + InMemory) · api.py
(list_deals, deal_detail) · dashboard/.
Tests: `tests/test_predictive_*.py`.
