# SA Spec — ML Predictive Dashboard (CRM Sales)

**Project:** RAG_Azure · **Date:** 2026-06-05 · **Mode:** STRICT (4 phases) · **Status:** Confirmed — ready for build
**Data source:** Power BI `SALES DATA MODEL` (dataset `00000000-0000-0000-0000-000000000000`, workspace `SALES_DATA` `00000000-...`)
**Proposal origin:** Obsidian `00_Inbox/ML Predictive Dashboard - CRM Sales.md` (#work)

---

## Phase 1 — Requirements, NFR & Integration Landscape

### Business Context
ทีม sales/ผู้บริหาร CSI ตัดสินใจจาก pipeline ดิบใน Dynamics 365 CE (ผ่าน Power BI `SALES DATA MODEL`) — ไม่รู้ว่าดีลไหนจะปิดจริง เดือนนี้จะได้ยอดเท่าไหร่ ทีมจะถึงเป้าไหม. ต้องการ ML Predictive Dashboard ที่เปลี่ยน pipeline ดิบเป็นตัวเลขถ่วงน้ำหนักด้วยโอกาสจริง + เหตุผล (driver) เพื่อให้ sales action ได้ และ forecast แม่นขึ้น.

### Actors
- **Primary** — Sales Manager: ดู ranked deals + at-risk, จัดลำดับทีม
- **Primary** — Sales Executive Director (Boss): weighted pipeline + revenue forecast + target attainment
- **Secondary** — Sales Rep: win% + driver ดีลตัวเอง (ผ่าน agent/Teams)
- **Secondary** — Data/ML maintainer (Boss): retrain, monitor
- **System** — Foundry agent `crm-ragrs-agent`: narrate ภาษาไทย

### Business Goals
- **G1** — ทำนายโอกาสปิดดีลแต่ละรายการ + เหตุผล (จัดลำดับ + เร่งดีลถูกตัว)
- **G2** — forecast รายได้ล่วงหน้า 3 เดือน
- **G3** — บอกว่า sales แต่ละคน/ทีมจะถึง quota เดือนนี้ไหม
- **G4** *(revised Phase 4)* — ส่งผลทำนายผ่าน **External Web Dashboard เป็นช่องทางหลัก + Teams agent narration เป็นช่องทางเสริม**
- **G5** — ทุกผลทำนายต้อง explainable (มี driver)

### Constraints & NFRs
| Item | Type | Value / Target | Source |
|---|---|---|---|
| Region lock | Constraint | southeastasia | Project |
| Compute | Constraint | Azure Function Linux Consumption (Flex blocked) | Project |
| Dashboard | Constraint | **External Web (custom)** | Boss decision |
| Score write target | Constraint | **Postgres → REST API → Web** (ไม่ write กลับ PBI) | Boss decision |
| Re-score cadence | NFR | **Daily batch** | Boss decision |
| Data read | Constraint | PBI executeQueries (SP auth), refresh ≥1h | Project |
| Win-Prob perf | NFR | AUC ≥ 0.75 [Assumption] | Industry baseline |
| Forecast acc | NFR | MAPE ≤ 20% [Assumption] | Industry baseline |
| Dashboard load | NFR | < 5s [Assumption] | Industry |
| Explainability | NFR | ทุก score แสดง ≥3 SHAP drivers (hard, G5) | G5 |
| PDPA | NFR | internal-only, ไม่ export ชื่อลูกค้านอก tenant, train ใน Azure | PDPA |
| Users | Assumption | < 50 internal | Assumption |

### Integration Landscape
| System | Type | Current Role | Integration Needed |
|---|---|---|---|
| PBI `SALES DATA MODEL` | Data source | semantic model 31 tables | READ features (executeQueries) |
| Azure Function `function-app` | Compute | RAG pipeline | + ML train/score module |
| Postgres (pgvector) | Store | RAG vector store | + score store |
| External Web | Presentation | (ใหม่) | dashboard + REST API |
| Foundry agent | Presentation | NL→DAX Q&A | + narrate scores (P2) |
| Entra ID | Auth | — | OIDC login web |

### Data verification (read-only, 2026-06-05)
- 31 real tables. Win-Prob training feasibility CONFIRMED.
- `Fact_Opportunity` by Status: **Won 708 / Lost 465 = 1,173 closed (60/40 balance, trainable)**; Open 1,396 = inference set.
- **DATA-QUALITY TRAP:** column `IsWon` ใช้เป็น label ไม่ได้ — 400 Open deals มี IsWon=1 (computed flag). **ใช้ `Status` (Won/Lost) เป็น ground-truth label.**

---

## Phase 2 — To-Be Process Flow

### Happy Path
| Step | Actor | Action | Input | Output | Sys/Manual |
|---|---|---|---|---|---|
| 1 | Timer (daily) | trigger pipeline | schedule | run start | System |
| 2 | Azure Function | ดึง features จาก PBI | executeQueries | feature frame | System |
| 3 | Azure Function | build features + score 3 models | feature frame | win_prob, revenue_fc, attainment + SHAP | System |
| 4 | Azure Function | เขียน scores | predictions | rows ใน Postgres | System |
| 5 | REST API | serve scores | HTTP GET | JSON | System |
| 6 | External Web | render dashboard | API JSON | 3 zones | System |
| 7 | Sales Manager | ดู + จัดลำดับ | dashboard | decision | Manual |
| 8 | Sales Rep | ถามรายดีลใน Teams | NL question | Thai narration | System |

### Exception Flows
| Scenario | Trigger | Handling | Outcome |
|---|---|---|---|
| PBI refresh ไม่เสร็จ/ดึงไม่ได้ | executeQueries fail | retry + ใช้ score รอบก่อน (stale flag) | "ข้อมูล ณ <วันที่>" |
| feature ไม่ครบ | null features | impute + ลด confidence band | score มี caveat |
| Model drift | weekly eval | alert → retrain | model ใหม่ |
| Forecast span สั้น | TimesFM low conf | band กว้าง + เตือน | เห็น uncertainty |
| score store ล่ม | API 5xx | cached + error banner | graceful degrade |

---

## Phase 3 — System & Integration Design

### System Boundary
- **In-system:** feature builder, 3 ML models, SHAP explainer, score store, REST API, External Web, daily scheduler
- **External:** PBI (source), Foundry agent (narration), Entra ID (auth)

### Integration Strategy
| Integration | Direction | Protocol | Purpose | Priority |
|---|---|---|---|---|
| PBI ← Function | read | REST executeQueries (SP) | features รายวัน | P1 |
| Postgres ↔ Function | r/w | psycopg/SQLAlchemy | scores | P1 |
| Web ← API | read | HTTPS REST (JSON) | serve scores | P1 |
| Web → Entra | auth | OAuth2/OIDC | login CSI | P1 |
| Agent ← Postgres/API | read | REST/SQL | narrate รายดีล | P2 |
| Function (timer) | internal | Azure Timer | daily re-score | P1 |

### System Modules
| Module | Responsibility | Priority |
|---|---|---|
| M1 Data Ingest | ดึง+validate features จาก PBI | P1 |
| M2 Feature Builder | feature frame ต่อ entity | P1 |
| M3 Model–WinProb | train/score win-prob + SHAP | P1 |
| M4 Model–Forecast | revenue forecast 3mo (TimesFM) | P1 |
| M5 Model–Attainment | quota attainment รายคน | P1 |
| M6 Score Store | persist + version scores | P1 |
| M7 REST API | serve scores | P1 |
| M8 Web Dashboard | render 3 zones + drill-down | P1 |
| M9 Auth | OIDC + RBAC | P1 |
| M10 Agent Bridge | feed scores → Foundry narrate | P2 |
| M11 Ops/Monitor | log metrics (P1) / drift alert (P2) | P1/P2 |

### Data Model (Conceptual)
- SalesPerson → owns → Opportunity → belongs-to → Account
- Opportunity → has → PredictionScore (win_prob, drivers, scored_at)
- Account → has → RFM/ChurnScore *(Phase 2)*
- SalesPerson → has → AttainmentForecast (per month)
- (global) → RevenueForecast (per month, horizon)
- PredictionScore → versioned-by → ModelRun

### Key Attributes (PII tagging)
| Entity | Key Attributes | PII? |
|---|---|---|
| Opportunity (feature) | opp_id, account_id, sales_id, amount, possibility, progress, aging_days, cycle_days, flag_hot, last_activity, solution, **status (label)** | indirect |
| Account | account_id, **name**, recency, frequency, monetary | Yes |
| SalesPerson | sales_id, **name**, target_amount | Yes |
| PredictionScore | opp_id, win_prob, band, top_drivers[], scored_at, model_run_id | indirect |
| RevenueForecast | month, forecast, lower, upper, model_run_id | No |
| AttainmentForecast | sales_id, month, actual_mtd, target, predicted_eom, attainment_pct | indirect |
| ModelRun | run_id, model_type, metrics, trained_at, version | No |

---

## Phase 4 — Final Function List & Traceability

### A. Final Function List
| # | Module | Function | Description | Priority |
|---|---|---|---|---|
| F1 | M1 | fetch_features | executeQueries ดึง Opp/Activity/Invoice/SO/GoalMonth | P1 |
| F2 | M1 | validate_schema | เช็ค column/type/null + stale fallback | P1 |
| F3 | M2 | build_opp_features | feature ต่อดีล (aging, cycle, hot, activity count, amount) | P1 |
| F4 | M2 | build_account_rfm | RFM ต่อ account | P2 |
| F5 | M2 | build_timeseries | monthly revenue series | P1 |
| F6 | M3 | train_winprob | LightGBM บน 1,173 closed (label=Status) | P1 |
| F7 | M3 | score_winprob | score 1,396 Open deals | P1 |
| F8 | M3 | explain_shap | top-N drivers ต่อดีล | P1 |
| F9 | M4 | forecast_revenue | TimesFM 3mo + band | P1 |
| F10 | M5 | compute_pacing | actual MTD vs target รายคน | P1 |
| F11 | M5 | predict_eom | คาดยอดสิ้นเดือน + attainment % | P1 |
| F12 | M6 | upsert_scores | เขียน scores + version | P1 |
| F13 | M6 | get_latest / score_history | อ่าน score | P1 |
| F14 | M7 | api_deals | GET ranked deals + filter | P1 |
| F15 | M7 | api_forecast | GET revenue forecast | P1 |
| F16 | M7 | api_attainment | GET attainment รายคน/ทีม | P1 |
| F17 | M7 | api_deal_detail | GET รายดีล + SHAP | P1 |
| F18 | M8 | render_pipeline_zone | ranked deals + weighted pipeline | P1 |
| F19 | M8 | render_revenue_zone | forecast chart + band | P1 |
| F20 | M8 | render_attainment_zone | quota gauge รายคน | P1 |
| F21 | M8 | deal_detail_view | drill-down + drivers + action hint | P1 |
| F22 | M9 | login_oidc / rbac_filter | auth + กรองตาม role | P1 |
| F23 | M10 | enrich_agent_context | ส่ง scores ให้ agent narrate | P2 |
| F24a | M11 | log_metrics | บันทึก AUC/MAPE ทุก run ลง ModelRun | P1 |
| F24b | M11 | drift_alert | เตือน drift + retrain trigger | P2 |
| F25 | M0(timer) | daily_orchestrate | รัน F1→F12 รายวัน | P1 |

**P1 (MVP):** F1,F2,F3,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F24a,F25
**P2:** F4, F23, F24b

### B. Goal Coverage
| Goal | Functions | Coverage |
|---|---|---|
| G1 | F3,F6,F7,F8,F14,F17,F18,F21 | ✅ |
| G2 | F5,F9,F15,F19 | ✅ |
| G3 | F10,F11,F16,F20 | ✅ |
| G4 (revised) | F18–F21,F22,F23(P2) | ✅ |
| G5 | F8,F17,F21,F24a | ✅ |

### C. Out-of-Scope (MVP)
| Item | Reason |
|---|---|
| Deal-at-Risk, Churn/RFM (F4) | Phase 2 |
| Cross-sell, deal-velocity | Phase 3 (รอ Fact_OpportunityMovement history) |
| Agent Bridge (F23) | P2 |
| Auto drift-alert/retrain (F24b) | P2 |
| Write-back เข้า Power BI | Boss เลือก External Web |
| Real-time scoring | daily batch พอ |

### D. Risks & Dependencies
| Risk | L | I | Mitigation |
|---|---|---|---|
| `IsWon` label ใช้ไม่ได้ | confirmed | H | ใช้ `Status` (Won/Lost) |
| Forecast span สั้น ~20mo | M | M | TimesFM + band กว้าง + เตือน |
| External Web build effort ใหญ่ | H | M | deliver API+model ก่อน, web ตามหลัง |
| สิทธิ์ write Postgres / Entra auth | M | M | verify SP + app registration ก่อน build |
| PBI refresh/timeout | M | M | stale fallback + score รอบก่อน |
| Model drift | M | M | F24a log (P1) + F24b alert (P2) |
| PDPA ชื่อลูกค้าบน web | M | H | auth+RBAC, internal-only, region/tenant host |
| Open deal feature ไม่ครบ | M | L | impute + ลด confidence |

### E. Assumptions Log
External Web platform · daily re-score · scores → Postgres+API (ไม่ใช่ PBI) · AUC≥0.75 / MAPE≤20% / load<5s · users<50 internal · label=Status · training=1,173 (Won 708/Lost 465)

---

## Build Recommendation (next)
เริ่ม **vertical slice** แรก: `F1 → F2 → F3 → F6 → F7` (ingest → feature → train → score win-prob) ให้ได้ score จริงก่อน แล้วต่อ F8 (SHAP) → F12/F13 (store) → F14/F17 (API) → F18/F21 (web). Forecast (F9) + Attainment (F10/F11) เป็น track ขนาน.
