/* CRM Predictive Dashboard — vanilla JS, consumes /api/predictive/* endpoints. */

const DEFAULT_API = "https://function-app.azurewebsites.net/api";
const $ = (id) => document.getElementById(id);
let currentTab = "incomeplan";   // last content tab shown (Settings overlays it)

const fmtBaht = (n) =>
  n == null ? "—" : "฿" + Math.round(n).toLocaleString("en-US");
const fmtM = (n) => (n == null ? "—" : "฿" + (n / 1e6).toFixed(1) + "M");

function apiBase() {
  return ($("apiBase").value || DEFAULT_API).replace(/\/+$/, "");
}

function withKey(url) {
  const key = $("apiKey").value.trim();
  if (!key) return url;
  return url + (url.includes("?") ? "&" : "?") + "code=" + encodeURIComponent(key);
}

function withModel(url) {
  // model is chosen in Settings (no top-bar dropdown); source of truth = localStorage
  const model = localStorage.getItem("dash_model") || "CRM_PDT_BASE";
  return url + (url.includes("?") ? "&" : "?") + "model=" + encodeURIComponent(model);
}

async function getJSON(path) {
  const res = await fetch(withKey(withModel(apiBase() + path)));
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return res.json();
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  setTimeout(() => (t.hidden = true), 5000);
}

function driverStr(drivers) {
  return (drivers || [])
    .map((d) => `${d.feature} (${d.impact > 0 ? "+" : ""}${d.impact})`)
    .join(", ");
}

/* ---------- SHAP feature groups ---------- */
const FEATURE_GROUPS = {
  amount_log: "Deal", aging_days: "Deal", days_since_last_activity: "Deal",
  solution: "Deal", prospect_category: "Deal",
  activity_count_30d: "Activity", activity_count_90d: "Activity",
  activity_count_total: "Activity", activity_trend: "Activity",
  total_duration_mins: "Activity", distinct_activity_types: "Activity",
  meeting_count: "Activity",
  prior_won_count: "AccHistory", is_repeat_buyer: "AccHistory",
  bant_total: "BANT", competitiveness_score: "BANT", bant_has_data: "BANT",
  has_parent_account: "BizProfile", industry_l1: "BizProfile",
  customer_level: "BizProfile", province: "BizProfile", biz_sector: "BizProfile",
  income_line_count: "Income-Line", income_total_p_log: "Income-Line",
  income_line_month_spread: "Income-Line", income_has_multi_line: "Income-Line",
};
const GROUP_ORDER = ["Deal", "Activity", "AccHistory", "BANT", "BizProfile", "Income-Line"];

// Always returns ALL groups in GROUP_ORDER (+ any extras). `sum` is null when the
// deal carries no feature for that group (e.g. Income-Line on a BASE-model deal),
// so the tooltip can show every group with "—" instead of hiding it.
function groupScores(drivers) {
  const ds = drivers || [];
  const sums = {}, present = {};
  ds.forEach((d) => {
    const g = FEATURE_GROUPS[d.feature] || "Other";
    sums[g] = (sums[g] || 0) + (d.impact || 0);
    present[g] = true;
  });
  const extra = Object.keys(sums).filter((g) => !GROUP_ORDER.includes(g));
  return [...GROUP_ORDER, ...extra].map((g) => ({
    group: g,
    sum: present[g] ? Math.round(sums[g] * 1000) / 1000 : null,
  }));
}

function renderGroupedDrivers(drivers) {
  if (!drivers || !drivers.length) return `<p class="muted">No drivers available.</p>`;
  const byGroup = {};
  drivers.forEach((d) => {
    const g = FEATURE_GROUPS[d.feature] || "Other";
    (byGroup[g] = byGroup[g] || []).push(d);
  });
  const order = [...GROUP_ORDER, "Other"].filter((g) => g in byGroup);
  return order.map((g) => {
    const feats = byGroup[g].slice().sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
    const gsum = feats.reduce((s, d) => s + (d.impact || 0), 0);
    const gsumStr = (gsum >= 0 ? "+" : "") + gsum.toFixed(3);
    const gCls = gsum >= 0 ? "imp-pos" : "imp-neg";
    const rows = feats.map((x) =>
      `<div class="driver">
        <span>${x.feature} = ${x.value ?? "—"}</span>
        <span class="${x.impact >= 0 ? "imp-pos" : "imp-neg"}">${x.impact >= 0 ? "+" : ""}${x.impact}</span>
      </div>`
    ).join("");
    return `<div class="shap-group">
      <div class="shap-group-head">
        <span class="shap-group-label">${g}</span>
        <span class="${gCls}">${gsumStr}</span>
      </div>
      ${rows}
    </div>`;
  }).join("");
}

/* ---------- SHAP hover tooltip (singleton) ---------- */
let _shapTip = null;
function _ensureShapTip() {
  if (_shapTip) return _shapTip;
  _shapTip = document.createElement("div");
  _shapTip.className = "shap-tip";
  _shapTip.hidden = true;
  document.body.appendChild(_shapTip);
  return _shapTip;
}
function showShapTip(evt, drivers) {
  if (!drivers || !drivers.length) return;
  const tip = _ensureShapTip();
  const rows = groupScores(drivers).map((gs) => {
    const val = gs.sum === null
      ? `<span class="muted">—</span>`
      : `<span class="${gs.sum >= 0 ? "imp-pos" : "imp-neg"}">${gs.sum >= 0 ? "+" : ""}${gs.sum.toFixed(3)}</span>`;
    return `<div class="shap-tip-row"><span class="shap-tip-group">${gs.group}</span>${val}</div>`;
  }).join("");
  tip.innerHTML = `<div class="shap-tip-title">SHAP by group</div>${rows}`;
  tip.hidden = false;
  _posShapTip(evt);
}
function _posShapTip(evt) {
  if (!_shapTip || _shapTip.hidden) return;
  const pad = 12;
  const tw = _shapTip.offsetWidth || 180;
  const th = _shapTip.offsetHeight || 100;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + tw > window.innerWidth - 8)  x = evt.clientX - tw - pad;
  if (y + th > window.innerHeight - 8) y = evt.clientY - th - pad;
  _shapTip.style.left = (x + window.scrollX) + "px";
  _shapTip.style.top  = (y + window.scrollY) + "px";
}
function hideShapTip() {
  if (_shapTip) _shapTip.hidden = true;
}

/* ---------- Zone 0: SO Forecast (monthly → yearly) ---------- */
async function loadSoForecast(asof) {
  const d = await getJSON(`/predictive/so-forecast?asof=${asof}`);

  // Two values compared: Expected (statistical mean) vs Conservative (P10 — 90%
  // chance of exceeding). Each deal is binary, so the conservative floor respects
  // the win-all/lose-all nature instead of trusting the point estimate alone.
  $("soYearlyKpis").innerHTML = `
    ${kpi("Yearly Expected SO", fmtM(d.yearly_expected))}
    ${kpi("Yearly Conservative (P10)", fmtM(d.yearly_conservative))}
    ${kpi("Actual (elapsed)", fmtM(d.elapsed_actual))}
    ${kpi("Yearly Prospect Pipeline", fmtM(d.pipeline_plan))}
    ${kpi("Deal Plan", d.n_deals)}`;

  // power bar — win-prob mix (High green / Mid amber / Low red) of the open deals
  // planned the rest of the year, same as the pipeline bar.
  const b = d.bands || { High: 0, Mid: 0, Low: 0 };
  const bt = (b.High + b.Mid + b.Low) || 1;
  $("annualBar").innerHTML =
    seg("high", b.High / bt) + seg("mid", b.Mid / bt) + seg("low", b.Low / bt);

  const fill = (id, rows) => {
    const tb = $(id).querySelector("tbody");
    tb.innerHTML = "";
    rows.forEach((r) => tb.insertAdjacentHTML("beforeend", r));
  };
  const months = d.months || [];
  fill(
    "soActualTable",
    months.filter((m) => m.type === "actual").map(
      (m) => `<tr><td>${m.month}</td><td class="num">${fmtM(m.expected)}</td></tr>`
    )
  );
  fill(
    "soPredTable",
    months.filter((m) => m.type === "predicted").map(
      (m) => {
        const wp = m.raw ? (m.expected / m.raw * 100).toFixed(1) : "0.0";
        return `<tr><td>${m.month}</td><td class="num muted">${fmtM(m.raw)}</td>
        <td class="num">${wp}%</td>
        <td class="num">${fmtM(m.expected)}</td>
        <td class="num muted">${fmtM(m.conservative)}</td></tr>`;
      }
    )
  );
}

/* ---------- Zone 0b/0c: current-month + next-3-months SO Plan deals ---------- */
function nextMonths(ym, n) {
  const [y, m] = ym.split("-").map(Number);
  const out = [];
  for (let i = 1; i <= n; i++) {
    const d = new Date(y, m - 1 + i, 1);
    out.push(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`);
  }
  return out;
}

async function loadSoDeals(asof) {
  const month = (asof || new Date().toISOString().slice(0, 10)).slice(0, 7);
  const data = await getJSON(`/predictive/deals?status=Open&limit=3000`);
  const deals = data.deals || [];
  const acct = (d) => d.account_name || d.opp_name || (d.opp_id || "").slice(0, 8) + "…";
  const months = nextMonths(month, 3);
  const n3 = deals.filter((d) => months.includes((d.so_plan_date || "").slice(0, 7)));

  // KPIs (Expected/Conservative summed from the so-forecast months — same MC as
  // the Annual/Current-Month views; SO Plan + Deal Plan from the deals).
  const sof = await getJSON(`/predictive/so-forecast?asof=${asof}`);
  const m3 = (sof.months || []).filter((m) => months.includes(m.month));
  const sum = (k) => m3.reduce((s, m) => s + (m[k] || 0), 0);
  $("next3Kpis").innerHTML = `
    ${kpi("3-Mo Expected SO", fmtM(sum("expected")))}
    ${kpi("3-Mo Conservative SO", fmtM(sum("conservative")))}
    ${kpi("3-Mo Prospect Pipeline", fmtM(sum("raw")))}
    ${kpi("Deal Plan", n3.length)}`;

  // band bar — win-prob mix of the next-3-month deals
  const b = { High: 0, Mid: 0, Low: 0 };
  n3.forEach((d) => { b[d.win_prob >= 0.7 ? "High" : d.win_prob < 0.4 ? "Low" : "Mid"]++; });
  const bt = (b.High + b.Mid + b.Low) || 1;
  $("next3Bar").innerHTML = seg("high", b.High / bt) + seg("mid", b.Mid / bt) + seg("low", b.Low / bt);

  $("next3Meta").textContent = `${n3.length} deals across ${months.join(", ")}`;
  const tb5 = $("next3Table").querySelector("tbody");
  tb5.innerHTML = "";
  n3.sort((a, b2) => (b2.amount || 0) - (a.amount || 0)).slice(0, 50).forEach((d, i) => {
    const sub = d.opp_name && d.account_name ? `<div class="muted">${d.opp_name}</div>` : "";
    tb5.insertAdjacentHTML("beforeend",
      `<tr><td class="num" data-sort="${i + 1}">${i + 1}</td><td data-sort="${acct(d)}">${acct(d)}${sub}</td>
        <td data-sort="${(d.so_plan_date || "")}">${(d.so_plan_date || "").slice(0, 7)}</td>
        <td class="num" data-sort="${d.amount || 0}">${fmtBaht(d.amount)}</td>
        <td class="num" data-sort="${d.win_prob}">${d.win_pct}%</td>
        <td data-sort="${d.win_prob}"><span class="badge b-${d.band.toLowerCase()}">${d.band}</span></td></tr>`);
  });
}

/* ---------- Zone 1: Pipeline ---------- */
async function loadPipeline(asof) {
  // Primary view = deals PLANNED to convert this month (SO Plan Date), the
  // actionable near-term pipeline. at_risk (past-due plan) is counted across ALL
  // open deals via a second lightweight call.
  const month = (asof || new Date().toISOString().slice(0, 10)).slice(0, 7);
  const limit = parseInt(($("pipeLimit") || {}).value, 10) || 50;
  // order=amount → the page is the top deals BY AMOUNT (not gated by win-prob top-N),
  // so big low-win-prob deals aren't hidden. Matches the IncomePlan table's behaviour.
  const data = await getJSON(`/predictive/deals?status=Open&so_plan_month=${month}&limit=${limit}&order=amount`);
  const bands = data.bands || { High: 0, Mid: 0, Low: 0 };

  // current-month Expected / Conservative come from the same Monte-Carlo source
  // as the Annual view (so-forecast), so the two sections reconcile.
  const sof = await getJSON(`/predictive/so-forecast?asof=${asof}`);
  const cm = (sof.months || []).find((m) => m.month === month) || {};

  $("pipelineKpis").innerHTML = `
    ${kpi("Month Expected SO", fmtM(cm.expected != null ? cm.expected : data.weighted_pipeline))}
    ${kpi("Month Conservative SO", fmtM(cm.conservative))}
    ${kpi("Monthly Prospect Pipeline", fmtM(cm.raw != null ? cm.raw : data.raw_pipeline))}
    ${kpi("Deal Plan", data.count)}`;

  const total = (bands.High + bands.Mid + bands.Low) || 1;
  $("bandBar").innerHTML =
    seg("high", bands.High / total) + seg("mid", bands.Mid / total) + seg("low", bands.Low / total);
  $("pipeMeta").textContent = `showing ${data.deals.length} of ${data.count} deals planned ${month}`;

  const tb = $("dealsTable").querySelector("tbody");
  tb.innerHTML = "";
  data.deals.sort((a, b) => (b.amount || 0) - (a.amount || 0));   // default sort by amount
  data.deals.forEach((d, i) => {
    const tr = document.createElement("tr");
    const name = d.account_name || d.opp_name || d.opp_id.slice(0, 8) + "…";
    const sub = d.opp_name && d.account_name ? `<div class="muted">${d.opp_name}</div>` : "";
    const hasDrivers = d.drivers && d.drivers.length;
    tr.innerHTML = `
      <td class="num" data-sort="${i + 1}">${i + 1}</td>
      <td data-sort="${name}">${name}${sub}</td>
      <td class="num" data-sort="${d.amount || 0}">${fmtBaht(d.amount)}</td>
      <td class="num wincell" data-sort="${d.win_prob}">${d.win_pct}%</td>
      <td data-sort="${d.win_prob}"><span class="badge b-${d.band.toLowerCase()}">${d.band}</span></td>
      <td class="muted">${driverStr(d.drivers)}</td>`;
    const winCell = tr.querySelector(".wincell");
    if (hasDrivers) {
      winCell.addEventListener("mouseenter", (e) => showShapTip(e, d.drivers));
      winCell.addEventListener("mousemove", _posShapTip);
      winCell.addEventListener("mouseleave", hideShapTip);
      winCell.addEventListener("click", (e) => {
        e.stopPropagation();
        openDriverPopup({
          title: name, sub: d.opp_name || "",
          win_pct: d.win_pct, band: d.band, drivers: d.drivers,
        });
      });
    }
    tr.onclick = () => openDeal(d.opp_id, d.drivers);
    tb.appendChild(tr);
  });
}

/* ---------- Deal drill-down ---------- */
async function openDeal(oppId, prefetchedDrivers) {
  try {
    const d = await getJSON(`/predictive/deal/${oppId}`);
    // Merge prefetched drivers if the fetch returns fewer (e.g. live endpoint still top-3)
    const drivers = (d.drivers && d.drivers.length >= (prefetchedDrivers || []).length)
      ? d.drivers : (prefetchedDrivers || d.drivers || []);
    $("modalBody").innerHTML = `
      <h3>${d.account_name || d.opp_id}</h3>
      <p class="muted">${[d.opp_name, d.status, fmtBaht(d.amount)].filter(Boolean).join(" · ")}</p>`
      + `<div class="muted" style="margin:-8px 0 10px;font-size:11px">${d.opp_id}</div>` + `
      <div class="kpis">
        ${kpi("Win Probability", d.win_pct + "%")}
        ${kpi("Band", d.band)}
      </div>
      <h4>Why this score (SHAP)</h4>
      ${renderGroupedDrivers(drivers)}`;
    $("dealModal").hidden = false;
  } catch (e) {
    toast(e.message);
  }
}

function openDriverPopup({ title, sub, win_pct, band, drivers }) {
  $("modalBody").innerHTML = `
    <h3>${title}</h3>
    <p class="muted">${sub || ""}</p>
    <div class="kpis">
      ${kpi("Win Probability", win_pct + "%")}
      ${kpi("Band", band)}
    </div>
    <h4>Why this score (SHAP)</h4>
    ${renderGroupedDrivers(drivers)}`;
  $("dealModal").hidden = false;
}

/* ---------- helpers ---------- */
const kpi = (label, value) =>
  `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div></div>`;
const seg = (cls, frac) => {
  const p = Math.round(frac * 100);
  return `<span class="seg-${cls}" style="width:${(frac * 100).toFixed(1)}%">${p >= 7 ? p + "%" : ""}</span>`;
};

// Click any column header to sort the table by that field (toggles asc/desc).
// Uses each cell's data-sort attribute (numeric when present, else text).
function enableSortable(table) {
  const ths = table.tHead.rows[0].cells;
  [...ths].forEach((th, idx) => {
    th.classList.add("th-sort");
    th.onclick = () => {
      const tb = table.tBodies[0];
      const asc = th.dataset.asc !== "true";
      [...ths].forEach((h) => { delete h.dataset.asc; h.classList.remove("sort-asc", "sort-desc"); });
      th.dataset.asc = asc;
      th.classList.add(asc ? "sort-asc" : "sort-desc");
      const val = (tr) => {
        const c = tr.cells[idx];
        const ds = c.getAttribute("data-sort");
        const raw = ds !== null ? ds : c.innerText;
        const n = parseFloat(raw);
        return raw !== "" && !isNaN(n) && /^[-\d.]+$/.test(raw) ? n : String(raw).toLowerCase();
      };
      [...tb.rows]
        .sort((a, b) => { const x = val(a), y = val(b); return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1); })
        .forEach((r) => tb.appendChild(r));
    };
  });
}

// Top-right Hide/Show button on each zone — collapses only the detail below the
// power bar (.zone-detail); KPIs + power bar stay visible.
function enableCollapsible() {
  document.querySelectorAll("section.zone .hide-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const collapsed = btn.closest("section.zone").classList.toggle("collapsed");
      btn.textContent = collapsed ? "Show" : "Hide";
    });
  });
}

/* ---------- TAB 2: Fact_IncomePlan (income-line grain) — Yearly + Monthly + Next3 ---------- */
const bandbar = (id, b) => {
  b = b || { High: 0, Mid: 0, Low: 0 };
  const t = (b.High + b.Mid + b.Low) || 1;
  $(id).innerHTML = seg("high", b.High / t) + seg("mid", b.Mid / t) + seg("low", b.Low / t);
};
// aggregate income-lines → one row per customer (opp) for a set of months
function customerRows(lines, months) {
  const by = {};
  lines.filter((l) => months.includes(l.ym)).forEach((l) => {
    const k = l.opp_id;
    if (!by[k]) by[k] = { ...l, amount: 0 };
    by[k].amount += l.amount || 0;
  });
  return Object.values(by).sort((a, b) => b.amount - a.amount);
}
function fillCustomerTable(tableId, rows, limit) {
  const acct = (d) => d.account_name || d.opp_name || (d.opp_id || "").slice(0, 8) + "…";
  const tb = $(tableId).querySelector("tbody");
  tb.innerHTML = "";
  const shown = limit && limit > 0 ? rows.slice(0, limit) : rows;   // 0/falsy → All
  shown.forEach((d, i) => {
    const sub = d.opp_name && d.account_name ? `<div class="muted">${d.opp_name}</div>` : "";
    const hasDrivers = d.drivers && d.drivers.length;
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="num" data-sort="${i + 1}">${i + 1}</td>
        <td data-sort="${acct(d)}">${acct(d)}${sub}</td>
        <td data-sort="${d.ym}">${d.ym}</td>
        <td class="num" data-sort="${d.amount || 0}">${fmtBaht(d.amount)}</td>
        <td class="num wincell" data-sort="${d.win_prob}">${d.win_pct}%</td>
        <td data-sort="${d.win_prob}"><span class="badge b-${(d.band || "").toLowerCase()}">${d.band}</span></td>`;
    const winCell = tr.querySelector(".wincell");
    if (hasDrivers) {
      winCell.addEventListener("mouseenter", (e) => showShapTip(e, d.drivers));
      winCell.addEventListener("mousemove", _posShapTip);
      winCell.addEventListener("mouseleave", hideShapTip);
      winCell.addEventListener("click", (e) => {
        e.stopPropagation();
        openDriverPopup({
          title: acct(d), sub: d.opp_name || "",
          win_pct: d.win_pct, band: d.band, drivers: d.drivers,
        });
      });
    }
    tb.appendChild(tr);
  });
}
const ipLimit = (id) => parseInt(($(id) || {}).value, 10);          // NaN-safe; 0 = All
let _ipMonthlyRows = [], _ipNext3Rows = [], _ipDelayedRows = [];     // cached for row-count re-render
function renderIpMonthly() {
  $("ipMonthlyMeta").textContent = `${_ipMonthlyRows.length} customers this month (income-line P) · sorted by amount`;
  fillCustomerTable("ipMonthlyTable", _ipMonthlyRows, ipLimit("ipMonthlyLimit"));
}
function renderIpNext3() {
  $("ipNext3Meta").textContent = `${_ipNext3Rows.length} customers · sorted by amount`;
  fillCustomerTable("ipNext3Table", _ipNext3Rows, ipLimit("ipNext3Limit"));
}
function renderIpDelayed() {
  $("ipDelayedMeta").textContent = `${_ipDelayedRows.length} past-due open customers (income-line) · sorted by amount`;
  fillCustomerTable("ipDelayedTable", _ipDelayedRows, ipLimit("ipDelayedLimit"));
}
async function loadIncomePlanTab(asof) {
  const d = await getJSON(`/predictive/so-source?source=incomeplan&asof=${asof}`);
  const months = d.months || [];
  const cur = d.current_month || (asof || "").slice(0, 7);
  const next3 = nextMonths(cur, 3);
  const lines = d.lines || [];

  // --- Yearly: Actual (elapsed) + Prediction (future), like Fact_Opportunity ---
  $("ipYearlyKpis").innerHTML = `
    ${kpi("Yearly Expected SO", fmtM(d.yearly_expected))}
    ${kpi("Actual (elapsed)", fmtM(d.elapsed_actual))}
    ${kpi("Yearly Prospect Pipeline", fmtM(d.yearly_raw))}
    ${kpi("Opportunities", d.n_opps)}`;
  bandbar("ipYearlyBar", d.bands);
  const fill = (id, rowsHtml) => {
    const tb = $(id).querySelector("tbody"); tb.innerHTML = ""; rowsHtml.forEach((r) => tb.insertAdjacentHTML("beforeend", r));
  };
  fill("ipActualTable", months.filter((m) => m.type === "actual").map(
    (m) => `<tr><td>${m.month}</td><td class="num">${fmtM(m.expected)}</td></tr>`));
  fill("ipPredTable", months.filter((m) => m.type === "predicted").map((m) => {
    const wp = m.raw ? (m.expected / m.raw * 100).toFixed(1) : "0.0";
    return `<tr><td>${m.month}</td><td class="num muted">${fmtM(m.raw)}</td>
      <td class="num">${wp}%</td><td class="num">${fmtM(m.expected)}</td></tr>`;
  }));

  // --- Monthly (current month) by customer ---
  const cm = months.find((m) => m.month === cur) || { raw: 0, expected: 0 };
  _ipMonthlyRows = customerRows(lines, [cur]);
  $("ipMonthlyKpis").innerHTML = `
    ${kpi("Month Expected SO", fmtM(cm.expected))}
    ${kpi("Monthly Prospect Pipeline", fmtM(cm.raw))}
    ${kpi("Customers", _ipMonthlyRows.length)}`;
  bandbar("ipMonthlyBar", bandsOf(_ipMonthlyRows));
  renderIpMonthly();

  // --- Next 3 months by customer ---
  const m3 = months.filter((m) => next3.includes(m.month));
  const sum = (k) => m3.reduce((s, m) => s + (m[k] || 0), 0);
  _ipNext3Rows = customerRows(lines, next3);
  $("ipNext3Kpis").innerHTML = `
    ${kpi("3-Mo Expected SO", fmtM(sum("expected")))}
    ${kpi("3-Mo Prospect Pipeline", fmtM(sum("raw")))}
    ${kpi("Customers", _ipNext3Rows.length)}`;
  bandbar("ipNext3Bar", bandsOf(_ipNext3Rows));
  renderIpNext3();

  // --- Delay Prospects (past-due income-lines, deal still open) ---
  const delayed = d.delayed || [];
  _ipDelayedRows = customerRows(delayed, [...new Set(delayed.map((l) => l.ym))]);
  const delExp = delayed.reduce((s, l) => s + (l.amount || 0) * (l.win_prob || 0), 0);
  $("ipDelayedKpis").innerHTML = `
    ${kpi("Delayed Total (P)", fmtM(d.delayed_total))}
    ${kpi("Delayed Expected", fmtM(delExp))}
    ${kpi("Customers", _ipDelayedRows.length)}`;
  bandbar("ipDelayedBar", bandsOf(_ipDelayedRows));
  renderIpDelayed();
}
// win-prob band mix of a set of customer rows
function bandsOf(rows) {
  const b = { High: 0, Mid: 0, Low: 0 };
  rows.forEach((r) => { b[r.win_prob >= 0.7 ? "High" : r.win_prob < 0.4 ? "Low" : "Mid"]++; });
  return b;
}

// Opportunity-tab Delay Prospects = at-risk (header past-due) open deals, by amount.
async function loadOppDelayed(asof) {
  const tot = await getJSON(`/predictive/deals?status=Open&limit=1`);   // totals only
  const lim = ipLimit("delayedLimit");
  const data = await getJSON(`/predictive/deals?status=Open&at_risk=true&order=amount&limit=${lim > 0 ? lim : 3000}`);
  const rows = (data.deals || []).map((d) => ({ ...d, ym: (d.so_plan_date || "").slice(0, 7) }));
  $("delayedKpis").innerHTML = `
    ${kpi("Delayed Total", fmtM(tot.at_risk_raw))}
    ${kpi("Delayed Expected", fmtM(tot.at_risk_weighted))}
    ${kpi("Deals", tot.at_risk)}`;
  bandbar("delayedBar", bandsOf(rows));
  $("delayedMeta").textContent = `${tot.at_risk} past-due open deals (header) · showing ${rows.length} · sorted by amount`;
  fillCustomerTable("delayedTable", rows, lim);
}

async function refreshAll() {
  const asof = $("asof").value;
  const activeBtn = document.querySelector(".tab-btn.active");
  if (!activeBtn) return;                      // Settings panel is open — nothing to load
  const active = activeBtn.dataset.tab;
  const loaders = active === "incomeplan"
    ? [["incomeplan", loadIncomePlanTab]]
    : [["so-forecast", loadSoForecast], ["so-deals", loadSoDeals], ["pipeline", loadPipeline], ["delayed", loadOppDelayed]];
  for (const [name, fn] of loaders) {
    try {
      await fn(asof);
    } catch (e) {
      toast(`${name}: ${e.message}`);
    }
  }
}

/* ---------- settings: tab visibility + model selection ---------- */
const CONTENT_TABS = ["incomeplan", "opp"];     // "settings" tab is always visible
function getVisibleTabs() {
  try {
    const v = JSON.parse(localStorage.getItem("dash_tabs"));
    if (Array.isArray(v) && v.length) return v.filter((t) => CONTENT_TABS.includes(t));
  } catch (e) { /* ignore bad json */ }
  return [...CONTENT_TABS];
}
function applyTabVisibility() {
  const vis = getVisibleTabs();
  CONTENT_TABS.forEach((t) => {
    const btn = document.querySelector(`.tab-btn[data-tab="${t}"]`);
    if (btn) btn.style.display = vis.includes(t) ? "" : "none";
  });
  // if the currently-active content tab got hidden, jump to the first visible one
  const active = document.querySelector(".tab-btn.active");
  if (active && active.dataset.tab !== "settings" && !vis.includes(active.dataset.tab)) {
    const first = document.querySelector(`.tab-btn[data-tab="${vis[0]}"]`);
    if (first) first.click();
  }
}
function setModel(model) {
  if ($("modelSel")) $("modelSel").value = model;
  document.querySelectorAll('input[name="modelRadio"]').forEach((r) => (r.checked = r.value === model));
  localStorage.setItem("dash_model", model);
}
function openSettings() {
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== "panel-settings"));
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
  if ($("settingsBtn")) $("settingsBtn").classList.add("active");
}
function closeSettings() {
  if ($("settingsBtn")) $("settingsBtn").classList.remove("active");
  // return to the last content tab (or the first visible one if it got hidden)
  const vis = getVisibleTabs();
  const tab = vis.includes(currentTab) ? currentTab : vis[0];
  const btn = document.querySelector(`.tab-btn[data-tab="${tab}"]`);
  if (btn) btn.click();          // shows the panel + sets active + lazy-loads
}
function setupSettings() {
  // model radios persist only — the screen refreshes on Save (not per click)
  document.querySelectorAll('input[name="modelRadio"]').forEach((r) => {
    r.onchange = () => { if (r.checked) setModel(r.value); };
  });
  // tab-visibility checkboxes (applied live; data refresh happens on Save)
  const vis = getVisibleTabs();
  document.querySelectorAll(".tabToggle").forEach((cb) => {
    cb.checked = vis.includes(cb.value);
    cb.onchange = () => {
      let v = [...document.querySelectorAll(".tabToggle")].filter((c) => c.checked).map((c) => c.value);
      if (!v.length) { cb.checked = true; v = [cb.value]; toast("At least one tab must stay visible"); }
      localStorage.setItem("dash_tabs", JSON.stringify(v));
      applyTabVisibility();
    };
  });
  // gear button (top bar) opens Settings; Save applies + refreshes the screen
  if ($("settingsBtn")) $("settingsBtn").onclick = openSettings;
  if ($("settingsSave")) $("settingsSave").onclick = () => { closeSettings(); refreshAll(); };
}

/* ---------- init ---------- */
function init() {
  $("apiBase").value = localStorage.getItem("dash_api") || DEFAULT_API;
  $("apiKey").value = localStorage.getItem("dash_key") || "";
  setModel(localStorage.getItem("dash_model") || "CRM_PDT_BASE");
  $("asof").value = new Date().toISOString().slice(0, 10);
  $("refresh").onclick = () => {
    localStorage.setItem("dash_api", $("apiBase").value);
    localStorage.setItem("dash_key", $("apiKey").value);
    refreshAll();
  };
  $("modalClose").onclick = () => ($("dealModal").hidden = true);
  $("dealModal").onclick = (e) => {
    if (e.target.id === "dealModal") $("dealModal").hidden = true;
  };
  // row-count selectors → re-render with the chosen limit
  const pl = $("pipeLimit");
  if (pl) pl.onchange = () => loadPipeline($("asof").value).catch((e) => toast(e.message));
  const iml = $("ipMonthlyLimit");
  if (iml) iml.onchange = renderIpMonthly;     // re-slice cached rows (no refetch)
  const inl = $("ipNext3Limit");
  if (inl) inl.onchange = renderIpNext3;
  const idl = $("ipDelayedLimit");
  if (idl) idl.onchange = renderIpDelayed;     // re-slice cached rows (no refetch)
  const dl = $("delayedLimit");
  if (dl) dl.onchange = () => loadOppDelayed($("asof").value).catch((e) => toast(e.message));
  // make the deal tables click-to-sort (header handlers persist across re-renders)
  document.querySelectorAll("table.sortable").forEach(enableSortable);
  // wire collapse/expand on each section header
  enableCollapsible();
  // tab switching — show the matching panel, lazy-load its data on first view
  const loaded = { opp: false, incomeplan: false };
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.onclick = () => {
      const tab = btn.dataset.tab;
      currentTab = tab;
      if ($("settingsBtn")) $("settingsBtn").classList.remove("active");
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== `panel-${tab}`));
      if (!loaded[tab]) { loaded[tab] = true; refreshAll(); }
    };
  });
  loaded.incomeplan = true;   // panel-incomeplan is visible by default
  setupSettings();            // model radios + tab-visibility checkboxes
  applyTabVisibility();       // hide tabs the user turned off (persisted)
  refreshAll();
}
document.addEventListener("DOMContentLoaded", init);
