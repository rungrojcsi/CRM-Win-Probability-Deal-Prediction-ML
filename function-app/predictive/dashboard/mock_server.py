"""mock_server.py — DEV-ONLY. Serves the static dashboard + canned /api/predictive/*
responses so the frontend can be previewed without the Azure Function backend.

Run:  python predictive/dashboard/mock_server.py   (serves on http://localhost:7071)
The dashboard's API box default (http://localhost:7071/api) points here.
NOT for production — real data comes from function_app.py endpoints.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


# Full mock SHAP driver sets — one per "persona" so different deals look different
# Each entry covers all 22 canonical features (26 for income-line rows with income-line group).
_DRIVERS_HIGH = [
    # Deal
    {"feature": "amount_log", "value": 15.26, "impact": 0.18},
    {"feature": "aging_days", "value": 42, "impact": -0.22},
    {"feature": "days_since_last_activity", "value": 3, "impact": 0.14},
    {"feature": "solution", "value": "MES", "impact": 0.15},
    {"feature": "prospect_category", "value": "Manufacturing IT", "impact": 0.12},
    # Activity
    {"feature": "activity_count_30d", "value": 8, "impact": 0.09},
    {"feature": "activity_count_90d", "value": 21, "impact": 0.07},
    {"feature": "activity_count_total", "value": 45, "impact": 0.06},
    {"feature": "activity_trend", "value": 0.8, "impact": 0.05},
    {"feature": "total_duration_mins", "value": 380, "impact": 0.04},
    {"feature": "distinct_activity_types", "value": 3, "impact": 0.03},
    {"feature": "meeting_count", "value": 4, "impact": 0.08},
    # Account history
    {"feature": "prior_won_count", "value": 2, "impact": 0.11},
    {"feature": "is_repeat_buyer", "value": 1, "impact": 0.10},
    # BANT
    {"feature": "bant_total", "value": 3, "impact": 0.06},
    {"feature": "competitiveness_score", "value": 0.7, "impact": -0.04},
    {"feature": "bant_has_data", "value": 1, "impact": 0.03},
    # Firmographic
    {"feature": "has_parent_account", "value": 0, "impact": -0.01},
    {"feature": "industry_l1", "value": "Automotive", "impact": 0.05},
    {"feature": "customer_level", "value": "A", "impact": 0.04},
    {"feature": "province", "value": "Rayong", "impact": 0.02},
    {"feature": "biz_sector", "value": "Manufacturing", "impact": 0.03},
]
_DRIVERS_MID = [
    {"feature": "amount_log", "value": 13.84, "impact": 0.06},
    {"feature": "aging_days", "value": 95, "impact": -0.13},
    {"feature": "days_since_last_activity", "value": 18, "impact": -0.09},
    {"feature": "solution", "value": "ERP", "impact": -0.03},
    {"feature": "prospect_category", "value": "ERP", "impact": -0.02},
    {"feature": "activity_count_30d", "value": 3, "impact": 0.04},
    {"feature": "activity_count_90d", "value": 9, "impact": 0.03},
    {"feature": "activity_count_total", "value": 18, "impact": 0.02},
    {"feature": "activity_trend", "value": 0.4, "impact": 0.01},
    {"feature": "total_duration_mins", "value": 120, "impact": 0.02},
    {"feature": "distinct_activity_types", "value": 2, "impact": 0.01},
    {"feature": "meeting_count", "value": 2, "impact": 0.03},
    {"feature": "prior_won_count", "value": 0, "impact": -0.05},
    {"feature": "is_repeat_buyer", "value": 0, "impact": -0.04},
    {"feature": "bant_total", "value": 1, "impact": -0.02},
    {"feature": "competitiveness_score", "value": 0.5, "impact": -0.03},
    {"feature": "bant_has_data", "value": 1, "impact": 0.01},
    {"feature": "has_parent_account", "value": 1, "impact": 0.02},
    {"feature": "industry_l1", "value": "Retail", "impact": -0.02},
    {"feature": "customer_level", "value": "B", "impact": -0.01},
    {"feature": "province", "value": "Bangkok", "impact": 0.01},
    {"feature": "biz_sector", "value": "Retail", "impact": -0.01},
]
_DRIVERS_LOW = [
    {"feature": "amount_log", "value": 15.45, "impact": -0.04},
    {"feature": "aging_days", "value": 280, "impact": -0.31},
    {"feature": "days_since_last_activity", "value": 62, "impact": -0.19},
    {"feature": "solution", "value": "IoT", "impact": -0.08},
    {"feature": "prospect_category", "value": "IoT", "impact": -0.07},
    {"feature": "activity_count_30d", "value": 0, "impact": -0.08},
    {"feature": "activity_count_90d", "value": 2, "impact": -0.06},
    {"feature": "activity_count_total", "value": 7, "impact": -0.05},
    {"feature": "activity_trend", "value": 0.1, "impact": -0.04},
    {"feature": "total_duration_mins", "value": 40, "impact": -0.03},
    {"feature": "distinct_activity_types", "value": 1, "impact": -0.02},
    {"feature": "meeting_count", "value": 0, "impact": -0.07},
    {"feature": "prior_won_count", "value": 0, "impact": -0.06},
    {"feature": "is_repeat_buyer", "value": 0, "impact": -0.05},
    {"feature": "bant_total", "value": 0, "impact": -0.04},
    {"feature": "competitiveness_score", "value": 0.3, "impact": 0.02},
    {"feature": "bant_has_data", "value": 0, "impact": -0.03},
    {"feature": "has_parent_account", "value": 0, "impact": -0.01},
    {"feature": "industry_l1", "value": "Chemical", "impact": -0.02},
    {"feature": "customer_level", "value": "C", "impact": -0.03},
    {"feature": "province", "value": "Chonburi", "impact": 0.01},
    {"feature": "biz_sector", "value": "Chemical", "impact": -0.02},
]
# Income-line rows also carry income-line group features
_DRIVERS_INCOME_HIGH = _DRIVERS_HIGH + [
    {"feature": "income_line_count", "value": 3, "impact": 0.07},
    {"feature": "income_total_p_log", "value": 17.15, "impact": 0.09},
    {"feature": "income_line_month_spread", "value": 4, "impact": 0.03},
    {"feature": "income_has_multi_line", "value": 1, "impact": 0.04},
]
_DRIVERS_INCOME_MID = _DRIVERS_MID + [
    {"feature": "income_line_count", "value": 2, "impact": 0.02},
    {"feature": "income_total_p_log", "value": 16.59, "impact": 0.03},
    {"feature": "income_line_month_spread", "value": 2, "impact": 0.01},
    {"feature": "income_has_multi_line", "value": 1, "impact": 0.01},
]
_DRIVERS_INCOME_LOW = _DRIVERS_LOW + [
    {"feature": "income_line_count", "value": 1, "impact": -0.01},
    {"feature": "income_total_p_log", "value": 15.14, "impact": -0.02},
    {"feature": "income_line_month_spread", "value": 1, "impact": -0.01},
    {"feature": "income_has_multi_line", "value": 0, "impact": -0.02},
]

_BAND_DRIVERS = {"High": _DRIVERS_HIGH, "Mid": _DRIVERS_MID, "Low": _DRIVERS_LOW}

DEALS = {
    "count": 130, "returned": 11, "at_risk": 847,
    "at_risk_raw": 742_100_000, "at_risk_weighted": 296_200_000,
    "bands": {"High": 545, "Mid": 249, "Low": 601},
    "weighted_pipeline": 62_300_000, "raw_pipeline": 182_400_000,
    "deals": [
        {"opp_id": f"AB{i:02d}CD12-7788",
         "account_name": acct, "opp_name": opp,
         "win_pct": p, "win_prob": p / 100,
         "band": b, "amount": a, "status": "Open",
         "so_plan_date": f"2026-0{6 + (i % 4)}-15", "at_risk": False,
         "drivers": _BAND_DRIVERS[b]}
        # SYNTHETIC demo names only — never real customers (no PII in committed code).
        for i, (p, b, a, acct, opp) in enumerate([
            (94, "High", 4_200_000, "Acme Manufacturing Co.", "MES Phase 1"),
            (88, "High", 2_100_000, "Beta Industries Ltd.", "FLEX Implementation"),
            (81, "High", 1_850_000, "Gamma Tech Co.", "Lamp Support"),
            (72, "High", 3_300_000, "Delta Motors Co.", "SCADA Upgrade"),
            (66, "Mid", 980_000, "Epsilon Steel Co.", "QC Vision"),
            (61, "Mid", 1_400_000, "Zeta Retail Co.", "ERP Add-on"),
            (54, "Mid", 760_000, "Eta Electronics Co.", "IoT Sensor"),
            (48, "Mid", 2_200_000, "Theta Storage Co.", "kintone Rollout"),
            (33, "Low", 5_100_000, "Iota Chemical Co.", "Data Platform"),
            (24, "Low", 7_500_000, "Kappa Electric Co.", "Smart Factory"),
            (19, "Low", 950_000, "Lambda Micro Co.", "Vision QC")])
    ],
}
FORECAST = {
    "method": "trailing-median + damped-trend (baseline)", "last_complete_month": "2026-05",
    "history_months": 18,
    "forecast": [
        {"month": "2026-06", "forecast": 60_390_000, "lower": 45_660_000, "upper": 75_120_000},
        {"month": "2026-07", "forecast": 60_390_000, "lower": 39_560_000, "upper": 81_220_000},
        {"month": "2026-08", "forecast": 59_960_000, "lower": 34_450_000, "upper": 85_470_000},
    ],
}
ATTAIN = {
    "month": "2026-05", "month_fraction_elapsed": 0.9, "low_confidence": False, "note": None,
    "team": {"target": 39_177_973, "actual_mtd": 44_200_000, "predicted_eom": 49_100_000, "attainment_pct": 125.3},
    "by_sales": [
        {"sales_id": "S1AAAA-11", "sales_name": "Demo Rep A", "target": 2_764_000, "actual_mtd": 3_900_000, "predicted_eom": 4_330_000, "attainment_pct": 156.7, "status": "ahead"},
        {"sales_id": "S2BBBB-22", "sales_name": "Demo Rep B", "target": 1_500_000, "actual_mtd": 1_450_000, "predicted_eom": 1_610_000, "attainment_pct": 107.3, "status": "on_track"},
        {"sales_id": "S3CCCC-33", "sales_name": "Demo Rep C", "target": 3_000_000, "actual_mtd": 1_200_000, "predicted_eom": 1_330_000, "attainment_pct": 44.3, "status": "behind"},
        {"sales_id": "S4DDDD-44", "sales_name": "Demo Rep D", "target": 0, "actual_mtd": 800_000, "predicted_eom": 890_000, "attainment_pct": None, "status": "no_target"},
    ],
}


# at-risk (header past-due, open) deals — for the Opportunity Delay Prospects block
DEALS_ATRISK = {
    "count": 0, "returned": 3, "at_risk": 847,
    "at_risk_raw": 742_100_000, "at_risk_weighted": 296_200_000,
    "bands": {"High": 0, "Mid": 0, "Low": 0},
    "weighted_pipeline": 0, "raw_pipeline": 0,
    "deals": [
        {"opp_id": "ZZ01", "account_name": "Acme Manufacturing Co.", "opp_name": "Auto Call 2026",
         "win_pct": 82.9, "win_prob": 0.829, "band": "High", "amount": 62_132_200,
         "status": "Open", "so_plan_date": "2026-05-01", "at_risk": True,
         "drivers": _DRIVERS_HIGH},
        {"opp_id": "ZZ02", "account_name": "Beta Industries Ltd.", "opp_name": "Legacy Migration",
         "win_pct": 35.0, "win_prob": 0.35, "band": "Low", "amount": 18_000_000,
         "status": "Open", "so_plan_date": "2026-03-31", "at_risk": True,
         "drivers": _DRIVERS_LOW},
        {"opp_id": "ZZ03", "account_name": "Gamma Tech Co.", "opp_name": "Phase 0",
         "win_pct": 58.0, "win_prob": 0.58, "band": "Mid", "amount": 9_500_000,
         "status": "Open", "so_plan_date": "2026-04-15", "at_risk": True,
         "drivers": _DRIVERS_MID},
    ],
}


def _m(month, typ, expected, conservative, optimistic, raw):
    return {"month": month, "type": typ, "expected": expected,
            "conservative": conservative, "optimistic": optimistic, "raw": raw}


SO_FORECAST = {
    "asof": "2026-06-06", "year": 2026, "current_month": "2026-06", "n_sims": 10000,
    "elapsed_actual": 198_300_000,
    "yearly_expected": 395_700_000,
    "yearly_conservative": 305_400_000,   # P10
    "yearly_optimistic": 470_900_000,     # P90
    "pipeline_plan": 700_300_000,
    "n_deals": 418,
    "bands": {"High": 230, "Mid": 110, "Low": 78},
    "at_risk": {"weighted": 296_200_000, "raw": 742_100_000},
    "months": [
        _m("2026-01", "actual", 43_500_000, 43_500_000, 43_500_000, 43_500_000),
        _m("2026-02", "actual", 28_700_000, 28_700_000, 28_700_000, 28_700_000),
        _m("2026-03", "actual", 60_000_000, 60_000_000, 60_000_000, 60_000_000),
        _m("2026-04", "actual", 45_200_000, 45_200_000, 45_200_000, 45_200_000),
        _m("2026-05", "actual", 21_000_000, 21_000_000, 21_000_000, 21_000_000),
        _m("2026-06", "predicted", 68_600_000, 41_200_000, 96_300_000, 182_400_000),
        _m("2026-07", "predicted", 37_500_000, 19_800_000, 58_100_000, 134_800_000),
        _m("2026-08", "predicted", 23_500_000, 10_100_000, 39_400_000, 100_800_000),
        _m("2026-09", "predicted", 23_900_000, 10_500_000, 40_000_000, 109_200_000),
        _m("2026-10", "predicted", 19_600_000, 7_800_000, 33_900_000, 71_100_000),
        _m("2026-11", "predicted", 12_000_000, 3_900_000, 22_700_000, 56_900_000),
        _m("2026-12", "predicted", 12_300_000, 4_100_000, 23_000_000, 67_500_000),
    ],
}


def _src_months(raw_by_m, wf=0.4):
    out = []
    for mm in range(1, 13):
        m = f"2026-{mm:02d}"
        r = raw_by_m.get(mm, 0)
        out.append({"month": m, "raw": r, "expected": round(r * wf)})
    return out


# real P-only vectors measured live (raw); expected ≈ raw × ~win-prob
SO_SOURCE = {
    "incomeplan": {
        "source": "incomeplan", "asof": "2026-06-06", "year": 2026, "current_month": "2026-06",
        "yearly_raw": 580_000_000, "yearly_expected": 430_000_000, "elapsed_actual": 198_300_000,
        "bands": {"High": 545, "Mid": 249, "Low": 601}, "n_opps": 760, "n_lines": 426,
        "months": (
            # elapsed (Jan–May) = realized SO Actual
            [{"month": f"2026-{m:02d}", "type": "actual", "raw": a, "expected": a}
             for m, a in {1: 43_525_833, 2: 28_673_475, 3: 59_723_093, 4: 45_411_366, 5: 20_583_432}.items()]
            # current+future (Jun–Dec) = predicted Expected (raw = prospect pipeline)
            + [{"month": f"2026-{m:02d}", "type": "predicted", "raw": r, "expected": round(r * 0.4)}
               for m, r in {6: 143_702_962, 7: 117_467_863, 8: 101_828_976, 9: 111_871_778,
                            10: 77_382_123, 11: 56_916_811, 12: 68_313_669}.items()]
        ),
        "lines": [  # SYNTHETIC — current..next-3 only, for the by-customer tables
            {"opp_id": "AB01", "account_name": "Acme Manufacturing Co.", "opp_name": "MES Phase 1 (P)",
             "ym": "2026-06", "amount": 28_000_000, "win_prob": 0.82, "win_pct": 82.0, "band": "High",
             "drivers": _DRIVERS_INCOME_HIGH},
            {"opp_id": "AB02", "account_name": "Beta Industries Ltd.", "opp_name": "FLEX (P)",
             "ym": "2026-06", "amount": 16_000_000, "win_prob": 0.55, "win_pct": 55.0, "band": "Mid",
             "drivers": _DRIVERS_INCOME_MID},
            {"opp_id": "AB03", "account_name": "Gamma Tech Co.", "opp_name": "SCADA (P)",
             "ym": "2026-06", "amount": 9_500_000, "win_prob": 0.30, "win_pct": 30.0, "band": "Low",
             "drivers": _DRIVERS_INCOME_LOW},
            {"opp_id": "2C046621-AAAA", "account_name": None, "opp_name": "Unnamed-Account Project",
             "ym": "2026-06", "amount": 5_000_000, "win_prob": 0.0, "win_pct": 0.0, "band": "Low",
             "drivers": []},
            {"opp_id": "AB04", "account_name": "Delta Motors Co.", "opp_name": "Line Upgrade (P)",
             "ym": "2026-07", "amount": 21_000_000, "win_prob": 0.71, "win_pct": 71.0, "band": "High",
             "drivers": _DRIVERS_INCOME_HIGH},
            {"opp_id": "AB05", "account_name": "Epsilon Steel Co.", "opp_name": "QC Vision (P)",
             "ym": "2026-07", "amount": 7_200_000, "win_prob": 0.42, "win_pct": 42.0, "band": "Mid",
             "drivers": _DRIVERS_INCOME_MID},
            {"opp_id": "AB06", "account_name": "Zeta Retail Co.", "opp_name": "ERP Add-on (P)",
             "ym": "2026-08", "amount": 12_500_000, "win_prob": 0.36, "win_pct": 36.0, "band": "Low",
             "drivers": _DRIVERS_INCOME_LOW},
        ],
        "delayed_total": 80_000_000,
        "delayed": [  # past-due income-lines (Jan–May) of still-open deals
            {"opp_id": "ZZ01", "account_name": "Acme Manufacturing Co.", "opp_name": "Auto Call 2026",
             "ym": "2026-05", "amount": 62_000_000, "win_prob": 0.829, "win_pct": 82.9, "band": "High",
             "drivers": _DRIVERS_INCOME_HIGH},
            {"opp_id": "ZZ02", "account_name": "Beta Industries Ltd.", "opp_name": "Legacy Migration",
             "ym": "2026-03", "amount": 12_000_000, "win_prob": 0.35, "win_pct": 35.0, "band": "Low",
             "drivers": _DRIVERS_INCOME_LOW},
            {"opp_id": "ZZ03", "account_name": "Gamma Tech Co.", "opp_name": "Phase 0",
             "ym": "2026-04", "amount": 6_000_000, "win_prob": 0.58, "win_pct": 58.0, "band": "Mid",
             "drivers": _DRIVERS_INCOME_MID},
        ],
    },
    "soplan": {
        "source": "soplan", "asof": "2026-06-06", "year": 2026,
        "yearly_raw": 61_500_000, "yearly_expected": 22_100_000,
        "bands": {"High": 120, "Mid": 90, "Low": 80}, "n_opps": 290, "n_lines": 290,
        "months": _src_months({1: 22_129_385, 2: 5_815_935, 3: 6_280_304, 4: 15_827_479,
                               5: 1_437_941, 6: 2_296_400, 7: 0, 8: 0, 9: 0, 10: 1_929_000,
                               11: 0, 12: 0}),
    },
}


# --- model-aware mock: CRM_PDT_MIX returns visibly different numbers ----------
# CRM_PDT_BASE = current values (unchanged). CRM_PDT_MIX = scale money ~1.1x and
# nudge win-prob up ~1.05x (capped) so switching the dropdown is obviously different.
_MONEY_KEYS = {
    "amount", "weighted_pipeline", "raw_pipeline", "at_risk_raw", "at_risk_weighted",
    "elapsed_actual", "yearly_expected", "yearly_conservative", "yearly_optimistic",
    "yearly_raw", "pipeline_plan", "delayed_total", "expected", "conservative",
    "optimistic", "raw", "weighted",
}


def _scale_inc(obj):
    """Recursively return a copy of a mock payload scaled for CRM_PDT_MIX."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _MONEY_KEYS and isinstance(v, (int, float)):
                out[k] = round(v * 1.1)
            elif k == "win_prob" and isinstance(v, (int, float)):
                out[k] = round(min(v * 1.05, 1.0), 4)
            elif k == "win_pct" and isinstance(v, (int, float)):
                out[k] = round(min(v * 1.05, 100.0), 1)
            else:
                out[k] = _scale_inc(v)
        return out
    if isinstance(obj, list):
        return [_scale_inc(x) for x in obj]
    return obj


def _by_model(payload, model):
    return _scale_inc(payload) if model == "CRM_PDT_MIX" else payload


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        p = self.path.split("?")[0]
        q = parse_qs(urlparse(self.path).query)
        model = q.get("model", ["CRM_PDT_BASE"])[0]
        if p.startswith("/api/predictive/so-source"):
            src = q.get("source", ["incomeplan"])[0]
            return self._send(_by_model(SO_SOURCE.get(src, SO_SOURCE["incomeplan"]), model))
        if p.startswith("/api/predictive/so-forecast"):
            return self._send(_by_model(SO_FORECAST, model))
        if p.startswith("/api/predictive/deals"):
            if q.get("at_risk", [""])[0].lower() == "true":
                return self._send(_by_model(DEALS_ATRISK, model))
            return self._send(_by_model(DEALS, model))
        if p.startswith("/api/predictive/forecast"):
            return self._send(FORECAST)
        if p.startswith("/api/predictive/attainment"):
            return self._send(ATTAIN)
        if p.startswith("/api/predictive/deal/"):
            return self._send(DEALS["deals"][0])
        fname = "index.html" if p in ("/", "") else p.lstrip("/")
        fpath = os.path.join(HERE, fname)
        if os.path.isfile(fpath):
            ctype = "text/html" if fname.endswith(".html") else (
                "text/css" if fname.endswith(".css") else "application/javascript")
            with open(fpath, "rb") as f:
                return self._send(f.read(), ctype)
        self.send_error(404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7071"))
    print(f"mock dashboard on http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
