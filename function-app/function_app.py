"""
Azure Function: CRM Predictive ML API.

Win-probability scoring + SO/revenue forecasting + target attainment.
Data is pulled live from the Power BI semantic model via transform.pbi_client.

Routes (all AuthLevel.FUNCTION — require a function key):
  GET  /api/predictive/deals              ranked win-probability deals
  GET  /api/predictive/deal/{opp_id}      one deal + SHAP drivers
  POST /api/predictive/run                (re)score open opportunities
  POST /api/predictive/backtest           honest backtest metrics
  GET  /api/predictive/so-forecast        SO monthly->yearly forecast
  GET  /api/predictive/so-source          SO source breakdown
  GET  /api/predictive/so-conversion-forecast
  GET  /api/predictive/forecast           revenue forecast
  GET  /api/predictive/attainment         per-salesperson target pacing
"""

import json
import logging
from datetime import date, datetime

import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Predictive — Win-Probability dashboard API (F14 deals, F17 detail, F25 run)
# ──────────────────────────────────────────────────────────────────────────

@app.route(route="predictive/deals", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_deals(req: func.HttpRequest) -> func.HttpResponse:
    """F14 — ranked win-probability deals from the score store.

    Query: status (default Open), band (High/Mid/Low), min_prob (0-1), limit (int),
    so_plan_month (YYYY-MM — deals planned to convert that month),
    order (win_prob | amount — ranking for the returned page; default win_prob).
    """
    from predictive.api import list_deals
    from predictive.store import default_store

    try:
        p = req.params
        status = p.get("status", "Open") or None
        band = p.get("band") or None
        min_prob = float(p["min_prob"]) if p.get("min_prob") else None
        limit = int(p.get("limit", 50))
        so_plan_month = p.get("so_plan_month") or None
        order_by = p.get("order") or "win_prob"
        at_risk = {"true": True, "false": False}.get((p.get("at_risk") or "").lower())
        model_id = p.get("model")
        out = list_deals(
            default_store(), status=status, band=band, min_prob=min_prob,
            limit=limit, so_plan_month=so_plan_month, order_by=order_by, at_risk=at_risk,
            model_id=model_id,
        )
        return func.HttpResponse(
            json.dumps(out, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/deals failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/deal/{opp_id}", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_deal(req: func.HttpRequest) -> func.HttpResponse:
    """F17 — one deal's win-probability + SHAP drivers."""
    from predictive.api import deal_detail
    from predictive.store import default_store

    opp_id = req.route_params.get("opp_id", "")
    try:
        detail = deal_detail(default_store(), opp_id, model_id=req.params.get("model"))
        if detail is None:
            return func.HttpResponse(
                json.dumps({"error": "not scored", "opp_id": opp_id}),
                status_code=404, mimetype="application/json",
            )
        return func.HttpResponse(
            json.dumps(detail, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/deal failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/run", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_run(req: func.HttpRequest) -> func.HttpResponse:
    """F25 — (re)train + score all open deals, persist to store. Body: {asof?}."""
    from datetime import date

    from predictive import schema as S
    from predictive.pipeline import run_scoring, run_so_scoring
    from predictive.store import default_store

    try:
        body = req.get_json() if req.get_body() else {}
        asof = body.get("asof") or date.today().isoformat()
        model_id = body.get("model") or req.params.get("model")
        if model_id == S.MODEL_AZ:           # SO-conversion model (label = Fact_SalesOrder)
            summary = run_so_scoring(default_store(), asof=asof)
        else:
            summary = run_scoring(default_store(), asof=asof, model_id=model_id)
        return func.HttpResponse(
            json.dumps(summary, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/run failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/backtest", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_backtest(req: func.HttpRequest) -> func.HttpResponse:
    """Temporal holdout backtest on historical closed deals.

    Body: {asof?, test_frac?}. Trains on the earliest (1 - test_frac) of closed
    deals, evaluates on the most recent test_frac. Returns out-of-sample AUC,
    Brier, accuracy, base rates and a calibration table.
    """
    from datetime import date

    from predictive.pipeline import run_backtest
    from predictive.store import default_store

    try:
        body = req.get_json() if req.get_body() else {}
        asof = body.get("asof") or date.today().isoformat()
        test_frac = float(body.get("test_frac", 0.30))
        model_id = body.get("model") or req.params.get("model")
        res = run_backtest(default_store(), asof=asof, test_frac=test_frac, model_id=model_id)
        return func.HttpResponse(
            json.dumps(res, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/backtest failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/so-forecast", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_so_forecast(req: func.HttpRequest) -> func.HttpResponse:
    """Monthly + yearly SO predictive: realized actuals (elapsed months) + weighted
    SO Plan (current/future). Query: asof (YYYY-MM-DD). Reads persisted win-probs,
    so it needs no ML deps in the runtime."""
    from datetime import date

    from predictive.so_forecast import run_so_forecast
    from predictive.store import default_store

    try:
        asof = req.params.get("asof") or date.today().isoformat()
        out = run_so_forecast(default_store(), asof=asof, model_id=req.params.get("model"))
        return func.HttpResponse(
            json.dumps(out, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/so-forecast failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/so-source", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_so_source(req: func.HttpRequest) -> func.HttpResponse:
    """Monthly SO Plan (P) RE-BASED from an alternative source for comparison vs the
    deal-header view. Query: source=incomeplan|soplan, asof (YYYY-MM-DD). IncomePlan =
    income-line grain (spread across delivery months); SOPlan = pre-commitment sheet.
    Expected = Σ amount × persisted win-prob (joined by opp)."""
    from datetime import date

    from predictive.so_forecast import run_source_forecast
    from predictive.store import default_store

    try:
        source = (req.params.get("source") or "incomeplan").lower()
        asof = req.params.get("asof") or date.today().isoformat()
        out = run_source_forecast(default_store(), source=source, asof=asof,
                                  model_id=req.params.get("model"))
        return func.HttpResponse(
            json.dumps(out, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/so-source failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/so-conversion-forecast", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_so_conversion_forecast(req: func.HttpRequest) -> func.HttpResponse:
    """F13 — the NEW 2-stage SO forecast. Stage A conversion probs (model_id
    CRM_PDT_AZ, persisted by `predictive/run?model=CRM_PDT_AZ`) × Stage B IncomePlan
    realization. Query: asof (YYYY-MM-DD). No ML deps — serving-runtime safe."""
    from datetime import date

    from predictive.so_realization import run_so_conversion_forecast
    from predictive.store import default_store

    try:
        asof = req.params.get("asof") or date.today().isoformat()
        out = run_so_conversion_forecast(default_store(), asof=asof)
        return func.HttpResponse(
            json.dumps(out, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/so-conversion-forecast failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/forecast", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_forecast(req: func.HttpRequest) -> func.HttpResponse:
    """F9 — revenue forecast (next N months + 80% band). Query: horizon, asof."""
    from datetime import date

    from predictive.forecast import forecast_revenue
    from predictive.ingest import fetch_invoices

    try:
        horizon = int(req.params.get("horizon", 3))
        asof = req.params.get("asof") or date.today().isoformat()
        out = forecast_revenue(fetch_invoices(), asof=asof, horizon=horizon)
        return func.HttpResponse(
            json.dumps(out, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/forecast failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )


@app.route(route="predictive/attainment", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def http_predictive_attainment(req: func.HttpRequest) -> func.HttpResponse:
    """F10/F11 — target attainment + end-of-month projection per salesperson. Query: asof."""
    from datetime import date

    from predictive.attainment import compute_attainment
    from predictive.ingest import fetch_sales_orders, fetch_salesperson_names, fetch_targets

    try:
        asof = req.params.get("asof") or date.today().isoformat()
        # actuals = Sales Orders (bookings) — IDs align 15/15 with targets, not invoices.
        out = compute_attainment(
            fetch_targets(), fetch_sales_orders(), asof=asof,
            names=fetch_salesperson_names(),
        )
        return func.HttpResponse(
            json.dumps(out, ensure_ascii=False, default=str),
            status_code=200, mimetype="application/json",
        )
    except Exception as exc:
        logger.exception("predictive/attainment failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json"
        )
