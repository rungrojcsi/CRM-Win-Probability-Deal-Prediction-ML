"""store.py — F12 upsert_scores, F13 get_latest/get_deal, F24a model-run log.

Two interchangeable implementations of the same interface:
  - PostgresScoreStore — production (psycopg2 + POSTGRES_CONN_STR)
  - InMemoryScoreStore — offline tests / local dev (no DB)

Score row shape (the API contract):
  {opp_id, win_prob, band, drivers(list), amount, status, scored_at, model_run_id}
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import schema as S

POSTGRES_CONN_STR = os.getenv("POSTGRES_CONN_STR", "")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS winprob_model_runs (
    run_id      TEXT PRIMARY KEY,
    model_type  TEXT NOT NULL,
    metrics     JSONB NOT NULL,
    trained_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS winprob_scores (
    opp_id       TEXT PRIMARY KEY,
    opp_name     TEXT,
    account_name TEXT,
    win_prob     DOUBLE PRECISION NOT NULL,
    band         TEXT NOT NULL,
    drivers      JSONB NOT NULL,
    amount       DOUBLE PRECISION,
    status       TEXT,
    scored_at    TIMESTAMPTZ NOT NULL,
    model_run_id TEXT REFERENCES winprob_model_runs(run_id)
);
ALTER TABLE winprob_scores ADD COLUMN IF NOT EXISTS opp_name TEXT;
ALTER TABLE winprob_scores ADD COLUMN IF NOT EXISTS account_name TEXT;
ALTER TABLE winprob_scores ADD COLUMN IF NOT EXISTS so_plan_date DATE;
ALTER TABLE winprob_scores ADD COLUMN IF NOT EXISTS at_risk BOOLEAN DEFAULT false;
-- model_id: tag each score by which named model produced it so both
-- CRM_PDT_BASE and CRM_PDT_MIX scores coexist for the same opp.
ALTER TABLE winprob_scores ADD COLUMN IF NOT EXISTS model_id TEXT NOT NULL DEFAULT 'CRM_PDT_BASE';
-- migrate the PK from (opp_id) to (opp_id, model_id) so the upsert conflict
-- target separates the two models. Drop the old single-column PK if present.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'winprob_scores_pkey'
          AND (SELECT count(*) FROM unnest(conkey)) = 1
    ) THEN
        ALTER TABLE winprob_scores DROP CONSTRAINT winprob_scores_pkey;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'winprob_scores_pkey'
    ) THEN
        ALTER TABLE winprob_scores ADD PRIMARY KEY (opp_id, model_id);
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS idx_winprob_scores_prob ON winprob_scores (win_prob DESC);
CREATE INDEX IF NOT EXISTS idx_winprob_scores_so_plan ON winprob_scores (so_plan_date);
CREATE INDEX IF NOT EXISTS idx_winprob_scores_model ON winprob_scores (model_id);
"""


def _matches(row: dict, status, band, min_prob, so_plan_month=None, at_risk=None,
             model_id=None) -> bool:
    if model_id is not None and row.get("model_id", S.DEFAULT_MODEL_ID) != model_id:
        return False
    if status is not None and row.get("status") != status:
        return False
    if band is not None and row.get("band") != band:
        return False
    if min_prob is not None and row.get("win_prob", 0) < min_prob:
        return False
    if at_risk is not None and bool(row.get("at_risk", False)) != at_risk:
        return False
    if so_plan_month is not None:
        spd = row.get("so_plan_date")
        if spd is None:
            return False
        spd = spd.isoformat() if hasattr(spd, "isoformat") else str(spd)
        if spd[:7] != so_plan_month:
            return False
    return True


class InMemoryScoreStore:
    """Dict-backed store mirroring the SQL semantics. For tests / local."""

    def __init__(self) -> None:
        self.scores: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}

    def ensure_schema(self) -> None:  # no-op
        pass

    def insert_model_run(self, run_id: str, model_type: str, metrics: dict) -> str:
        self.runs[run_id] = {"run_id": run_id, "model_type": model_type, "metrics": metrics}
        return run_id

    def upsert_scores(self, rows: list[dict]) -> int:
        for r in rows:
            row = dict(r)
            mid = S.normalize_model_id(row.get("model_id"))
            row["model_id"] = mid
            self.scores[(row["opp_id"], mid)] = row
        return len(rows)

    def get_latest_scores(
        self, status=None, band=None, min_prob=None, limit=50, so_plan_month=None,
        order_by="win_prob", at_risk=None, model_id=S.DEFAULT_MODEL_ID,
    ) -> list[dict]:
        rows = [
            r for r in self.scores.values()
            if _matches(r, status, band, min_prob, so_plan_month, at_risk, model_id)
        ]
        key = order_by if order_by in ("win_prob", "amount") else "win_prob"
        rows.sort(key=lambda r: (r.get(key) or 0), reverse=True)
        return rows[:limit]

    def get_deal_score(self, opp_id: str, model_id=S.DEFAULT_MODEL_ID) -> dict | None:
        return self.scores.get((opp_id, S.normalize_model_id(model_id)))

    def summarize(self, status=None, band=None, min_prob=None, so_plan_month=None,
                  model_id=S.DEFAULT_MODEL_ID) -> dict:
        matched = [
            r for r in self.scores.values()
            if _matches(r, status, band, min_prob, so_plan_month, model_id=model_id)
        ]
        at_risk_rows = [r for r in matched if r.get("at_risk")]
        at_risk = len(at_risk_rows)
        # Clean pipeline (count/weighted/raw/bands) EXCLUDES past-due at_risk deals so
        # the Current-Month KPIs reconcile with the Annual SO-forecast (which also
        # drops at_risk); at_risk is reported separately above.
        rows = [r for r in matched if not r.get("at_risk")]
        bands = {"High": 0, "Mid": 0, "Low": 0}
        for r in rows:
            if r.get("band") in bands:
                bands[r["band"]] += 1
        return {
            "count": len(rows),
            "weighted": sum((r.get("amount") or 0) * r.get("win_prob", 0) for r in rows),
            "raw": sum((r.get("amount") or 0) for r in rows),
            "bands": bands,
            "at_risk": at_risk,
            "at_risk_raw": sum((r.get("amount") or 0) for r in at_risk_rows),
            "at_risk_weighted": sum((r.get("amount") or 0) * r.get("win_prob", 0) for r in at_risk_rows),
        }


class PostgresScoreStore:
    """Production store. Lazily imports psycopg2 so tests need no DB driver."""

    def __init__(self, conn_str: str | None = None) -> None:
        self.conn_str = conn_str or POSTGRES_CONN_STR
        if not self.conn_str:
            raise ValueError("POSTGRES_CONN_STR not configured")

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self.conn_str)

    def ensure_schema(self) -> None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
        finally:
            conn.close()

    def insert_model_run(self, run_id: str, model_type: str, metrics: dict) -> str:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO winprob_model_runs (run_id, model_type, metrics) "
                    "VALUES (%s, %s, %s::jsonb) ON CONFLICT (run_id) DO NOTHING",
                    (run_id, model_type, json.dumps(metrics)),
                )
        finally:
            conn.close()
        return run_id

    def upsert_scores(self, rows: list[dict]) -> int:
        from psycopg2.extras import execute_values

        if not rows:
            return 0
        tuples = [
            (
                r["opp_id"], r.get("opp_name"), r.get("account_name"),
                r["win_prob"], r["band"], json.dumps(r["drivers"]),
                r.get("amount"), r.get("status"), r["scored_at"], r.get("model_run_id"),
                r.get("so_plan_date"), bool(r.get("at_risk", False)),
                S.normalize_model_id(r.get("model_id")),
            )
            for r in rows
        ]
        sql = (
            "INSERT INTO winprob_scores "
            "(opp_id, opp_name, account_name, win_prob, band, drivers, amount, status, "
            "scored_at, model_run_id, so_plan_date, at_risk, model_id) "
            "VALUES %s ON CONFLICT (opp_id, model_id) DO UPDATE SET "
            "opp_name=EXCLUDED.opp_name, account_name=EXCLUDED.account_name, "
            "win_prob=EXCLUDED.win_prob, band=EXCLUDED.band, drivers=EXCLUDED.drivers, "
            "amount=EXCLUDED.amount, status=EXCLUDED.status, scored_at=EXCLUDED.scored_at, "
            "model_run_id=EXCLUDED.model_run_id, so_plan_date=EXCLUDED.so_plan_date, "
            "at_risk=EXCLUDED.at_risk"
        )
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                execute_values(
                    cur, sql, tuples,
                    template="(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s)", page_size=200,
                )
        finally:
            conn.close()
        # execute_values pages internally, so cur.rowcount only reflects the last
        # page. Every tuple is upserted (INSERT or DO UPDATE), so the row count is
        # exactly len(tuples).
        return len(tuples)

    def get_latest_scores(
        self, status=None, band=None, min_prob=None, limit=50, so_plan_month=None,
        order_by="win_prob", at_risk=None, model_id=S.DEFAULT_MODEL_ID,
    ):
        where, params = self._filters(status, band, min_prob, so_plan_month, at_risk, model_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        col = order_by if order_by in ("win_prob", "amount") else "win_prob"  # whitelist
        params.append(limit)
        return self._query(
            f"SELECT * FROM winprob_scores {clause} ORDER BY {col} DESC NULLS LAST LIMIT %s",
            params,
        )

    @staticmethod
    def _filters(status, band, min_prob, so_plan_month, at_risk=None,
                 model_id=S.DEFAULT_MODEL_ID):
        where, params = [], []
        if model_id is not None:
            where.append("model_id = %s"); params.append(S.normalize_model_id(model_id))
        if status is not None:
            where.append("status = %s"); params.append(status)
        if band is not None:
            where.append("band = %s"); params.append(band)
        if min_prob is not None:
            where.append("win_prob >= %s"); params.append(min_prob)
        if at_risk is not None:
            where.append("COALESCE(at_risk, false) = %s"); params.append(at_risk)
        if so_plan_month is not None:
            where.append("to_char(so_plan_date, 'YYYY-MM') = %s"); params.append(so_plan_month)
        return where, params

    def get_deal_score(self, opp_id: str, model_id=S.DEFAULT_MODEL_ID) -> dict | None:
        rows = self._query(
            "SELECT * FROM winprob_scores WHERE opp_id = %s AND model_id = %s",
            [opp_id, S.normalize_model_id(model_id)],
        )
        return rows[0] if rows else None

    def summarize(self, status=None, band=None, min_prob=None, so_plan_month=None,
                  model_id=S.DEFAULT_MODEL_ID) -> dict:
        where, params = self._filters(status, band, min_prob, so_plan_month,
                                      model_id=model_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        # Clean pipeline aggregates EXCLUDE past-due at_risk deals (FILTER) so the
        # Current-Month KPIs reconcile with the Annual SO-forecast; at_risk counts
        # the excluded past-due deals across the same filtered set.
        ok = "FILTER (WHERE NOT COALESCE(at_risk, false))"
        sql = (
            f"SELECT count(*) {ok} AS count, "
            f"COALESCE(sum(amount * win_prob) {ok}, 0) AS weighted, "
            f"COALESCE(sum(amount) {ok}, 0) AS raw, "
            f"COALESCE(sum((band = 'High')::int) {ok}, 0) AS high, "
            f"COALESCE(sum((band = 'Mid')::int) {ok}, 0) AS mid, "
            f"COALESCE(sum((band = 'Low')::int) {ok}, 0) AS low, "
            "COALESCE(sum(at_risk::int), 0) AS at_risk, "
            "COALESCE(sum(amount) FILTER (WHERE at_risk), 0) AS at_risk_raw, "
            "COALESCE(sum(amount * win_prob) FILTER (WHERE at_risk), 0) AS at_risk_weighted "
            f"FROM winprob_scores {clause}"
        )
        r = self._query(sql, params)[0]
        return {
            "count": int(r["count"]),
            "weighted": float(r["weighted"]),
            "raw": float(r["raw"]),
            "bands": {"High": int(r["high"]), "Mid": int(r["mid"]), "Low": int(r["low"])},
            "at_risk": int(r["at_risk"]),
            "at_risk_raw": float(r["at_risk_raw"]),
            "at_risk_weighted": float(r["at_risk_weighted"]),
        }

    def _query(self, sql: str, params: list) -> list[dict]:
        import psycopg2.extras

        conn = self._connect()
        try:
            with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


def default_store() -> Any:
    """Pick Postgres if configured, else in-memory (local/dev)."""
    return PostgresScoreStore() if POSTGRES_CONN_STR else InMemoryScoreStore()
