"""score_entry.py — Azure ML batch entry for the predictive scoring pilot (F11).

Trains + scores all three models and upserts to the prod Postgres score store:
  CRM_PDT_BASE / CRM_PDT_MIX (run_scoring) and CRM_PDT_AZ (run_so_scoring).

Runs in an ML-enabled compute (sklearn/shap present), UNLIKE the lean serving
function runtime. Secrets (Postgres conn str + PBI Service-Principal creds) are read
from Key Vault via the compute's managed identity, with env-var fallback for local
runs. This is the scheduled replacement for the manual local re-score.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("score_entry")

# Key Vault secret names (override via env). The compute's MSI needs GET on these.
KV_URI = os.getenv("KEYVAULT_URI", "https://kv-crm-app.vault.azure.net/")
SECRET_MAP = {
    "POSTGRES_CONN_STR": os.getenv("KV_SECRET_PG", "POSTGRES-CONN-STR"),
    "PBI_CLIENT_ID": os.getenv("KV_SECRET_PBI_CLIENT_ID", "PBI-CLIENT-ID"),
    "PBI_CLIENT_SECRET": os.getenv("KV_SECRET_PBI_CLIENT_SECRET", "PBI-CLIENT-SECRET"),
    "PBI_TENANT_ID": os.getenv("KV_SECRET_PBI_TENANT_ID", "PBI-TENANT-ID"),
}


def _load_secrets_into_env() -> None:
    """Populate the env vars the pipeline/pbi_client read. Prefer Key Vault (MSI);
    fall back to whatever is already in the environment (local dev)."""
    missing = [k for k in SECRET_MAP if not os.getenv(k)]
    if not missing:
        log.info("all secrets present in env — skipping Key Vault")
        return
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        client = SecretClient(vault_url=KV_URI, credential=DefaultAzureCredential())
        for env_name, secret_name in SECRET_MAP.items():
            if os.getenv(env_name):
                continue
            try:
                os.environ[env_name] = client.get_secret(secret_name).value
                log.info("loaded %s from Key Vault", env_name)
            except Exception as exc:  # a missing optional secret (e.g. no SP yet) is non-fatal
                log.warning("Key Vault secret %s unavailable: %s", secret_name, exc)
    except Exception as exc:
        log.warning("Key Vault load skipped (%s) — relying on env vars", exc)


def main() -> None:
    asof = os.getenv("SCORE_ASOF") or date.today().isoformat()
    _load_secrets_into_env()
    # SP auth path: pbi_client uses PBI_CLIENT_ID + PBI_CLIENT_SECRET (+ tenant).
    os.environ.setdefault("PBI_CLIENT_SECRET", os.getenv("PBI_CLIENT_SECRET", ""))

    from predictive import schema as S
    from predictive.pipeline import run_scoring, run_so_scoring
    from predictive.store import default_store

    store = default_store()
    started = datetime.now(timezone.utc).isoformat()
    log.info("=== predictive scoring run asof=%s started=%s ===", asof, started)

    for mid in (S.MODEL_BASE, S.MODEL_MIX):
        r = run_scoring(store, asof=asof, model_id=mid)
        log.info("%s: scored=%s bands=%s auc_cv=%s",
                 mid, r["scored"], r["bands"], r["metrics"].get("auc_cv"))

    r = run_so_scoring(store, asof=asof)
    log.info("%s: scored=%s bands=%s converted_train=%s",
             S.MODEL_AZ, r["scored"], r["bands"], r["n_converted_train"])

    log.info("=== scoring run complete ===")


if __name__ == "__main__":
    main()
