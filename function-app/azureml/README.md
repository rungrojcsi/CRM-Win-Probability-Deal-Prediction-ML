# Azure ML scoring pilot (F11–F12)

Moves the **manual local re-score** of the predictive models onto a scheduled,
managed Azure ML batch job. Pilot scope = all three models (BASE/MIX/AZ) on one
monthly job; the design target was the AZ (Sales-Order) model.

## Files (F11 — authored, no cloud cost)
| File | Role |
|---|---|
| `score_entry.py` | Entry: `run_scoring` (BASE, MIX) + `run_so_scoring` (AZ) → prod Postgres. Secrets from Key Vault via MSI, env fallback. |
| `conda_env.yml` | ML env (sklearn/shap/psycopg2 + KeyVault SDK). |
| `score_job.yml` | AML command job (ships the whole `function-app` package). |
| `score_schedule.yml` | Monthly cron (2nd 02:00 Bangkok). |

## Provisioning (gated — creates billable resources)
```bash
RG=RESOURCE_GROUP; WS=aml-crm-app; LOC=southeastasia
# 1. Workspace (reuses the existing storage + kv-crm-app where possible)
az ml workspace create -n $WS -g $RG -l $LOC
# 2. Low-priority compute, scale-to-0 (cost only while a job runs)
az ml compute create -g $RG -w $WS -n cpu-cluster --type AmlCompute \
  --size Standard_DS3_v2 --min-instances 0 --max-instances 1 --tier low_priority
# 3. Grant the workspace/compute MSI 'get' on the Key Vault secrets
az keyvault set-policy -n kv-crm-app --object-id <WS_MSI_OBJECT_ID> --secret-permissions get
# 4. Store secrets (one-time)
az keyvault secret set --vault-name kv-crm-app -n POSTGRES-CONN-STR     --value "<conn>"
az keyvault secret set --vault-name kv-crm-app -n PBI-CLIENT-ID         --value "<sp app id>"
az keyvault secret set --vault-name kv-crm-app -n PBI-CLIENT-SECRET     --value "<sp secret>"
az keyvault secret set --vault-name kv-crm-app -n PBI-TENANT-ID         --value "<tenant>"
# 5. Run once, then schedule
az ml job create      -f score_job.yml      -g $RG -w $WS
az ml schedule create -f score_schedule.yml -g $RG -w $WS
```
**Postgres firewall:** the AML compute egress IPs must be allowed on
`pg-crm-app` (add the compute subnet / a VNet rule, or use Private Link) —
the manual re-score used a temporary per-IP rule, which won't work for ephemeral
low-priority nodes.

## ⚠️ F12 — PBI Service-Principal auth (BLOCKER to confirm)
The job authenticates to Power BI via SP (`PBI_CLIENT_ID/SECRET/TENANT`,
`pbi_client._sp_client_credentials`). Setup needs, in order:
1. **AAD app registration** + client secret — *may require admin* (Boss is not
   tenant admin per project notes). Confirm rights or request from IT.
2. **PBI tenant setting** "Allow service principals to use Power BI APIs" enabled
   (Fabric/PBI admin).
3. Add the SP as a **Viewer** on the `SALES_DATA` workspace (dataset read).
Until the SP exists, the job can run with a delegated `PBI_ACCESS_TOKEN` env var
for a manual test, but the unattended schedule REQUIRES the SP.

## Cost note
Workspace is free; Standard_DS3_v2 low-priority ≈ a few ฿/hour, only while the
monthly job runs (~minutes). Storage/App Insights minimal. Well within the
$10k <subscription> credit.
