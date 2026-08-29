"""ingest.py — F1 fetch_features (live).

Thin wrapper over transform.pbi_client.execute_queries that pulls Fact_Opportunity
into a DataFrame shaped for predictive.features.build_opp_features. Live-only;
offline tests use synthetic fixtures instead of calling this.
"""

from __future__ import annotations

import pandas as pd

# DAX projects the exact source columns the feature builder expects.
OPPORTUNITY_DAX = """
EVALUATE
SELECTCOLUMNS(
    Fact_Opportunity,
    "Opportunity ID", Fact_Opportunity[Opportunity ID],
    "Opportunity Name", Fact_Opportunity[Opportunity Name],
    "Account Name", RELATED(Dim_Account[Account Name]),
    "Account ID", Fact_Opportunity[Account ID],
    "SO Actual Date", Fact_Opportunity[SO Actual Date],
    "SO Actual Amount (P)", Fact_Opportunity[SO Actual Amount (P)],
    "Status", Fact_Opportunity[Status],
    "SO Plan Amount (P)", Fact_Opportunity[SO Plan Amount (P)],
    "Possibility", Fact_Opportunity[Possibility],
    "Progress", Fact_Opportunity[Progress],
    "Aging Days", Fact_Opportunity[Aging Days],
    "Sales Cycle Days", Fact_Opportunity[Sales Cycle Days],
    "Flag Hot", Fact_Opportunity[Flag Hot],
    "Create Date", Fact_Opportunity[Create Date],
    "Last Activity Date", Fact_Opportunity[Last Activity Date],
    "SO Plan Date", Fact_Opportunity[SO Plan Date],
    "Solution Name", Fact_Opportunity[Solution Name],
    "Prospect Category Name", Fact_Opportunity[Prospect Category Name]
)
"""


def _rows_to_frame(result: dict) -> pd.DataFrame:
    """pbi_client.execute_queries returns {"rows": [...]}. Keys arrive bracketed
    like '[Status]' — strip to bare column names."""
    df = pd.DataFrame(result["rows"])
    df.columns = [c.strip("[]") for c in df.columns]
    return df


INVOICE_DAX = """
EVALUATE
SELECTCOLUMNS(
    Fact_Invoice,
    "Sales Person ID", Fact_Invoice[Sales Person ID],
    "Created On", Fact_Invoice[Created On],
    "Grand Total", Fact_Invoice[Grand Total]
)
"""

SALESORDER_DAX = """
EVALUATE
SELECTCOLUMNS(
    Fact_SalesOrder,
    "Sales Person ID", Fact_SalesOrder[Sales Person ID],
    "Created On", Fact_SalesOrder[Created On],
    "Grand Total", Fact_SalesOrder[Grand Total]
)
"""

# SO-conversion ledger (Fact_SalesOrder) at OPPORTUNITY grain — the ground-truth
# signal that a deal produced a real Sales Order (label source for CRM_PDT_AZ).
# Selects only opp_id / amounts / created date; the dead cols (Status Code=1 for all,
# Is New Order=null for all) are deliberately excluded.
SO_CONVERSION_DAX = """
EVALUATE
SELECTCOLUMNS(
    FILTER(
        Fact_SalesOrder,
        NOT ISBLANK(Fact_SalesOrder[Opportunity ID])
        && Fact_SalesOrder[Opportunity ID] <> ""
    ),
    "opp_id", Fact_SalesOrder[Opportunity ID],
    "grand_total", Fact_SalesOrder[Grand Total],
    "invoiced", Fact_SalesOrder[Invoiced Amount],
    "created", Fact_SalesOrder[Created On]
)
"""

SALESPERSON_DAX = """
EVALUATE
SELECTCOLUMNS(
    Dim_SalesPerson,
    "Sales Person ID", Dim_SalesPerson[Sales Person ID],
    "Full Name", Dim_SalesPerson[Full Name]
)
"""

TARGET_DAX = """
EVALUATE
SELECTCOLUMNS(
    Fact_GoalMonth,
    "Sales Person ID", Fact_GoalMonth[Sales Person ID],
    "Target Month", Fact_GoalMonth[Target Month],
    "Target Amount", Fact_GoalMonth[Target Amount]
)
"""


def fetch_opportunities(dataset_id: str | None = None) -> pd.DataFrame:
    """F1 — pull the live Fact_Opportunity feature frame from the semantic model."""
    from transform.pbi_client import execute_queries  # lazy: avoids live import offline

    result = execute_queries(OPPORTUNITY_DAX, dataset_id=dataset_id)
    return _rows_to_frame(result)


# --- auxiliary feature sources (Groups 1, 3, 4) ---

ACTIVITY_DAX = """
EVALUATE
SELECTCOLUMNS(
    FILTER(Fact_Activity, Fact_Activity[Opportunity ID] <> ""),
    "Opportunity ID", Fact_Activity[Opportunity ID],
    "Activity Date", Fact_Activity[Activity Date],
    "Activity Type", Fact_Activity[Activity Type],
    "Duration (Mins)", Fact_Activity[Duration (Mins)]
)
"""

MOVEMENT_DAX = """
EVALUATE
SELECTCOLUMNS(
    Fact_OpportunityMovement,
    "Opportunity ID", Fact_OpportunityMovement[Opportunity ID],
    "Modified On", Fact_OpportunityMovement[Modified On],
    "Budget Score", Fact_OpportunityMovement[Budget Score],
    "Authority Score", Fact_OpportunityMovement[Authority Score],
    "Need Score", Fact_OpportunityMovement[Need Score],
    "Timing Score", Fact_OpportunityMovement[Timing Score],
    "Competitiveness Score", Fact_OpportunityMovement[Competitiveness Score]
)
"""

ACCOUNT_DAX = """
EVALUATE
SELECTCOLUMNS(
    Dim_Account,
    "Account ID", Dim_Account[Account ID],
    "Industry L1", Dim_Account[Industry L1],
    "Industry L2", Dim_Account[Industry L2],
    "Customer Level", Dim_Account[Customer Level],
    "Province", Dim_Account[Province],
    "Biz Sector", Dim_Account[Biz Sector],
    "Parent Account ID", Dim_Account[Parent Account ID]
)
"""

INVOICE_HISTORY_DAX = """
EVALUATE
SELECTCOLUMNS(
    FILTER(Fact_Invoice, Fact_Invoice[Opportunity ID] <> ""),
    "Opportunity ID", Fact_Invoice[Opportunity ID],
    "Created On", Fact_Invoice[Created On],
    "Grand Total", Fact_Invoice[Grand Total]
)
"""


def fetch_activities(dataset_id: str | None = None) -> pd.DataFrame:
    """Group 1 — opportunity-linked activities (Fact_Activity)."""
    from transform.pbi_client import execute_queries

    return _rows_to_frame(execute_queries(ACTIVITY_DAX, dataset_id=dataset_id))


def fetch_movements(dataset_id: str | None = None) -> pd.DataFrame:
    """Group 3 — BANT/competitiveness snapshots (Fact_OpportunityMovement)."""
    from transform.pbi_client import execute_queries

    return _rows_to_frame(execute_queries(MOVEMENT_DAX, dataset_id=dataset_id))


def fetch_accounts(dataset_id: str | None = None) -> pd.DataFrame:
    """Group 4 — account firmographics (Dim_Account)."""
    from transform.pbi_client import execute_queries

    return _rows_to_frame(execute_queries(ACCOUNT_DAX, dataset_id=dataset_id))


def fetch_invoice_history(dataset_id: str | None = None) -> pd.DataFrame:
    """Group 2 — opportunity-linked invoices for account lifetime value."""
    from transform.pbi_client import execute_queries

    return _rows_to_frame(execute_queries(INVOICE_HISTORY_DAX, dataset_id=dataset_id))


SO_ACTUAL_DAX = """
EVALUATE
SELECTCOLUMNS(
    Fact_Opportunity,
    "d", Fact_Opportunity[SO Actual Date],
    "a", Fact_Opportunity[SO Actual Amount (P)]
)
"""


def fetch_so_actual_by_month(year: int, dataset_id: str | None = None) -> dict:
    """Realized SO Actual amount per month ('YYYY-MM' → amount) for `year`."""
    from transform.pbi_client import execute_queries

    df = _rows_to_frame(execute_queries(SO_ACTUAL_DAX, dataset_id=dataset_id))
    df["d"] = pd.to_datetime(df["d"], errors="coerce")
    df["a"] = pd.to_numeric(df["a"], errors="coerce")
    df = df[df["d"].dt.year == year]
    g = df.groupby(df["d"].dt.strftime("%Y-%m"))["a"].sum()
    return {k: float(v) for k, v in g.items()}


def fetch_income_plan_so_lines(year: int, dataset_id: str | None = None) -> pd.DataFrame:
    """Income-LINE level SO Plan (P) from Fact_IncomePlan for `year`, attributed by each
    line's own SO Plan Date → columns [opp_id, opp_name, ym, amount, created]. This is the
    grain the certified CRM report uses (a deal's project price spread across delivery
    months), unlike our deal-header Fact_Opportunity view. `created` (Created On) lets the
    INC win-prob feature builder apply a strict point-in-time cutoff (leakage guard); the
    SO-forecast paths ignore it. LEAKAGE-SAFE: only SO-PLAN side columns are projected —
    SO Actual Amount/Date and Invoice fields are never selected."""
    from transform.pbi_client import execute_queries

    dax = f"""
EVALUATE
SELECTCOLUMNS(
    FILTER(
        Fact_IncomePlan,
        Fact_IncomePlan[Income Type Label] = "SO Plan"
        && Fact_IncomePlan[SO Income Code Label] = "P"
        && YEAR(Fact_IncomePlan[SO Plan Date]) = {int(year)}
    ),
    "opp_id", Fact_IncomePlan[Opportunity ID],
    "opp_name", Fact_IncomePlan[Opportunity Name],
    "ym", FORMAT(Fact_IncomePlan[SO Plan Date], "YYYY-MM"),
    "amount", Fact_IncomePlan[SO Plan Amount],
    "created", Fact_IncomePlan[Created On]
)
"""
    df = _rows_to_frame(execute_queries(dax, dataset_id=dataset_id))
    if not df.empty:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def fetch_incomeplan_realization(dataset_id: str | None = None) -> pd.DataFrame:
    """F2/F8 feed — income-LINE plan-vs-actual (SO Plan, code P) for the realization
    curve → columns [opp_id, plan_ym, plan_amount, actual_ym, actual_amount]. Unlike
    fetch_income_plan_so_lines (leakage-safe, plan side only for per-deal features),
    this pulls BOTH sides because the realization CURVE is an aggregate statistic, not
    a per-deal feature. actual_ym is blank for lines not yet realized."""
    from transform.pbi_client import execute_queries

    dax = """
EVALUATE
SELECTCOLUMNS(
    FILTER(
        Fact_IncomePlan,
        Fact_IncomePlan[Income Type Label] = "SO Plan"
        && Fact_IncomePlan[SO Income Code Label] = "P"
        && NOT ISBLANK(Fact_IncomePlan[SO Plan Date])
    ),
    "opp_id", Fact_IncomePlan[Opportunity ID],
    "plan_ym", FORMAT(Fact_IncomePlan[SO Plan Date], "YYYY-MM"),
    "plan_amount", Fact_IncomePlan[SO Plan Amount],
    "actual_ym", FORMAT(Fact_IncomePlan[SO Actual Date], "YYYY-MM"),
    "actual_amount", Fact_IncomePlan[SO Actual Amount]
)
"""
    df = _rows_to_frame(execute_queries(dax, dataset_id=dataset_id))
    if df.empty:
        return df
    for c in ("plan_amount", "actual_amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def fetch_soplan_so_amounts(dataset_id: str | None = None) -> pd.DataFrame:
    """Pre-commitment-sheet SO Plan (P) from Fact_SOPlan (Income Code 1 = P), summed per
    opportunity → columns [opp_id, amount]. Fact_SOPlan has no date, so the caller
    attributes each opp's amount by that opp's SO Plan Date (from the score store)."""
    from transform.pbi_client import execute_queries

    dax = """
EVALUATE
SELECTCOLUMNS(
    FILTER(Fact_SOPlan, Fact_SOPlan[Income Code] = 1),
    "opp_id", Fact_SOPlan[Opportunity ID],
    "amount", Fact_SOPlan[SO Plan Amount]
)
"""
    df = _rows_to_frame(execute_queries(dax, dataset_id=dataset_id))
    if df.empty:
        return df
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df.groupby("opp_id", as_index=False)["amount"].sum()


def fetch_invoices(dataset_id: str | None = None) -> pd.DataFrame:
    """Pull invoice rows (sales, date, amount) — actuals for forecast + attainment."""
    from transform.pbi_client import execute_queries

    df = _rows_to_frame(execute_queries(INVOICE_DAX, dataset_id=dataset_id))
    return df.rename(
        columns={
            "Sales Person ID": "sales_id",
            "Created On": "date",
            "Grand Total": "amount",
        }
    )


def fetch_sales_orders(dataset_id: str | None = None) -> pd.DataFrame:
    """Pull sales-order rows (sales, date, amount). Sales-order salesperson IDs
    align 15/15 with Fact_GoalMonth targets (verified) — the correct actuals for
    attainment, unlike invoices (2/15 overlap)."""
    from transform.pbi_client import execute_queries

    df = _rows_to_frame(execute_queries(SALESORDER_DAX, dataset_id=dataset_id))
    return df.rename(
        columns={
            "Sales Person ID": "sales_id",
            "Created On": "date",
            "Grand Total": "amount",
        }
    )


def aggregate_so_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Pure: collapse the per-order SO ledger to OPPORTUNITY grain.
    Returns [opp_id, so_count, so_total, so_invoiced, so_first_date]. Presence of an
    opp_id here = the deal converted to a real Sales Order (the F3 label signal)."""
    cols = ["opp_id", "so_count", "so_total", "so_invoiced", "so_first_date"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    df["opp_id"] = df["opp_id"].astype(str).str.strip()
    df["grand_total"] = pd.to_numeric(df["grand_total"], errors="coerce").fillna(0.0)
    df["invoiced"] = pd.to_numeric(df.get("invoiced"), errors="coerce").fillna(0.0)
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    return df.groupby("opp_id", as_index=False).agg(
        so_count=("grand_total", "size"),
        so_total=("grand_total", "sum"),
        so_invoiced=("invoiced", "sum"),
        so_first_date=("created", "min"),
    )


def fetch_so_conversions(dataset_id: str | None = None) -> pd.DataFrame:
    """F1 — live pull of the SO-conversion ledger (Fact_SalesOrder) at opportunity
    grain → [opp_id, so_count, so_total, so_invoiced, so_first_date]."""
    from transform.pbi_client import execute_queries

    df = _rows_to_frame(execute_queries(SO_CONVERSION_DAX, dataset_id=dataset_id))
    return aggregate_so_conversions(df)


def fetch_salesperson_names(dataset_id: str | None = None) -> dict[str, str]:
    """Map Sales Person ID → Full Name from Dim_SalesPerson (for display)."""
    from transform.pbi_client import execute_queries

    df = _rows_to_frame(execute_queries(SALESPERSON_DAX, dataset_id=dataset_id))
    return dict(zip(df["Sales Person ID"].astype(str), df["Full Name"].astype(str)))


def fetch_targets(dataset_id: str | None = None) -> pd.DataFrame:
    """Pull monthly sales targets (quota) per salesperson from Fact_GoalMonth."""
    from transform.pbi_client import execute_queries

    df = _rows_to_frame(execute_queries(TARGET_DAX, dataset_id=dataset_id))
    return df.rename(
        columns={
            "Sales Person ID": "sales_id",
            "Target Month": "month",
            "Target Amount": "target",
        }
    )
