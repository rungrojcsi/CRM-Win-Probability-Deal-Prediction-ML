"""Predictive ML package for the CRM Predictive Dashboard.

Modules:
  schema   — column names, label logic, feature spec (single source of truth)
  features — F2 validate + F3 build_opp_features (pure pandas, offline-testable)
  winprob  — F6 train, F7 score, F8 explain (Win-Probability model)
  ingest   — F1 fetch_features (live PBI; thin wrapper over transform.pbi_client)
"""
