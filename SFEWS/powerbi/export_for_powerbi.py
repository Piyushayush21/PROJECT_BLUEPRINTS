"""
SFEWS — Power BI Export Pipeline
====================================

Generates a star-schema-friendly set of CSVs for Power BI: a fact table
(company-month risk scores) and dimension tables (company, sector, date),
plus a pre-aggregated portfolio summary for fast dashboard loading.

Why export CSVs rather than connect Power BI live to the model: Power BI
cannot call a Python model directly inside DirectQuery in a portable way for
a portfolio piece (it would require a Python/R script step wired to a local
gateway, which won't run on a recruiter's machine). Exporting scored CSVs
that Power BI ingests is the standard, deployable pattern used in real BI
teams: the ML pipeline runs on a schedule (cron / Airflow / GitHub Actions),
writes scored output, and BI tools visualize it. The repo's GitHub Actions
workflow (.github/workflows/retrain.yml) automates this refresh.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).parent.parent


def main():
    features = pd.read_csv(BASE / "data/processed/sfews_features.csv")
    master = pd.read_csv(BASE / "data/raw/startup_master.csv")
    model = joblib.load(BASE / "src/models/sfews_calibrated_model.joblib")
    with open(BASE / "src/models/feature_columns.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    shap_imp = pd.read_csv(BASE / "reports/shap_global_importance.csv")

    # ---- FACT TABLE: company-month risk scores ----
    proba = model.predict_proba(features[feature_cols])[:, 1]
    fact = features[["company_id", "month_index"]].copy()
    fact["risk_score"] = proba
    fact["risk_tier"] = pd.cut(fact["risk_score"], bins=[-0.01, 0.15, 0.45, 1.01],
                                labels=["Low", "Watch", "High"]).astype(str)
    fact["runway_months"] = features["runway_months_latest"]
    fact["mrr"] = features["mrr_latest"]
    fact["churn_rate"] = features["churn_rate_latest"]
    fact["ltv_cac_ratio"] = features["ltv_cac_ratio_latest"]
    fact["burn_multiple"] = features["burn_multiple"]
    fact["composite_risk_index"] = features["composite_risk_index"]
    fact["consecutive_low_runway_months"] = features["consecutive_low_runway_months"]
    fact["actual_label_fail_within_6m"] = features["label_fail_within_6m"]
    fact.to_csv(BASE / "powerbi/exports/fact_company_month_risk.csv", index=False)

    # ---- DIMENSION: company ----
    dim_company = master[["company_id", "sector", "region", "founded_year", "founder_count",
                           "founder_prior_exits", "founder_technical_pct",
                           "initial_funding_stage", "initial_capital", "failed", "acquired",
                           "fail_month", "months_observed"]].copy()
    dim_company["outcome"] = np.where(
        dim_company["failed"], "Failed",
        np.where(dim_company["acquired"], "Acquired", "Active")
    )
    dim_company.to_csv(BASE / "powerbi/exports/dim_company.csv", index=False)

    # ---- DIMENSION: month/date (synthetic calendar anchored to founding year) ----
    max_month = int(features["month_index"].max())
    dim_date = pd.DataFrame({"month_index": range(0, max_month + 1)})
    dim_date["relative_year"] = (dim_date["month_index"] // 12) + 1
    dim_date["relative_quarter"] = (dim_date["month_index"] // 3) + 1
    dim_date.to_csv(BASE / "powerbi/exports/dim_month.csv", index=False)

    # ---- PRE-AGGREGATED: portfolio summary by sector x risk tier ----
    summary = fact.merge(master[["company_id", "sector", "region"]], on="company_id", how="left")
    agg = summary.groupby(["sector", "risk_tier"]).agg(
        company_months=("company_id", "count"),
        avg_risk_score=("risk_score", "mean"),
        avg_runway=("runway_months", "mean"),
    ).reset_index()
    agg.to_csv(BASE / "powerbi/exports/agg_sector_risk_summary.csv", index=False)

    # ---- SHAP global importance for the "Why" page ----
    shap_imp.to_csv(BASE / "powerbi/exports/shap_global_importance.csv", index=False)

    print("Power BI exports written to powerbi/exports/:")
    for f in ["fact_company_month_risk.csv", "dim_company.csv", "dim_month.csv",
              "agg_sector_risk_summary.csv", "shap_global_importance.csv"]:
        path = BASE / "powerbi/exports" / f
        size_kb = path.stat().st_size / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
