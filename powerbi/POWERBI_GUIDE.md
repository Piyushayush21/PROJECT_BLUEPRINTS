# Power BI Dashboard — Build Guide

This guide turns the exported CSVs in `powerbi/exports/` into a 3-page executive
dashboard. Total build time: ~45-60 minutes following these steps exactly.

## 1. Data Model (Star Schema)

Import these 5 CSVs into Power BI Desktop (`Get Data → Text/CSV`):

| File | Role | Grain |
|---|---|---|
| `fact_company_month_risk.csv` | **Fact table** | One row per company per observed month |
| `dim_company.csv` | Dimension | One row per company (static attributes + final outcome) |
| `dim_month.csv` | Dimension | One row per relative month index (0-41) |
| `agg_sector_risk_summary.csv` | Pre-aggregate | One row per sector × risk tier |
| `shap_global_importance.csv` | Reference | One row per model feature |

### Relationships (Model view)
Build these exactly — direction matters for the slicers to filter correctly:

```
dim_company (1) ──────< (*) fact_company_month_risk     on company_id
dim_month   (1) ──────< (*) fact_company_month_risk     on month_index
```

Both relationships: **Single direction**, cardinality **One-to-Many**, cross-filter
direction **Single** (dim → fact). Do NOT mark `agg_sector_risk_summary` as related
to the fact table — it's a pre-computed standalone table used only on the
Executive Summary page for fast-loading KPI cards.

## 2. DAX Measures

Create a new Measure table called `_Measures` (Modeling → New Table → paste
`_Measures = ROW("placeholder", 0)` then delete the placeholder column visually,
or just add measures directly to `fact_company_month_risk`). Add these:

```dax
Total Companies = DISTINCTCOUNT(fact_company_month_risk[company_id])

High Risk Companies =
CALCULATE(
    DISTINCTCOUNT(fact_company_month_risk[company_id]),
    fact_company_month_risk[risk_tier] = "High"
)

High Risk Rate =
DIVIDE([High Risk Companies], [Total Companies], 0)

Avg Risk Score = AVERAGE(fact_company_month_risk[risk_score])

Avg Runway (Months) = AVERAGE(fact_company_month_risk[runway_months])

Model Recall Proxy =
-- Of company-months the model flagged High risk, what fraction actually
-- went on to fail within 6 months (sanity-check metric for the dashboard)
DIVIDE(
    CALCULATE(SUM(fact_company_month_risk[actual_label_fail_within_6m]),
        fact_company_month_risk[risk_tier] = "High"),
    CALCULATE(COUNTROWS(fact_company_month_risk),
        fact_company_month_risk[risk_tier] = "High"),
    0
)

Failed Companies (Historical) =
CALCULATE(
    DISTINCTCOUNT(dim_company[company_id]),
    dim_company[outcome] = "Failed"
)

Historical Failure Rate =
DIVIDE([Failed Companies (Historical)], DISTINCTCOUNT(dim_company[company_id]), 0)

Risk Trend Indicator =
VAR CurrentMonthRisk = [Avg Risk Score]
VAR PriorMonthRisk =
    CALCULATE(
        [Avg Risk Score],
        FILTER(ALL(dim_month), dim_month[month_index] = MAX(dim_month[month_index]) - 1)
    )
RETURN CurrentMonthRisk - PriorMonthRisk
```

## 3. Page 1 — Executive Summary

**Layout:** 4 KPI cards across the top, sector breakdown chart below, risk
tier donut on the right.

- **KPI Cards** (Card visual): `[Total Companies]`, `[High Risk Rate]`,
  `[Historical Failure Rate]`, `[Avg Runway (Months)]`
- **Bar chart**: Sector (axis) × `avg_risk_score` (value) from
  `agg_sector_risk_summary` — conditional formatting: red >45%, amber 15-45%,
  green <15% (matches the Streamlit app's risk tier thresholds — keep these
  consistent across both deliverables, an interviewer may ask why).
- **Donut chart**: `risk_tier` (legend) × count of `company_id` (values)
- **Slicers**: `sector`, `region`, `initial_funding_stage` (from `dim_company`)

## 4. Page 2 — Portfolio Risk Explorer

- **Table/Matrix visual**: rows = `company_id`, columns = `sector`, `risk_score`,
  `risk_tier`, `runway_months`, `churn_rate`, `ltv_cac_ratio`. Sort by `risk_score`
  descending. Add conditional formatting (data bars) on `risk_score`.
- **Scatter chart**: X = `runway_months`, Y = `churn_rate`, size = `mrr`,
  color = `risk_tier` — this single chart visually tells the whole "why" story:
  bottom-left quadrant (low runway, high churn) clusters red.
- **Slicer**: month_index range slider (from `dim_month`) to scrub through time
  if presenting a single company's trajectory live.

## 5. Page 3 — Model Transparency ("Why")

- **Horizontal bar chart**: `feature` × `mean_abs_shap` from
  `shap_global_importance.csv` — sorted descending, top 15.
- **Text box**: paste the methodology summary from `reports/model_card.md`
  (split-by-company, PR-AUC rationale, calibration note, synthetic data
  disclosure). This page is what separates a "pretty dashboard" from a
  dashboard a hiring manager trusts.
- **Card**: `[Model Recall Proxy]` — labeled "Of company-months flagged
  High Risk, % that failed within 6 months (back-test)."

## 6. Publishing

- File → Publish → Power BI Service (free account works) to get a shareable
  web link for your resume/LinkedIn.
- Alternative if you don't have Power BI Desktop installed: import the same
  5 CSVs into **Power BI Service** directly via "Get Data → Files → Local File"
  in the browser, or use Tableau Public / Looker Studio with the same star
  schema — the DAX measures above translate directly to LOD expressions
  (Tableau) or calculated fields (Looker Studio) if you'd rather build there.

## 7. Refresh Cadence

In production this isn't a one-time export. `.github/workflows/retrain.yml`
re-runs `powerbi/export_for_powerbi.py` on a schedule after each model
retrain, so the CSVs (and therefore the published dashboard, if using a
Power BI dataflow/gateway pointed at the repo) stay current. For a portfolio
project, documenting this refresh path matters more than actually wiring up
a live gateway — say so explicitly in your interview if asked.
