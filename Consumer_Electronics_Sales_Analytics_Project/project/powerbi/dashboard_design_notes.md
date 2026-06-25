# Power BI Dashboard Design Notes
### Consumer Electronics Sales Analytics & Customer Segmentation

This document is the wireframe spec a developer would follow to actually build the
`.pbix` file in Power BI Desktop. It is written so a beginner can follow it step by step.

---

## 0. Data Model Setup (do this before building any visual)

1. **Get Data → SQLite/SQL Server/CSV** → import:
   - `fact_sales`, `dim_customer`, `dim_product`, `dim_region`, `dim_date`
   - (or just `sales_clean_with_segments.csv` if you want a single flat table for a simpler build)
2. In **Model view**, confirm relationships are auto-detected:
   - `fact_sales[customer_key]` → `dim_customer[customer_key]` (Many-to-One)
   - `fact_sales[product_key]` → `dim_product[product_key]` (Many-to-One)
   - `fact_sales[region_key]` → `dim_region[region_key]` (Many-to-One)
   - `fact_sales[inward_date_key]` → `dim_date[date_key]` (Many-to-One)
3. Right-click `dim_date` → **Mark as Date Table** → pick `full_date`.
   **Why:** unlocks Power BI's built-in time intelligence (SAMEPERIODLASTYEAR, etc.)
   and gives clean Year/Quarter/Month hierarchies on any visual automatically.
4. Create these core DAX measures in a new table called `_Measures` (best practice:
   keep measures separate from data tables so they're easy to find):

```dax
Total Revenue = SUM(fact_sales[revenue])

Total Orders = DISTINCTCOUNT(fact_sales[order_id])

Total Units Sold = SUM(fact_sales[quantity_sold])

Average Order Value = DIVIDE([Total Revenue], [Total Orders])

Unique Customers = DISTINCTCOUNT(fact_sales[customer_key])

Avg Revenue per Customer = DIVIDE([Total Revenue], [Unique Customers])

Avg Dispatch Delay (Days) = AVERAGE(fact_sales[dispatch_delay_days])

Prior Month Revenue =
CALCULATE([Total Revenue], DATEADD(dim_date[full_date], -1, MONTH))

MoM Growth % =
DIVIDE([Total Revenue] - [Prior Month Revenue], [Prior Month Revenue])

Region Revenue Share % =
DIVIDE([Total Revenue], CALCULATE([Total Revenue], ALL(dim_region)))

Segment Revenue Share % =
DIVIDE([Total Revenue], CALCULATE([Total Revenue], ALL(dim_customer[rfm_segment])))
```

**Why measures instead of calculated columns:** measures recalculate dynamically based
on whatever filters/slicers the user clicks — a calculated column would freeze the % at
import time and break the moment someone filters to one region.

---

## Page 1 — Executive Summary

**Audience:** leadership, glance-and-go.

| Visual | Fields | Notes |
|---|---|---|
| 4 KPI Cards | Total Revenue, Total Orders, AOV, Unique Customers | Top row, large font, no chart clutter |
| Line chart | Revenue by Month (X), Total Revenue (Y) | Add a trend line; this is the single most-viewed visual on the page |
| Map visual | Region (location), Total Revenue (size/color) | Bubble size = revenue; instantly shows West dominance |
| Donut chart | Revenue by Product_Type | Confirms the near-50/50 Laptop/Mobile split at a glance |
| KPI card with conditional color | MoM Growth % | Green if positive, red if negative — conditional formatting rules under Format pane |

**Slicers (top of page, applies to whole page):** Date range, Region, Product_Type.

---

## Page 2 — Sales Performance

**Audience:** sales ops / commercial leadership.

| Visual | Fields | Notes |
|---|---|---|
| Line + column combo chart | Month (X), Revenue (line), Orders (column) | Shows whether revenue growth is from more orders or bigger orders |
| Table | Month, Revenue, Prior Month Revenue, MoM Growth % | Conditionally format MoM Growth % column (red/green) |
| Card | Best Month / Worst Month | Use a Top N filter on a hidden measure, or a DAX "Best Month" measure with TOPN |
| Quarterly bar chart | Order_Quarter (X), Revenue (Y) | Helps spot Q4 seasonality if present |

**Why combine line + column:** a single chart instantly shows whether a revenue dip is
a volume problem (fewer orders) or a pricing/mix problem (same orders, less revenue) —
two completely different fixes.

---

## Page 3 — Product Analysis

**Audience:** category/merchandising managers.

| Visual | Fields | Notes |
|---|---|---|
| Bar chart | Brand (Y), Total Revenue (X) | Sorted descending; this is the "who to negotiate with" chart |
| Bar chart | Processor (Y), Units Sold (X), split by Product_Type | Use a slicer for Product_Type so Laptop/Mobile processors don't blend |
| Scatter plot | Avg Price (X) vs Units Sold (Y), bubble = Brand | Identifies high-volume/low-price vs low-volume/high-price brands — flags margin risk areas even without cost data |
| Table | RAM/ROM combo, Units Sold, Avg Price | Helps with stocking decisions on configurations |

**Business reasoning for the scatter plot:** without a Cost column, this is the closest
proxy for spotting potential margin risk — a brand far bottom-right (low price, huge
volume) deserves a supplier-cost conversation even before formal margin data exists.

---

## Page 4 — Customer Analysis

**Audience:** CRM / retention marketing team.

| Visual | Fields | Notes |
|---|---|---|
| Donut chart | RFM Segment, Customer count | Shows segment size distribution |
| Bar chart | RFM Segment (Y), Segment Revenue Share % (X) | The key "where's the money" chart — pairs with the donut above to show size vs. value mismatch (e.g., small segment, huge revenue share) |
| Table | Top 10 customers: Name, Segment, Orders, Lifetime Revenue | Direct CRM hand-off list |
| Bar chart *(flagged simulated)* | Age_Band_SIMULATED, Revenue | Title must include "(Simulated demographic data)" |
| Bar chart *(flagged simulated)* | Income_Bracket_SIMULATED, Revenue | Same flag requirement |

**Critical formatting rule:** any visual using a `_SIMULATED` field must have "(Simulated
data — see README)" in its visual title text box. This is non-negotiable for an honest
dashboard — a viewer should never mistake fabricated demographic data for real customer
records.

**Slicer:** RFM Segment (lets CRM team click "At Risk (High Value)" and see just that
group's top customers in the table).

---

## Page 5 — Regional Analysis

**Audience:** regional ops / logistics.

| Visual | Fields | Notes |
|---|---|---|
| Filled map | Region, Total Revenue (color intensity) | Quick visual read on regional concentration |
| Bar chart | Region (Y), Revenue (X), Customers (X, secondary axis) | Shows whether high revenue = many customers or few big spenders |
| Bar chart | Region + Product_Type, Avg Dispatch Delay (Days) | Logistics bottleneck finder |
| Table | Region, Orders, Revenue, Revenue Share %, Avg Dispatch Delay | One-stop regional scorecard |

**Slicer:** Product_Type, Date range.

---

## Cross-page consistency rules

- Same color palette across all 5 pages: one fixed color per Region, one fixed color per
  Product_Type, one fixed color per RFM Segment (set this in the **Format → Data colors**
  pane once, globally, not per visual — avoids "West is blue on page 1 but green on page 5").
- Every page has the same slicer panel position (top or left) for muscle-memory navigation.
- Page titles include the page's audience implicitly via heading style (e.g., "Customer
  Analysis — Retention & CRM").

---

## Sharing the design back to the team

Because the `.pbix` binary can't be generated in this text-based environment, hand this
document plus the underlying CSVs/DB (`data/processed/`) to whoever builds the dashboard
in Power BI Desktop — every chart, field, and DAX measure needed is specified above.
