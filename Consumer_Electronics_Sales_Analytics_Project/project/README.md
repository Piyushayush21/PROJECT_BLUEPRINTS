# Consumer Electronics Sales Analytics & Customer Segmentation

A portfolio-level data analytics project covering the full pipeline:
**raw data → SQL data warehouse → Python cleaning/segmentation → Power BI dashboard**.

Built on the structure and statistical profile of the Kaggle dataset
[**Mobiles & Laptop Sales Data**](https://www.kaggle.com/datasets/vinothkannaece/mobiles-and-laptop-sales-data)
by vinothkannaece.

---

## ⚠️ Important note on data source (read this first)

This dataset has **no direct download access** from inside the build environment used to
create this project (no network route to kaggle.com). To make the project fully runnable,
`python/00_generate_sample_data.py` generates a **synthetic sample dataset** that
replicates the real dataset's **confirmed schema and statistical profile**, sourced from:

- The dataset author's own published EDA notebook
- An independent third-party EDA write-up of the same dataset

**Confirmed real columns:** `Product_Type, Brand, Price, Quantity_Sold, Region, Processor,
RAM, ROM, Customer_Name, Inward_Date, Dispatch_Date` — ~50,000+ rows, March 2023–May 2025.

**Confirmed real statistics replicated in the sample:** Laptop/Mobile split ≈50/50,
Apple leads laptops, Google leads mobiles, average price ≈₹102,500 for both categories,
West region leads sales, top processors include Intel i9/i5 and MediaTek Dimensity/Samsung
Exynos, average dispatch delay ≈30.6 days.

**Three columns are SIMULATED enrichment, not present in the real dataset:**
`Customer_Age_SIMULATED`, `Customer_Gender_SIMULATED`, `Customer_Income_Bracket_SIMULATED`.
These were added because the original dataset has no demographic fields, and this project's
brief asked for demographic target-audience analysis. They are clearly suffixed
`_SIMULATED` in every file, table, and chart so nobody downstream mistakes them for real
customer data. **If you have the real CSV, drop it at
`data/raw/mobiles_laptop_sales_real.csv` and point `01_data_cleaning.py` at it instead —
everything past that step works unchanged on real data, minus the three simulated columns.**

There is also **no Cost field** in the real dataset, so this project does **not** report
profit margin — only revenue. A "Recommended Addition" note covers this in Section 10.

---

## 1. Business Problem & Objectives

**Problem statement:** The business sells mobiles and laptops across four regions through
a single sales log, but has no structured way to know which products/regions/customers
drive revenue, or who deserves marketing/retention investment.

**Objectives:**
1. Quantify revenue performance by product, brand, processor spec, and region.
2. Segment customers by purchasing **behavior** (RFM) to prioritize retention spend.
3. Identify seasonal demand patterns to guide inventory planning.
4. Surface logistics bottlenecks (dispatch delay) by region/category.
5. Package all of the above into a self-refreshing Power BI dashboard.

---

## 2. Dataset Understanding

| Column | Type | Source | Description |
|---|---|---|---|
| Order_ID | Integer | Real | Unique transaction ID |
| Customer_Name | Text | Real | Customer identifier (name-based, no CRM ID in source) |
| Product_Type | Text | Real | Mobile or Laptop |
| Brand | Text | Real | e.g. Apple, Samsung, Dell, Google |
| Processor | Text | Real | Chipset, e.g. Intel i9, MediaTek Dimensity |
| RAM_GB / ROM_GB | Numeric | Real | Memory/storage specs |
| Price | Numeric | Real | Unit price |
| Quantity_Sold | Integer | Real | Units in that order line |
| Region | Text | Real | West / North / South / East |
| Inward_Date | Date | Real | Date item logged into inventory/order placed |
| Dispatch_Date | Date | Real | Date item shipped |
| Customer_Age/Gender/Income_SIMULATED | Various | **Simulated** | Enrichment for demographic analysis only |

**Derived columns (computed in Python, used everywhere downstream):**
`Revenue = Price × Quantity_Sold`, `Order_Month`, `Order_Quarter`, `Dispatch_Delay_Days`,
`Age_Band_SIMULATED`, `RFM Segment`.

---

## 3. Project Architecture

```
Raw CSV (Kaggle export)
      │
      ▼
[Python] 01_data_cleaning.py  →  dedup, type-fix, derive Revenue/Month/Quarter
      │
      ▼
[Python] 02_rfm_segmentation.py → Recency/Frequency/Monetary scoring → segments
      │
      ▼
[Python] 03_eda_and_kpis.py → validates every KPI number BEFORE Power BI
      │
      ▼
[SQL] 01_schema.sql → star schema (fact_sales + dim_customer/product/region/date)
[SQL] 04_build_and_test_database.py → loads cleaned data into the schema, tested
      │
      ▼
[SQL] 02_business_queries.sql → 10 core business questions, reusable in Power BI
      │
      ▼
[Power BI] .pbix → connects to the database (or processed CSVs) → 5 dashboard pages
      │
      ▼
Scheduled Refresh (Power BI Service / Gateway) → stakeholders see live numbers
```

**Why this order matters (beginner note):** clean data before segmenting, segment before
building KPIs, validate KPIs in Python before trusting them in Power BI. Skipping a step
means a DAX bug looks identical to a real insight — you won't know which one you're looking at.

---

## 4. SQL Database Design

Star schema with one fact table and four dimensions — see `sql/01_schema.sql`.

**Why a star schema and not one flat table in the database:**
- Avoids repeating brand/region/processor text thousands of times (storage + consistency)
- Lets the database enforce valid foreign keys (a region key must exist in `dim_region`)
- Is exactly the shape Power BI's data model expects for fast, correct aggregation

10 core business queries are in `sql/02_business_queries.sql`, covering monthly trends,
brand/region revenue, top customers, RFM segment value, MoM growth (`LAG()`), dispatch
delay, top processors per category (`RANK()` window function), and a simple retention
cohort check.

All 10 queries were executed end-to-end against a real SQLite database built from this
schema in `python/04_build_and_test_database.py` — confirmed working, not just written.

---

## 5. Python Cleaning & Analysis Steps

| Script | Purpose |
|---|---|
| `00_generate_sample_data.py` | Generates the sample dataset (skip if you have the real CSV) |
| `01_data_cleaning.py` | Dedup, standardize text casing, fix nulls/types, derive Revenue/Month/Quarter/Dispatch_Delay |
| `02_rfm_segmentation.py` | Computes Recency/Frequency/Monetary, assigns 6 named segments + recommended action per segment |
| `03_eda_and_kpis.py` | Computes every dashboard KPI directly in Python — the "answer key" Power BI must match |
| `04_build_and_test_database.py` | Loads cleaned data into a real star-schema database and runs/verifies the SQL queries |

Each script prints its own validation summary (row counts before/after, nulls remaining,
date range) — **never trust a cleaning step that doesn't show you what it changed.**

---

## 6. KPI Definitions

| KPI | Formula | Business Question Answered |
|---|---|---|
| Total Revenue | SUM(Price × Quantity_Sold) | How much are we selling? |
| Total Orders | COUNT(DISTINCT Order_ID) | How many transactions? |
| Average Order Value (AOV) | Total Revenue ÷ Total Orders | Is basket size growing? |
| Units Sold | SUM(Quantity_Sold) | Volume independent of price changes |
| Revenue per Customer | Total Revenue ÷ Unique Customers | How much is each customer worth on average? |
| MoM Growth % | (This month − Last month) ÷ Last month | Is growth accelerating/decelerating? |
| Regional Revenue Share % | Region Revenue ÷ Total Revenue | Where should marketing/inventory budget go? |
| Avg Dispatch Delay (days) | AVG(Dispatch_Date − Inward_Date) | Are we fulfilling orders fast enough? |
| RFM Segment Revenue Share % | Segment Revenue ÷ Total Revenue | How much revenue is concentrated in which customer tier? |
| Top Processor by Units | MAX(SUM(Quantity_Sold)) grouped by processor | What specs to prioritize in stocking? |

---

## 7. Power BI Dashboard Design

**Data model:** import the star schema tables (or `sales_clean_with_segments.csv` directly
for a simpler single-table model) and mark `dim_date` as a proper Date Table.

| Page | Key Visuals | Business Purpose |
|---|---|---|
| **Executive Summary** | KPI cards (Revenue, Orders, AOV, Customers), revenue trend line, region map | One-glance health check for leadership |
| **Sales Performance** | Monthly/quarterly trend, MoM growth %, best/worst month callouts | Spot seasonality and growth/decline early |
| **Product Analysis** | Revenue by brand/category, top processors bar chart, price distribution | Guide stocking and supplier negotiation |
| **Customer Analysis** | RFM segment donut, top 10 customers table, segment revenue share, age-band/income bar charts (flagged simulated) | Prioritize retention and marketing spend |
| **Regional Analysis** | Map visual, region revenue bar chart, dispatch delay by region | Guide regional inventory/logistics decisions |

**Filters/slicers used on every page:** Date range, Region, Product_Type, RFM Segment —
so any stakeholder can drill from "company-wide" to "West region, Mobiles, Champions only"
in a few clicks.

---

## 8. Customer Segmentation Strategy

RFM (Recency, Frequency, Monetary) segmentation — chosen because the real dataset has solid
transaction history but no CRM/demographic fields, and RFM works from transactions alone.

| Segment | Definition | Recommended Action |
|---|---|---|
| Champions | Recent, frequent, high spend | Reward, protect — don't discount |
| Loyal Customers | Frequent, moderate-recent, solid spend | Upsell premium SKUs, cross-sell accessories |
| New / Promising | Recent but low frequency | Second-purchase incentive within 30 days |
| At Risk (High Value) | High past spend, going quiet | Urgent win-back — highest revenue at stake |
| Needs Attention | Mid-tier across the board | Moderate-discount re-engagement |
| Lost / Churned | Low on all three dimensions | Low-cost reactivation only |

In the sample data: **Champions + Loyal Customers = 51.8% of revenue from 44% of
customers** — the single clearest argument for a retention-first marketing budget.

---

## 9. Target Audience Analysis

> Age, Gender, and Income figures below use the **SIMULATED** enrichment columns —
> directionally illustrative only, not derived from real customer records.

- **Age:** 25–34 is the single highest-revenue band, followed closely by 35–44 — together
  over 60% of simulated revenue. Marketing creative should center this group.
- **Income:** "High" and "Medium" brackets are roughly equal and dominate; "Premium" is a
  small but real segment worth a dedicated premium-SKU campaign.
- **Region × Product:** West leads in both categories — consistent with the real dataset's
  reported pattern, plausibly tied to urbanization/tech-adoption density.
- **Product preference:** Laptop and Mobile revenue are almost perfectly split (50/50),
  meaning marketing budget split should mirror that rather than favor one category.

---

## 10. Business Insights & Recommendations

1. **Revenue is evenly split between laptops and mobiles** — don't over-index marketing
   spend toward one category; the real growth lever is brand/processor mix, not category.
2. **West region drives ~34% of revenue** with the other three regions trailing evenly —
   investigate whether this is demand-driven (open more stock there) or under-investment
   elsewhere (an untapped opportunity in East, currently lowest at ~19%).
3. **Champions + Loyal customers drive ~52% of revenue from <45% of customers** — a
   retention program targeting just these two segments protects the majority of revenue
   at a fraction of the cost of broad-based acquisition marketing.
4. **"At Risk (High Value)" customers represent real revenue under threat** — these are
   former big spenders going quiet; a win-back campaign here has the highest ROI of any
   segment because the historical spend proves intent to buy.
5. **Dispatch delay is consistent (~30 days) across regions and categories** — operationally
   stable, but 30 days is long for electronics; even a 5-day reduction is a tangible
   customer-experience win worth investigating with the fulfillment/logistics team.
6. **Recommended addition to the data pipeline:** capture unit Cost in the source system.
   Without it, this project can only report revenue, not margin — and a high-revenue
   product can quietly be a low-margin product. This is the single highest-value next
   step for this analytics program.

---

## 11. Dashboard Automation Process

1. Land the raw export (CSV or DB extract) in `data/raw/` on a schedule (daily/weekly).
2. Run `01_data_cleaning.py` → `02_rfm_segmentation.py` → `03_eda_and_kpis.py` via a
   scheduled task (Windows Task Scheduler, cron, or Airflow for a larger setup).
3. Load the refreshed CSVs into the database using `04_build_and_test_database.py`
   (or point directly at a production MySQL/PostgreSQL instance using `01_schema.sql`).
4. In Power BI Desktop, set the data source to the database/CSV; publish to Power BI Service.
5. In Power BI Service, configure a **Scheduled Refresh** (via an on-premises gateway if the
   database is local) so the dashboard updates automatically without manual re-exports.
6. Set up data-driven alerts on key cards (e.g., "alert me if monthly revenue drops >10%").

---

## 12. Project Folder Structure

```
project/
├── data/
│   ├── raw/                          # original/sample CSV exports
│   └── processed/                    # cleaned CSVs, RFM output, SQLite DB, KPI summary
├── sql/
│   ├── 01_schema.sql                 # star schema DDL
│   └── 02_business_queries.sql       # 10 core business questions
├── python/
│   ├── 00_generate_sample_data.py
│   ├── 01_data_cleaning.py
│   ├── 02_rfm_segmentation.py
│   ├── 03_eda_and_kpis.py
│   └── 04_build_and_test_database.py
├── powerbi/
│   └── (dashboard_design_notes.md — page-by-page wireframe spec)
├── docs/
│   └── interview_questions.md
└── README.md
```

---

## Tech Stack

`Python (pandas, numpy, sqlite3)` · `SQL (star schema, window functions, CTEs)` ·
`Power BI (DAX, data modeling, scheduled refresh)`

---

## Author's Note on Methodology

This is a **portfolio/teaching project**. The data pipeline, SQL design, and segmentation
logic are production-grade techniques — but two specific limitations are flagged
throughout: (1) demographic columns are simulated, and (2) no cost/margin data exists in
the source. Both are clearly labeled wherever they appear so this project can be evaluated
honestly, and easily re-run against the real dataset if/when it becomes accessible.
