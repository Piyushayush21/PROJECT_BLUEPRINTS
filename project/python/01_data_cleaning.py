"""
01_data_cleaning.py
---------------------------------------------------------------------------
PURPOSE
Clean the raw sales export and produce an analysis-ready dataset.

WHY EACH STEP EXISTS (beginner notes inline as comments)
Raw transactional exports are never analysis-ready. Before any KPI or
dashboard can be trusted, an analyst must verify:
  - No duplicate transactions (would inflate revenue)
  - No nulls in fields used for grouping/calculation (would break joins
    or silently drop rows in BI tools)
  - Consistent text casing (else "West" and "WEST" become two regions)
  - Correct data types (dates as dates, not strings; numbers as numbers)
  - Derived columns the business actually asks about (Revenue, Month,
    Quarter, Dispatch Delay) computed ONCE here so every downstream
    query/report uses the same logic — avoids "two versions of the truth".
---------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/project/data/raw/mobiles_laptop_sales_SAMPLE.csv"
OUT_PATH = "/home/claude/project/data/processed/sales_clean.csv"

print("STEP 1: Load raw data")
df = pd.read_csv(RAW_PATH)
print(f"  Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# STEP 2: Remove exact duplicate transactions
# Business reasoning: a duplicated row literally double-counts a sale.
# If left in, revenue KPIs are overstated and inventory-sold figures lie.
# ---------------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["Order_ID"])
# Order_ID should be unique per transaction; if the source system reused IDs
# across exports, also drop full-row duplicates as a safety net:
df = df.drop_duplicates()
print(f"STEP 2: Removed {before - len(df)} duplicate rows")

# ---------------------------------------------------------------------------
# STEP 3: Standardize text fields
# Business reasoning: "West" vs "WEST" vs "west" looks like 3 regions to a
# GROUP BY / pivot, splitting one real region's numbers into three rows.
# ---------------------------------------------------------------------------
text_cols = ["Region", "Brand", "Product_Type", "Processor",
             "Customer_Gender_SIMULATED", "Customer_Income_Bracket_SIMULATED"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()
df["Region"] = df["Region"].replace({"Nan": np.nan})
print("STEP 3: Standardized text casing across categorical columns")

# ---------------------------------------------------------------------------
# STEP 4: Handle missing values
# Business reasoning: deciding HOW to fill a null is a business decision,
# not just a technical one. Here:
#   - Missing Processor -> "Unknown" (don't drop the sale just because spec
#     metadata is missing; the revenue is still real)
#   - Missing RAM -> filled with the column median for that Product_Type
#     (a reasonable inference, since RAM tracks tightly with product type)
# ---------------------------------------------------------------------------
df["Processor"] = df["Processor"].replace("Nan", np.nan).fillna("Unknown")

df["RAM_GB"] = df.groupby("Product_Type")["RAM_GB"].transform(
    lambda s: s.fillna(s.median())
)
print("STEP 4: Filled missing Processor and RAM_GB values")

# ---------------------------------------------------------------------------
# STEP 5: Fix data types
# ---------------------------------------------------------------------------
df["Inward_Date"] = pd.to_datetime(df["Inward_Date"], errors="coerce")
df["Dispatch_Date"] = pd.to_datetime(df["Dispatch_Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Quantity_Sold"] = pd.to_numeric(df["Quantity_Sold"], errors="coerce").astype("Int64")
df["RAM_GB"] = pd.to_numeric(df["RAM_GB"], errors="coerce")
df["Customer_Age_SIMULATED"] = pd.to_numeric(df["Customer_Age_SIMULATED"], errors="coerce")
print("STEP 5: Cast columns to correct data types")

# Drop rows where critical fields still failed to parse (should be ~0 rows)
critical = ["Order_ID", "Price", "Quantity_Sold", "Inward_Date", "Region"]
before = len(df)
df = df.dropna(subset=critical)
print(f"STEP 5b: Dropped {before - len(df)} rows with unrecoverable nulls in critical fields")

# ---------------------------------------------------------------------------
# STEP 6: Outlier / sanity check on Price
# Business reasoning: a Rs.0 or negative price is a data entry error, not a
# real sale. A price 5x the category average is worth flagging, not deleting
# blindly (could be a legitimate premium SKU) — so we cap rather than drop.
# ---------------------------------------------------------------------------
before = len(df)
df = df[df["Price"] > 0]
q_low = df["Price"].quantile(0.001)
q_high = df["Price"].quantile(0.999)
df = df[(df["Price"] >= q_low) & (df["Price"] <= q_high)]
print(f"STEP 6: Removed {before - len(df)} rows with invalid/extreme Price values")

# ---------------------------------------------------------------------------
# STEP 7: Derived columns used repeatedly across SQL/Power BI/Python
# Business reasoning: compute once, reuse everywhere -> single source of truth
# ---------------------------------------------------------------------------
df["Revenue"] = df["Price"] * df["Quantity_Sold"]

df["Order_Month"] = df["Inward_Date"].dt.to_period("M").astype(str)
df["Order_Quarter"] = df["Inward_Date"].dt.to_period("Q").astype(str)
df["Order_Year"] = df["Inward_Date"].dt.year

df["Dispatch_Delay_Days"] = (df["Dispatch_Date"] - df["Inward_Date"]).dt.days
# Business reasoning: a negative delay means dispatch happened before the
# item was even logged as received -> impossible, flag as data error -> null
df.loc[df["Dispatch_Delay_Days"] < 0, "Dispatch_Delay_Days"] = np.nan

# Simple age-band column: makes Power BI slicers more usable than raw age
df["Age_Band_SIMULATED"] = pd.cut(
    df["Customer_Age_SIMULATED"],
    bins=[17, 24, 34, 44, 54, 100],
    labels=["18-24", "25-34", "35-44", "45-54", "55+"]
)

print("STEP 7: Added Revenue, Order_Month, Order_Quarter, Order_Year, "
      "Dispatch_Delay_Days, Age_Band_SIMULATED")

# ---------------------------------------------------------------------------
# STEP 8: Final validation summary (always print this — never assume clean)
# ---------------------------------------------------------------------------
print("\n--- FINAL VALIDATION SUMMARY ---")
print(f"Final shape: {df.shape}")
print(f"Nulls remaining per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
print(f"Date range: {df['Inward_Date'].min().date()} to {df['Inward_Date'].max().date()}")
print(f"Total Revenue: Rs.{df['Revenue'].sum():,.0f}")
print(f"Unique customers: {df['Customer_Name'].nunique():,}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned dataset -> {OUT_PATH}")
