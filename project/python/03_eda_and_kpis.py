"""
03_eda_and_kpis.py
---------------------------------------------------------------------------
PURPOSE
Compute every KPI that will appear on the Power BI dashboard, directly in
Python first. WHY DO THIS BEFORE OPENING POWER BI?
A senior analyst always validates numbers in a second tool before trusting
a dashboard visual. If Power BI later shows total revenue of Rs.6.5B and
this script also says Rs.6.5B, you can trust the report. If they disagree,
you've caught a DAX bug before a stakeholder did.
---------------------------------------------------------------------------
"""

import pandas as pd

df = pd.read_csv("/home/claude/project/data/processed/sales_clean_with_segments.csv",
                  parse_dates=["Inward_Date", "Dispatch_Date"])

print("=" * 70)
print("EXECUTIVE KPIs")
print("=" * 70)
total_revenue = df["Revenue"].sum()
total_orders = df["Order_ID"].nunique()
total_units = df["Quantity_Sold"].sum()
aov = total_revenue / total_orders
unique_customers = df["Customer_Name"].nunique()
avg_revenue_per_customer = total_revenue / unique_customers

print(f"Total Revenue:            Rs.{total_revenue:,.0f}")
print(f"Total Orders:             {total_orders:,}")
print(f"Total Units Sold:         {total_units:,}")
print(f"Average Order Value:      Rs.{aov:,.0f}")
print(f"Unique Customers:         {unique_customers:,}")
print(f"Avg Revenue / Customer:   Rs.{avg_revenue_per_customer:,.0f}")
print(f"Avg Dispatch Delay (days):{df['Dispatch_Delay_Days'].mean():.1f}")

print("\n" + "=" * 70)
print("PRODUCT KPIs")
print("=" * 70)
by_type = df.groupby("Product_Type").agg(
    Revenue=("Revenue", "sum"), Units=("Quantity_Sold", "sum"),
    Avg_Price=("Price", "mean")
).sort_values("Revenue", ascending=False)
print(by_type)

print("\nTop 5 Brands by Revenue:")
print(df.groupby("Brand")["Revenue"].sum().sort_values(ascending=False).head(5))

print("\nTop 5 Processors by Units Sold:")
print(df.groupby("Processor")["Quantity_Sold"].sum().sort_values(ascending=False).head(5))

print("\n" + "=" * 70)
print("REGIONAL KPIs")
print("=" * 70)
by_region = df.groupby("Region").agg(
    Revenue=("Revenue", "sum"), Orders=("Order_ID", "nunique"),
    Customers=("Customer_Name", "nunique")
).sort_values("Revenue", ascending=False)
by_region["Revenue_Share_%"] = (by_region["Revenue"] / total_revenue * 100).round(1)
print(by_region)

print("\n" + "=" * 70)
print("TIME-BASED KPIs")
print("=" * 70)
monthly = df.groupby("Order_Month")["Revenue"].sum()
print(f"Best month: {monthly.idxmax()} (Rs.{monthly.max():,.0f})")
print(f"Worst month: {monthly.idxmin()} (Rs.{monthly.min():,.0f})")

quarterly = df.groupby("Order_Quarter")["Revenue"].sum().sort_values(ascending=False)
print("\nRevenue by Quarter:")
print(quarterly)

print("\n" + "=" * 70)
print("CUSTOMER / SEGMENT KPIs  (RFM-based)")
print("=" * 70)
seg_kpi = df.groupby("Segment").agg(
    Customers=("Customer_Name", "nunique"),
    Revenue=("Revenue", "sum"),
).sort_values("Revenue", ascending=False)
seg_kpi["Revenue_Share_%"] = (seg_kpi["Revenue"] / total_revenue * 100).round(1)
print(seg_kpi)

print("\n" + "=" * 70)
print("DEMOGRAPHIC KPIs  [SIMULATED ENRICHMENT FIELDS]")
print("=" * 70)
print("NOTE: Age/Gender/Income fields below are SIMULATED — not present in")
print("the original Kaggle dataset. See README for details.\n")

by_age = df.groupby("Age_Band_SIMULATED")["Revenue"].sum().sort_values(ascending=False)
print("Revenue by Age Band (simulated):")
print(by_age)

by_income = df.groupby("Customer_Income_Bracket_SIMULATED")["Revenue"].sum().sort_values(ascending=False)
print("\nRevenue by Income Bracket (simulated):")
print(by_income)

# Export a single summary table for documentation/README use
summary_rows = [
    ("Total Revenue", f"Rs.{total_revenue:,.0f}"),
    ("Total Orders", f"{total_orders:,}"),
    ("Total Units Sold", f"{total_units:,}"),
    ("Average Order Value", f"Rs.{aov:,.0f}"),
    ("Unique Customers", f"{unique_customers:,}"),
    ("Top Region", by_region.index[0]),
    ("Top Brand", df.groupby('Brand')['Revenue'].sum().idxmax()),
    ("Best Month", monthly.idxmax()),
    ("Avg Dispatch Delay (days)", f"{df['Dispatch_Delay_Days'].mean():.1f}"),
]
summary_df = pd.DataFrame(summary_rows, columns=["KPI", "Value"])
summary_df.to_csv("/home/claude/project/data/processed/kpi_summary.csv", index=False)
print("\nSaved -> kpi_summary.csv")
