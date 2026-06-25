"""
02_rfm_segmentation.py
---------------------------------------------------------------------------
PURPOSE
Build a customer segmentation model using RFM analysis (Recency, Frequency,
Monetary) — the industry-standard segmentation technique when you have
real transaction data but NOT demographic/CRM data.

WHY RFM AND NOT JUST "CLUSTER BY AGE"
Age/Income are useful for marketing CONTENT (what message, what channel),
but RFM tells you commercial VALUE (who to spend retention budget on).
A senior analyst always segments by behavior first, demographics second —
behavior predicts future revenue; demographics predict messaging style.
---------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/home/claude/project/data/processed/sales_clean.csv",
                  parse_dates=["Inward_Date", "Dispatch_Date"])

snapshot_date = df["Inward_Date"].max() + pd.Timedelta(days=1)

# ---------------------------------------------------------------------------
# STEP 1: Aggregate to customer level
#   Recency  = days since last purchase (lower = more engaged)
#   Frequency = number of orders (higher = more loyal)
#   Monetary  = total revenue (higher = more valuable)
# ---------------------------------------------------------------------------
rfm = df.groupby("Customer_Name").agg(
    Recency=("Inward_Date", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Order_ID", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

print(f"Customers analyzed: {len(rfm):,}")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# ---------------------------------------------------------------------------
# STEP 2: Score each dimension 1-5 using quintiles
# Business reasoning: quintile scoring is scale-independent — works whether
# you have 400 customers or 4 million, and doesn't require assuming a
# normal distribution (real purchase data is usually skewed).
# ---------------------------------------------------------------------------
rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

# ---------------------------------------------------------------------------
# STEP 3: Translate scores into business-readable segments
# Business reasoning: nobody in a leadership meeting wants to hear "RFM
# score 543" — they want "Champions" and a recommended action.
# ---------------------------------------------------------------------------
def segment(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New / Promising"
    elif r <= 2 and f >= 4 and m >= 4:
        return "At Risk (High Value)"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost / Churned"
    else:
        return "Needs Attention"

rfm["Segment"] = rfm.apply(segment, axis=1)

segment_action_map = {
    "Champions": "Reward with early access / loyalty perks. Protect, don't discount.",
    "Loyal Customers": "Upsell premium SKUs (e.g. higher RAM/ROM variants). Cross-sell accessories.",
    "New / Promising": "Onboard with a second-purchase incentive within 30 days.",
    "At Risk (High Value)": "Urgent win-back campaign — high revenue customers going cold.",
    "Needs Attention": "Re-engagement email/SMS with a moderate discount.",
    "Lost / Churned": "Low-cost reactivation only; do not over-invest.",
}
rfm["Recommended_Action"] = rfm["Segment"].map(segment_action_map)

print("\n--- Segment Distribution ---")
seg_summary = rfm.groupby("Segment").agg(
    Customers=("Customer_Name", "count"),
    Avg_Monetary=("Monetary", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Recency=("Recency", "mean"),
).sort_values("Avg_Monetary", ascending=False)
print(seg_summary)

out_path = "/home/claude/project/data/processed/customer_rfm_segments.csv"
rfm.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")

# ---------------------------------------------------------------------------
# STEP 4: Merge segment back onto transaction-level data for Power BI
# Business reasoning: Power BI needs segment available at the transaction
# grain so it can filter Sales/Product/Regional pages BY segment too.
# ---------------------------------------------------------------------------
df_with_segment = df.merge(rfm[["Customer_Name", "Segment", "RFM_Total"]],
                            on="Customer_Name", how="left")
df_with_segment.to_csv(
    "/home/claude/project/data/processed/sales_clean_with_segments.csv", index=False
)
print("Saved -> sales_clean_with_segments.csv (transaction-level + segment)")
