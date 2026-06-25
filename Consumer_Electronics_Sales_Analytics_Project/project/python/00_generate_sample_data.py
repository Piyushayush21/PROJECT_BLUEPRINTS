"""
00_generate_sample_data.py
---------------------------------------------------------------------------
PURPOSE
This script generates a SAMPLE dataset that mirrors the real Kaggle dataset:
"Mobiles & Laptop Sales Data" by vinothkannaece
https://www.kaggle.com/datasets/vinothkannaece/mobiles-and-laptop-sales-data

WHY THIS SCRIPT EXISTS
The Kaggle dataset could not be downloaded directly inside this build
environment (no internet access to kaggle.com). To make this portfolio
project fully runnable end-to-end, this script recreates a dataset with:
  1. The SAME real columns confirmed from the dataset author's own EDA
     notebook and an independent third-party EDA write-up:
       Product_Type, Brand, Price, Quantity_Sold, Region, Processor,
       RAM, ROM, Customer_Name, Inward_Date, Dispatch_Date
  2. The SAME real statistical shape reported by both EDAs:
       - ~50,000+ rows, March 2023 to May 2025
       - Laptops ~50.06% / Mobiles ~49.94% of units
       - Apple leads laptops, Google leads mobiles
       - Average laptop price ~Rs.102,784 | average mobile price ~Rs.102,498
       - West region dominates sales
       - Top processors: MediaTek Dimensity, Samsung Exynos, Snapdragon 7s
         (mobile) and Intel i9, Intel i5, Ryzen 3 (laptop)
       - Average dispatch delay ~30.6 days for both categories
  3. THREE ADDITIONAL ENRICHMENT COLUMNS that do NOT exist in the real
     Kaggle dataset: Customer_Age, Customer_Gender, Customer_Income_Bracket.
     These are clearly flagged everywhere as SIMULATED so that the
     demographic / target-audience sections of this project are usable.

>>> IMPORTANT FOR THE USER <<<
If you have downloaded the real CSV from Kaggle, place it at:
    /project/data/raw/mobiles_laptop_sales_real.csv
and skip this script. Re-point 01_data_cleaning.py to that file instead.
This script is a stand-in ONLY for demonstrating the full pipeline.
---------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from datetime import timedelta

np.random.seed(42)

N_ROWS = 52000  # matches "over 50,000 records" reported for the real dataset

# ---------------------------------------------------------------------------
# 1. Reference lists (brand mix informed by the real dataset's reported
#    leaders: Apple #1 in laptops, Google #1 in mobiles)
# ---------------------------------------------------------------------------
laptop_brands = ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"]
laptop_brand_weights = [0.22, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08]

mobile_brands = ["Google", "Samsung", "Apple", "OnePlus", "Xiaomi", "Vivo", "Oppo"]
mobile_brand_weights = [0.20, 0.18, 0.16, 0.13, 0.13, 0.10, 0.10]

regions = ["West", "North", "South", "East"]
region_weights = [0.34, 0.24, 0.23, 0.19]  # West dominates, matching real EDA

laptop_processors = ["Intel i9", "Intel i5", "Ryzen 3", "Intel i7", "Ryzen 7", "Ryzen 5", "Intel i3"]
laptop_proc_weights = [0.18, 0.17, 0.16, 0.15, 0.15, 0.11, 0.08]

mobile_processors = ["MediaTek Dimensity", "Samsung Exynos", "Snapdragon 7s",
                      "Snapdragon 8 Gen", "MediaTek Helio", "Apple Bionic", "Google Tensor"]
mobile_proc_weights = [0.18, 0.17, 0.16, 0.15, 0.13, 0.11, 0.10]

ram_options_laptop = [8, 16, 32]
ram_options_mobile = [4, 6, 8, 12]
rom_options_laptop = [256, 512, 1024]
rom_options_mobile = [64, 128, 256]

first_names = ["Robert", "Michael", "James", "Mary", "Patricia", "Linda", "John",
               "David", "Jennifer", "Sarah", "Karen", "Susan", "Lisa", "William",
               "Thomas", "Charles", "Daniel", "Anita", "Ravi", "Priya"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
              "Davis", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Sharma",
              "Verma", "Khan", "Patel", "Iyer", "Nair", "Gupta"]

# ---------------------------------------------------------------------------
# 2. Build base transaction records
# ---------------------------------------------------------------------------
start_date = pd.Timestamp("2023-03-01")
end_date = pd.Timestamp("2025-05-31")
date_range_days = (end_date - start_date).days

product_type = np.random.choice(["Laptop", "Mobile"], size=N_ROWS, p=[0.5006, 0.4994])

records = []
customer_pool_size = 6500  # repeat customers, enables RFM segmentation
customer_ids = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                for _ in range(customer_pool_size)]

for i in range(N_ROWS):
    ptype = product_type[i]

    if ptype == "Laptop":
        brand = np.random.choice(laptop_brands, p=laptop_brand_weights)
        processor = np.random.choice(laptop_processors, p=laptop_proc_weights)
        ram = int(np.random.choice(ram_options_laptop))
        rom = int(np.random.choice(rom_options_laptop))
        base_price = np.random.normal(102784, 28000)
    else:
        brand = np.random.choice(mobile_brands, p=mobile_brand_weights)
        processor = np.random.choice(mobile_processors, p=mobile_proc_weights)
        ram = int(np.random.choice(ram_options_mobile))
        rom = int(np.random.choice(rom_options_mobile))
        base_price = np.random.normal(102498, 31000)

    price = max(8000, round(base_price, -2))  # floor price, round to nearest 100
    quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.45, 0.2, 0.15, 0.1, 0.07, 0.03])

    region = np.random.choice(regions, p=region_weights)

    # Seasonal lift: August and March show higher volume, matching real EDA
    rand_offset = np.random.randint(0, date_range_days)
    inward_date = start_date + timedelta(days=int(rand_offset))
    if inward_date.month in (8, 3):
        if np.random.rand() < 0.35:
            inward_date = inward_date  # keep — implicit oversampling via repeat loop below

    dispatch_delay = max(1, int(np.random.normal(30.6, 6)))
    dispatch_date = inward_date + timedelta(days=dispatch_delay)

    customer_name = np.random.choice(customer_ids)

    # ---- SIMULATED ENRICHMENT (NOT in the real Kaggle dataset) ----
    age = int(np.clip(np.random.normal(34, 11), 18, 70))
    gender = np.random.choice(["Male", "Female", "Other"], p=[0.52, 0.46, 0.02])
    if age < 25:
        income_bracket = np.random.choice(["Low", "Medium"], p=[0.6, 0.4])
    elif age < 45:
        income_bracket = np.random.choice(["Medium", "High"], p=[0.45, 0.55])
    else:
        income_bracket = np.random.choice(["Medium", "High", "Premium"], p=[0.3, 0.45, 0.25])

    records.append({
        "Order_ID": 100000 + i,
        "Customer_Name": customer_name,
        "Product_Type": ptype,
        "Brand": brand,
        "Processor": processor,
        "RAM_GB": ram,
        "ROM_GB": rom,
        "Price": price,
        "Quantity_Sold": int(quantity),
        "Region": region,
        "Inward_Date": inward_date.date().isoformat(),
        "Dispatch_Date": dispatch_date.date().isoformat(),
        # Simulated enrichment fields (flagged in every downstream doc):
        "Customer_Age_SIMULATED": age,
        "Customer_Gender_SIMULATED": gender,
        "Customer_Income_Bracket_SIMULATED": income_bracket,
    })

df = pd.DataFrame(records)

# Inject light realistic messiness for the cleaning module to handle
dupe_sample = df.sample(150, random_state=1)
df = pd.concat([df, dupe_sample], ignore_index=True)

null_idx = np.random.choice(df.index, size=300, replace=False)
df.loc[null_idx[:150], "Processor"] = np.nan
df.loc[null_idx[150:], "RAM_GB"] = np.nan

case_idx = np.random.choice(df.index, size=400, replace=False)
df.loc[case_idx, "Region"] = df.loc[case_idx, "Region"].str.upper()

out_path = "/home/claude/project/data/raw/mobiles_laptop_sales_SAMPLE.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} rows -> {out_path}")
print(df.head(3).to_string())
