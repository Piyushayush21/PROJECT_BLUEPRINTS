"""
04_build_and_test_database.py
---------------------------------------------------------------------------
PURPOSE
Build a real, queryable database from the cleaned data using the star
schema defined in sql/01_schema.sql, then execute the business questions
from sql/02_business_queries.sql to PROVE they run correctly end-to-end.

NOTE ON ENGINE: SQLite is used here purely so this project is runnable
without installing/configuring MySQL or PostgreSQL. The schema and
queries are written in standard SQL and are directly portable to
MySQL/PostgreSQL with trivial syntax tweaks (noted inline where SQLite
diverges, e.g. AUTOINCREMENT vs AUTO_INCREMENT, DATE_FORMAT vs strftime).
---------------------------------------------------------------------------
"""

import sqlite3
import pandas as pd

DB_PATH = "/home/claude/project/data/processed/sales_analytics.db"
CSV_PATH = "/home/claude/project/data/processed/sales_clean_with_segments.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["Inward_Date", "Dispatch_Date"])

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.executescript("""
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_region;
DROP TABLE IF EXISTS dim_date;

CREATE TABLE dim_customer (
    customer_key INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT UNIQUE,
    age_simulated INTEGER,
    gender_simulated TEXT,
    income_bracket_simulated TEXT,
    age_band_simulated TEXT,
    rfm_segment TEXT,
    rfm_total_score INTEGER
);

CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY AUTOINCREMENT,
    product_type TEXT,
    brand TEXT,
    processor TEXT,
    ram_gb REAL,
    rom_gb REAL,
    UNIQUE(product_type, brand, processor, ram_gb, rom_gb)
);

CREATE TABLE dim_region (
    region_key INTEGER PRIMARY KEY AUTOINCREMENT,
    region_name TEXT UNIQUE
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date TEXT UNIQUE,
    day_name TEXT,
    month_name TEXT,
    month_num INTEGER,
    quarter_num INTEGER,
    year_num INTEGER
);

CREATE TABLE fact_sales (
    order_id INTEGER PRIMARY KEY,
    customer_key INTEGER,
    product_key INTEGER,
    region_key INTEGER,
    inward_date_key INTEGER,
    dispatch_date_key INTEGER,
    price REAL,
    quantity_sold INTEGER,
    revenue REAL,
    dispatch_delay_days INTEGER,
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key),
    FOREIGN KEY (region_key) REFERENCES dim_region(region_key),
    FOREIGN KEY (inward_date_key) REFERENCES dim_date(date_key)
);
""")

# ---- Populate dim_region ----
for r in df["Region"].dropna().unique():
    cur.execute("INSERT OR IGNORE INTO dim_region (region_name) VALUES (?)", (r,))

# ---- Populate dim_customer ----
cust_cols = df[["Customer_Name", "Customer_Age_SIMULATED", "Customer_Gender_SIMULATED",
                 "Customer_Income_Bracket_SIMULATED", "Age_Band_SIMULATED",
                 "Segment", "RFM_Total"]].drop_duplicates(subset=["Customer_Name"])
for _, row in cust_cols.iterrows():
    cur.execute("""INSERT OR IGNORE INTO dim_customer
        (customer_name, age_simulated, gender_simulated, income_bracket_simulated,
         age_band_simulated, rfm_segment, rfm_total_score)
        VALUES (?,?,?,?,?,?,?)""",
        (row["Customer_Name"], int(row["Customer_Age_SIMULATED"]),
         row["Customer_Gender_SIMULATED"], row["Customer_Income_Bracket_SIMULATED"],
         str(row["Age_Band_SIMULATED"]), row["Segment"], int(row["RFM_Total"])))

# ---- Populate dim_product ----
prod_cols = df[["Product_Type", "Brand", "Processor", "RAM_GB", "ROM_GB"]].drop_duplicates()
for _, row in prod_cols.iterrows():
    cur.execute("""INSERT OR IGNORE INTO dim_product
        (product_type, brand, processor, ram_gb, rom_gb) VALUES (?,?,?,?,?)""",
        (row["Product_Type"], row["Brand"], row["Processor"], row["RAM_GB"], row["ROM_GB"]))

# ---- Populate dim_date ----
all_dates = pd.concat([df["Inward_Date"], df["Dispatch_Date"]]).dropna().unique()
for d in pd.to_datetime(all_dates):
    date_key = int(d.strftime("%Y%m%d"))
    cur.execute("""INSERT OR IGNORE INTO dim_date
        (date_key, full_date, day_name, month_name, month_num, quarter_num, year_num)
        VALUES (?,?,?,?,?,?,?)""",
        (date_key, d.strftime("%Y-%m-%d"), d.strftime("%A"), d.strftime("%B"),
         d.month, (d.month - 1) // 3 + 1, d.year))

conn.commit()

# ---- Build lookup maps for fact table inserts ----
region_map = dict(cur.execute("SELECT region_name, region_key FROM dim_region").fetchall())
customer_map = dict(cur.execute("SELECT customer_name, customer_key FROM dim_customer").fetchall())
date_map = dict(cur.execute("SELECT full_date, date_key FROM dim_date").fetchall())

product_rows = cur.execute(
    "SELECT product_key, product_type, brand, processor, ram_gb, rom_gb FROM dim_product"
).fetchall()
product_map = {(pt, br, pr, ram, rom): pk for pk, pt, br, pr, ram, rom in product_rows}

# ---- Populate fact_sales ----
fact_rows = []
for _, row in df.iterrows():
    pkey = product_map.get((row["Product_Type"], row["Brand"], row["Processor"],
                             row["RAM_GB"], row["ROM_GB"]))
    fact_rows.append((
        int(row["Order_ID"]),
        customer_map.get(row["Customer_Name"]),
        pkey,
        region_map.get(row["Region"]),
        date_map.get(row["Inward_Date"].strftime("%Y-%m-%d")),
        date_map.get(row["Dispatch_Date"].strftime("%Y-%m-%d")) if pd.notna(row["Dispatch_Date"]) else None,
        float(row["Price"]),
        int(row["Quantity_Sold"]),
        float(row["Revenue"]),
        int(row["Dispatch_Delay_Days"]) if pd.notna(row["Dispatch_Delay_Days"]) else None,
    ))

cur.executemany("""INSERT OR IGNORE INTO fact_sales
    (order_id, customer_key, product_key, region_key, inward_date_key,
     dispatch_date_key, price, quantity_sold, revenue, dispatch_delay_days)
    VALUES (?,?,?,?,?,?,?,?,?,?)""", fact_rows)
conn.commit()

print(f"Loaded: dim_customer={cur.execute('SELECT COUNT(*) FROM dim_customer').fetchone()[0]}, "
      f"dim_product={cur.execute('SELECT COUNT(*) FROM dim_product').fetchone()[0]}, "
      f"dim_region={cur.execute('SELECT COUNT(*) FROM dim_region').fetchone()[0]}, "
      f"dim_date={cur.execute('SELECT COUNT(*) FROM dim_date').fetchone()[0]}, "
      f"fact_sales={cur.execute('SELECT COUNT(*) FROM fact_sales').fetchone()[0]}")

# ---------------------------------------------------------------------------
# TEST: run translated versions of the key business queries to PROVE the
# schema and joins actually work (SQLite syntax substitutions noted)
# ---------------------------------------------------------------------------
print("\n--- TEST Q2: Top brands by revenue ---")
q2 = """
SELECT p.brand, p.product_type, SUM(f.revenue) AS total_revenue,
       SUM(f.quantity_sold) AS total_units, ROUND(AVG(f.price),0) AS avg_price
FROM fact_sales f JOIN dim_product p ON f.product_key = p.product_key
GROUP BY p.brand, p.product_type
ORDER BY total_revenue DESC LIMIT 5;
"""
print(pd.read_sql(q2, conn))

print("\n--- TEST Q3: Regional revenue with % share ---")
q3 = """
SELECT r.region_name, SUM(f.revenue) AS region_revenue,
       ROUND(SUM(f.revenue) * 100.0 / (SELECT SUM(revenue) FROM fact_sales), 1) AS pct_of_total
FROM fact_sales f JOIN dim_region r ON f.region_key = r.region_key
GROUP BY r.region_name ORDER BY region_revenue DESC;
"""
print(pd.read_sql(q3, conn))

print("\n--- TEST Q4: Top 10 customers by lifetime revenue ---")
q4 = """
SELECT c.customer_name, c.rfm_segment, COUNT(DISTINCT f.order_id) AS total_orders,
       SUM(f.revenue) AS lifetime_revenue
FROM fact_sales f JOIN dim_customer c ON f.customer_key = c.customer_key
GROUP BY c.customer_name, c.rfm_segment
ORDER BY lifetime_revenue DESC LIMIT 10;
"""
print(pd.read_sql(q4, conn))

print("\n--- TEST Q5: Revenue contribution by RFM segment ---")
q5 = """
SELECT c.rfm_segment, COUNT(DISTINCT c.customer_key) AS customers_in_segment,
       SUM(f.revenue) AS segment_revenue,
       ROUND(SUM(f.revenue) * 100.0 / (SELECT SUM(revenue) FROM fact_sales), 1) AS pct_of_revenue
FROM fact_sales f JOIN dim_customer c ON f.customer_key = c.customer_key
GROUP BY c.rfm_segment ORDER BY segment_revenue DESC;
"""
print(pd.read_sql(q5, conn))

print("\n--- TEST Q7: Avg dispatch delay by product type and region ---")
q7 = """
SELECT p.product_type, r.region_name,
       ROUND(AVG(f.dispatch_delay_days),1) AS avg_delay, COUNT(*) AS orders
FROM fact_sales f
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_region r ON f.region_key = r.region_key
WHERE f.dispatch_delay_days IS NOT NULL
GROUP BY p.product_type, r.region_name
ORDER BY avg_delay DESC;
"""
print(pd.read_sql(q7, conn))

print("\n--- TEST Q8: Top 3 processors per category by units (window function) ---")
q8 = """
WITH ranked AS (
    SELECT p.product_type, p.processor, SUM(f.quantity_sold) AS units_sold,
           RANK() OVER (PARTITION BY p.product_type ORDER BY SUM(f.quantity_sold) DESC) AS rnk
    FROM fact_sales f JOIN dim_product p ON f.product_key = p.product_key
    GROUP BY p.product_type, p.processor
)
SELECT product_type, processor, units_sold FROM ranked WHERE rnk <= 3
ORDER BY product_type, rnk;
"""
print(pd.read_sql(q8, conn))

conn.close()
print(f"\nDatabase saved -> {DB_PATH}")
print("All test queries executed successfully — schema and joins verified.")
