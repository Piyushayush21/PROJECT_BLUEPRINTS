/* =========================================================================
   01_schema.sql
   ---------------------------------------------------------------------
   PURPOSE
   Define a normalized relational schema for the cleaned sales data so it
   can be loaded into a proper database (MySQL/PostgreSQL/SQL Server)
   instead of querying one flat CSV forever.

   WHY NORMALIZE AT ALL? (beginner explanation)
   The cleaned CSV is one wide "flat" table — easy for Python/Excel, but
   bad for a real database because:
     1. Brand/Region/Processor names repeat thousands of times -> wasted
        storage and risk of typos creating fake duplicate categories.
     2. You can't easily enforce "Region must be one of 4 valid values"
        on a flat file — a database CAN enforce that with a foreign key.
     3. Star-schema style design (fact + dimension tables) is exactly what
        Power BI expects for fast, correct aggregations.

   DESIGN: A STAR SCHEMA
     FACT TABLE:  fact_sales        (one row per transaction)
     DIMENSIONS:  dim_customer, dim_product, dim_region, dim_date
   ========================================================================= */

-- ---------------------------------------------------------------------
-- DIMENSION: Customer
-- Business reasoning: customer attributes (segment, age band) change
-- slowly and get reused across thousands of orders — storing them once
-- here avoids repeating "Champions" / "25-34" on every single row.
-- ---------------------------------------------------------------------
CREATE TABLE dim_customer (
    customer_key        INT PRIMARY KEY AUTO_INCREMENT,
    customer_name        VARCHAR(100) NOT NULL,
    age_simulated         INT NULL,
    gender_simulated      VARCHAR(10) NULL,
    income_bracket_simulated VARCHAR(20) NULL,
    age_band_simulated    VARCHAR(10) NULL,
    rfm_segment           VARCHAR(30) NULL,
    rfm_total_score       INT NULL,
    CONSTRAINT uq_customer_name UNIQUE (customer_name)
);

-- ---------------------------------------------------------------------
-- DIMENSION: Product
-- Business reasoning: a product's brand/processor/spec doesn't change
-- per sale — defining it once lets us add a real Cost field later
-- (for margin analysis) without touching the fact table at all.
-- ---------------------------------------------------------------------
CREATE TABLE dim_product (
    product_key      INT PRIMARY KEY AUTO_INCREMENT,
    product_type     VARCHAR(20) NOT NULL,      -- Laptop / Mobile
    brand             VARCHAR(50) NOT NULL,
    processor         VARCHAR(50) NULL,
    ram_gb            DECIMAL(5,1) NULL,
    rom_gb            DECIMAL(6,1) NULL,
    CONSTRAINT uq_product UNIQUE (product_type, brand, processor, ram_gb, rom_gb)
);

-- ---------------------------------------------------------------------
-- DIMENSION: Region
-- ---------------------------------------------------------------------
CREATE TABLE dim_region (
    region_key   INT PRIMARY KEY AUTO_INCREMENT,
    region_name  VARCHAR(20) NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------
-- DIMENSION: Date
-- Business reasoning: a dedicated date table is THE standard Power BI
-- practice — it enables clean Year/Quarter/Month slicers and time
-- intelligence (YoY, MoM growth) without DAX gymnastics.
-- ---------------------------------------------------------------------
CREATE TABLE dim_date (
    date_key     INT PRIMARY KEY,         -- format YYYYMMDD
    full_date    DATE NOT NULL UNIQUE,
    day_name     VARCHAR(10),
    month_name   VARCHAR(10),
    month_num    INT,
    quarter_num  INT,
    year_num     INT
);

-- ---------------------------------------------------------------------
-- FACT TABLE: Sales
-- Business reasoning: this is the "grain" of the business question --
-- one row = one order line. Every KPI (revenue, units, AOV) is a SUM or
-- COUNT over this table, filtered/grouped by the dimension keys.
-- ---------------------------------------------------------------------
CREATE TABLE fact_sales (
    order_id              BIGINT PRIMARY KEY,
    customer_key          INT NOT NULL,
    product_key           INT NOT NULL,
    region_key             INT NOT NULL,
    inward_date_key        INT NOT NULL,
    dispatch_date_key      INT NULL,
    price                  DECIMAL(12,2) NOT NULL,
    quantity_sold          INT NOT NULL,
    revenue                DECIMAL(14,2) NOT NULL,   -- price * quantity_sold, stored for speed
    dispatch_delay_days    INT NULL,
    CONSTRAINT fk_customer FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    CONSTRAINT fk_product  FOREIGN KEY (product_key)  REFERENCES dim_product(product_key),
    CONSTRAINT fk_region   FOREIGN KEY (region_key)   REFERENCES dim_region(region_key),
    CONSTRAINT fk_inward_date FOREIGN KEY (inward_date_key) REFERENCES dim_date(date_key)
);

-- Indexes: speed up the GROUP BYs every dashboard page will run
CREATE INDEX idx_fact_customer ON fact_sales(customer_key);
CREATE INDEX idx_fact_product  ON fact_sales(product_key);
CREATE INDEX idx_fact_region   ON fact_sales(region_key);
CREATE INDEX idx_fact_date     ON fact_sales(inward_date_key);
