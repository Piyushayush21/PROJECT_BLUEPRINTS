/* =========================================================================
   02_business_queries.sql
   ---------------------------------------------------------------------
   PURPOSE
   The most important SQL queries an analyst would run for this project,
   each with a one-line "business question it answers" and notes on why
   the query is written the way it is.
   ========================================================================= */

-- -------------------------------------------------------------------------
-- Q1. What is total revenue, orders, and units by month?
-- BUSINESS QUESTION: "Is the business growing month over month?"
-- -------------------------------------------------------------------------
SELECT
    d.year_num,
    d.month_num,
    d.month_name,
    SUM(f.revenue)          AS total_revenue,
    COUNT(DISTINCT f.order_id) AS total_orders,
    SUM(f.quantity_sold)    AS total_units
FROM fact_sales f
JOIN dim_date d ON f.inward_date_key = d.date_key
GROUP BY d.year_num, d.month_num, d.month_name
ORDER BY d.year_num, d.month_num;


-- -------------------------------------------------------------------------
-- Q2. Which brands generate the most revenue, and what's their avg price?
-- BUSINESS QUESTION: "Where should we negotiate better supplier terms?"
-- WHY: high revenue + low avg price = high volume, low margin risk --
-- worth checking against actual cost data if/when available.
-- -------------------------------------------------------------------------
SELECT
    p.brand,
    p.product_type,
    SUM(f.revenue)            AS total_revenue,
    SUM(f.quantity_sold)      AS total_units,
    ROUND(AVG(f.price), 0)    AS avg_price
FROM fact_sales f
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY p.brand, p.product_type
ORDER BY total_revenue DESC;


-- -------------------------------------------------------------------------
-- Q3. Regional performance with % share of total revenue
-- BUSINESS QUESTION: "Which region deserves more marketing/inventory spend?"
-- WHY a window function: lets us show % share alongside the raw number in
-- one pass, instead of a separate query + manual division.
-- -------------------------------------------------------------------------
SELECT
    r.region_name,
    SUM(f.revenue) AS region_revenue,
    ROUND(
        SUM(f.revenue) * 100.0 / SUM(SUM(f.revenue)) OVER (), 1
    ) AS pct_of_total_revenue
FROM fact_sales f
JOIN dim_region r ON f.region_key = r.region_key
GROUP BY r.region_name
ORDER BY region_revenue DESC;


-- -------------------------------------------------------------------------
-- Q4. Top 10 customers by lifetime revenue (with their RFM segment)
-- BUSINESS QUESTION: "Who are our most valuable customers, and how should
-- the CRM team treat them differently?"
-- -------------------------------------------------------------------------
SELECT
    c.customer_name,
    c.rfm_segment,
    COUNT(DISTINCT f.order_id) AS total_orders,
    SUM(f.revenue)              AS lifetime_revenue
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
GROUP BY c.customer_name, c.rfm_segment
ORDER BY lifetime_revenue DESC
LIMIT 10;


-- -------------------------------------------------------------------------
-- Q5. Revenue contribution by RFM segment
-- BUSINESS QUESTION: "If we lost our 'At Risk' customers tomorrow, how much
-- revenue is genuinely at stake?" -- justifies retention budget.
-- -------------------------------------------------------------------------
SELECT
    c.rfm_segment,
    COUNT(DISTINCT c.customer_key) AS customers_in_segment,
    SUM(f.revenue)                  AS segment_revenue,
    ROUND(SUM(f.revenue) * 100.0 / SUM(SUM(f.revenue)) OVER (), 1) AS pct_of_revenue
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
GROUP BY c.rfm_segment
ORDER BY segment_revenue DESC;


-- -------------------------------------------------------------------------
-- Q6. Month-over-month revenue growth rate
-- BUSINESS QUESTION: "Is growth accelerating or decelerating?"
-- WHY LAG(): standard SQL pattern to compare each row to the prior period
-- without a self-join.
-- -------------------------------------------------------------------------
WITH monthly AS (
    SELECT
        d.year_num, d.month_num,
        SUM(f.revenue) AS monthly_revenue
    FROM fact_sales f
    JOIN dim_date d ON f.inward_date_key = d.date_key
    GROUP BY d.year_num, d.month_num
)
SELECT
    year_num, month_num, monthly_revenue,
    LAG(monthly_revenue) OVER (ORDER BY year_num, month_num) AS prior_month_revenue,
    ROUND(
        (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY year_num, month_num))
        * 100.0 / LAG(monthly_revenue) OVER (ORDER BY year_num, month_num), 1
    ) AS mom_growth_pct
FROM monthly
ORDER BY year_num, month_num;


-- -------------------------------------------------------------------------
-- Q7. Average dispatch delay by product type and region
-- BUSINESS QUESTION: "Where is our logistics/fulfillment underperforming?"
-- -------------------------------------------------------------------------
SELECT
    p.product_type,
    r.region_name,
    ROUND(AVG(f.dispatch_delay_days), 1) AS avg_dispatch_delay_days,
    COUNT(*) AS orders
FROM fact_sales f
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_region r ON f.region_key = r.region_key
WHERE f.dispatch_delay_days IS NOT NULL
GROUP BY p.product_type, r.region_name
ORDER BY avg_dispatch_delay_days DESC;


-- -------------------------------------------------------------------------
-- Q8. Most popular processor specs by units sold (within each category)
-- BUSINESS QUESTION: "What specs should we stock more of next quarter?"
-- WHY RANK(): lets us get "top 3 per category" instead of one global list,
-- which would be dominated by whichever category has more SKUs.
-- -------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        p.product_type,
        p.processor,
        SUM(f.quantity_sold) AS units_sold,
        RANK() OVER (PARTITION BY p.product_type ORDER BY SUM(f.quantity_sold) DESC) AS rnk
    FROM fact_sales f
    JOIN dim_product p ON f.product_key = p.product_key
    GROUP BY p.product_type, p.processor
)
SELECT product_type, processor, units_sold
FROM ranked
WHERE rnk <= 3
ORDER BY product_type, rnk;


-- -------------------------------------------------------------------------
-- Q9. Customer cohort retention check: customers active in month 1 who
-- also purchased in month 2 (simple retention proxy)
-- BUSINESS QUESTION: "Are new customers coming back?"
-- -------------------------------------------------------------------------
WITH first_purchase AS (
    SELECT
        customer_key,
        MIN(DATE_FORMAT(d.full_date, '%Y-%m-01')) AS cohort_month
    FROM fact_sales f
    JOIN dim_date d ON f.inward_date_key = d.date_key
    GROUP BY customer_key
),
activity AS (
    SELECT DISTINCT
        f.customer_key,
        DATE_FORMAT(d.full_date, '%Y-%m-01') AS activity_month
    FROM fact_sales f
    JOIN dim_date d ON f.inward_date_key = d.date_key
)
SELECT
    fp.cohort_month,
    COUNT(DISTINCT fp.customer_key) AS cohort_size,
    COUNT(DISTINCT CASE
        WHEN a.activity_month = DATE_ADD(fp.cohort_month, INTERVAL 1 MONTH)
        THEN a.customer_key END) AS retained_month_2
FROM first_purchase fp
LEFT JOIN activity a ON fp.customer_key = a.customer_key
GROUP BY fp.cohort_month
ORDER BY fp.cohort_month;


-- -------------------------------------------------------------------------
-- Q10. [SIMULATED FIELDS] Revenue by age band and income bracket
-- BUSINESS QUESTION: "Which demographic should marketing target for new
-- product launches?"
-- NOTE: age_simulated / income_bracket_simulated are enrichment fields,
-- not present in the original Kaggle dataset. See README.
-- -------------------------------------------------------------------------
SELECT
    c.age_band_simulated,
    c.income_bracket_simulated,
    SUM(f.revenue) AS total_revenue,
    COUNT(DISTINCT c.customer_key) AS customers
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
GROUP BY c.age_band_simulated, c.income_bracket_simulated
ORDER BY total_revenue DESC;
