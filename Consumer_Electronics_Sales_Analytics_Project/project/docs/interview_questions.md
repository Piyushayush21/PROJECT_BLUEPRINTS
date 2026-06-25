# Data Analyst Interview Questions & Answers
### Based on the "Consumer Electronics Sales Analytics & Customer Segmentation" Project

These are written the way an interviewer would actually probe a candidate who lists this
project on their resume — including the harder, "defend your choices" questions.

---

### Q1. Walk me through your project end-to-end in two minutes.

**A:** I took a transactional mobiles/laptop sales dataset and built a full pipeline:
cleaned and validated the data in Python, designed a star-schema SQL database for proper
querying, segmented customers using RFM analysis since the data had no CRM fields, and
designed a 5-page Power BI dashboard covering executive, sales, product, customer, and
regional views. The goal was to answer one business question: which products, regions,
and customers actually drive revenue, so the business can target marketing and retention
spend instead of treating every customer the same.

---

### Q2. Why did you choose RFM segmentation instead of clustering on demographics?

**A:** The real dataset only had transaction-level fields — no age, income, or CRM data.
RFM (Recency, Frequency, Monetary) only needs purchase history, so it was the right tool
for the data I actually had. It's also more directly tied to business value than
demographics: RFM tells you *who's worth retaining*, while demographics tell you *how to
message them*. I'd use both together if demographic data were available — segment by
value first, then tailor messaging within each value segment by demographic.

---

### Q3. How did you decide on quintile-based scoring instead of fixed thresholds for RFM?

**A:** Quintiles (`pd.qcut`) split customers into 5 equal-sized groups per dimension
regardless of the underlying distribution shape. Purchase data is usually skewed — a few
customers spend far more than the median — so fixed thresholds (like "Frequency > 100 =
high") would be arbitrary and break if the business doubles in size. Quintiles
self-adjust to whatever the current data looks like.

---

### Q4. Your dataset didn't have a Cost column. How did that limit your analysis, and what would you do about it?

**A:** Without Cost, I could only report revenue, not gross margin. A product can have
high revenue and still be unprofitable if its margin is thin — revenue alone hides that.
I flagged this explicitly in my README and recommended it as the top next step for the
data pipeline: even an approximate cost-per-SKU field would unlock real profitability
analysis instead of just sales volume analysis. In the meantime, I used a price-vs-volume
scatter plot as an imperfect proxy — high-volume, low-price brands are the ones most worth
investigating for margin risk first.

---

### Q5. You used simulated demographic data. Isn't that misleading?

**A:** Only if it's not disclosed — which is why I labeled every simulated column with a
`_SIMULATED` suffix in the data itself, flagged it in the README's first section, and
required any dashboard visual using those fields to say "(Simulated data)" directly in the
visual title. The brief asked for demographic target-audience analysis, but the real
dataset doesn't have those fields. Rather than skip that requirement or quietly fabricate
numbers, I built it as clearly-labeled illustrative analysis and was upfront that it
should be replaced with real CRM data the moment it's available. Hiding that distinction
would be the actually misleading move.

---

### Q6. Why a star schema instead of just querying one flat CSV?

**A:** Three reasons. First, normalization avoids storing repeated text (brand, region,
processor names) across tens of thousands of rows — better storage and consistency, since
a typo like "West " vs "West" would otherwise silently create a duplicate category.
Second, foreign keys let the database enforce valid values, which a flat file can't do.
Third, and most practically, Power BI's data model is built around star schemas — fact
table in the middle, dimension tables around it — so designing it that way from day one
makes the Power BI build straightforward instead of needing rework later.

---

### Q7. How did you validate that your numbers were correct before building the dashboard?

**A:** I computed every KPI directly in Python first — total revenue, AOV, regional
splits, segment revenue share — and printed them as a "ground truth" reference. Then I
built the actual SQL database and ran the same business questions as SQL queries against
it, and confirmed the numbers matched exactly. Only after both independently agreed did I
treat the numbers as safe to put in front of a dashboard. If Power BI later showed a
different number, I'd know immediately it was a DAX or relationship bug, not a real
insight — because I had two independent calculations to compare against.

---

### Q8. What does your `dim_date` table give you that you couldn't get from just using the date column directly?

**A:** Power BI's time intelligence functions (like `DATEADD`, `SAMEPERIODLASTYEAR`) only
work reliably against a marked Date Table with one row per calendar day, no gaps. It also
gives clean, pre-built Year/Quarter/Month hierarchies for slicers, and avoids having to
recompute things like "month name" or "quarter number" with messy date-string logic
inside every visual.

---

### Q9. Your MoM growth query uses `LAG()`. Explain what that's doing and why not just a self-join.

**A:** `LAG()` is a window function that looks at the previous row's value, ordered by
month, without needing a second copy of the table joined back to itself. A self-join would
require matching `month = month - 1`, which gets messy with month/year boundaries (e.g.
December to January). `LAG() OVER (ORDER BY year, month)` handles that boundary correctly
and is far more readable.

---

### Q10. If this dashboard goes into production, how would you make sure it stays accurate over time?

**A:** Three things. First, automate the refresh — scheduled Python cleaning jobs feeding
a database, with Power BI Service set to pull on a schedule rather than manual CSV
re-uploads, which are where most "stale dashboard" problems come from. Second, keep the
validation step from Q7 as a permanent part of the pipeline — re-run the Python KPI
calculation each refresh and alert if it drifts from the database/dashboard numbers.
Third, set up threshold alerts (e.g., "alert if monthly revenue drops more than 10%") so
problems are caught the day they happen, not the day someone happens to open the report.

---

### Q11. What was the single most important business insight from this project, and why?

**A:** That Champions and Loyal Customers — under half the customer base — generate over
half of total revenue. That's the clearest, most actionable finding because it directly
justifies reallocating marketing budget from broad acquisition toward retention for a
specific, identifiable group, and it's the kind of insight that's invisible until you
actually segment customers by behavior instead of looking at company-wide averages.

---

### Q12. What would you do differently if you had another week on this project?

**A:** Two things. I'd push to get the real dataset and real cost data rather than working
around their absence, since margin analysis is the single biggest gap right now. And I'd
build a basic cohort retention analysis further than the simple month-2 check I included —
tracking full retention curves over 6–12 months would tell us not just who's valuable
today, but whether the business's overall retention is improving or declining over time.
