# Employability & Unemployment by Degree in India

A data analytics project comparing graduate employability across major Indian degree
streams — **Engineering, BCA, B.Sc, B.Com, B.A, MCA, and MBA/Management** — from 2024 to 2026.

## Project Files

| File | Description |
|---|---|
| `Employability_By_Degree_India_Report.docx` | Full written report: introduction, methodology, data table, 5 charts with analysis, key findings, conclusion, references |
| `Employability_By_Degree_India_Analysis.ipynb` | Jupyter notebook with all Python code — builds the dataset and regenerates every chart from scratch |
| `india_employability_by_degree.csv` | Main dataset: Year, Degree, Employability %, Estimated Skill Gap %, Data Type (Reported/Estimated), Source |
| `india_employability_by_gender.csv` | Supporting dataset: all-India male vs. female employability, 2024–2026 |

##  Objective

Compare how employable graduates of different degree streams are, track how this has
changed year over year, and identify which degrees carry the largest "skill gap" risk.

##  Data Source

**India Skills Report (ISR)** — published annually by **Wheebox**, in partnership with
**CII** (Confederation of Indian Industry) and **AICTE**. Based on the Global Employability
Test (GET), administered to 5–6.5+ lakh graduating students per year across India, plus
feedback from 1,000+ hiring companies.

**Why this source?** India's official labour survey (PLFS, by MoSPI) reports unemployment
by general *education level* (e.g. "Graduate & above") but does **not** break it down by
specific *degree stream*. ISR is the standard real-world source used for exactly this kind
of degree-wise comparison, and is what most Indian news coverage on this topic relies on.

##  Important: Employability ≠ Unemployment

- **Employability %** = the share of graduates considered industry-ready *right now*,
  per ISR's standardised skills assessment.
- **Estimated Skill Gap %** = `100 − Employability %`. A proxy for "graduates not yet
  employable," used informally in media commentary in an unemployment-style sense.
- Neither of these is the same as the **official unemployment rate** (% of the labour
  force jobless while actively seeking work), which PLFS reports — but not by degree.

This distinction is explained in full in Section 2 of the report.

##  Data Quality Flag

Every row in `india_employability_by_degree.csv` is tagged:

- **Reported** — taken directly from published ISR figures (Engineering, B.Com, B.Sc,
  B.A, MBA/Management for 2025–2026; Engineering and B.Com for 2024).
- **Estimated** — interpolated where ISR doesn't publish an exact figure for that year/degree.
  This applies most importantly to **BCA**, which ISR does not track as a standalone
  category — it's estimated here as a computer-applications stream between the
  Commerce/Science tier and the MCA/Engineering tier.

**For formal academic submission:** verify "Reported" figures against the original
India Skills Report PDFs at [wheebox.com](https://wheebox.com), and treat "Estimated"
figures (especially BCA) as illustrative, not official.

##  Tools Used

- Python — Pandas (data handling), Matplotlib & Seaborn (visualization)
- Jupyter Notebook
- Microsoft Word (final report, via docx generation)

##  How to Run

1. Open `Employability_By_Degree_India_Analysis.ipynb` in Jupyter / VS Code / Google Colab.
2. Run all cells top to bottom — it rebuilds the dataset and regenerates all 5 charts.
3. Install dependencies if needed: `pip install pandas matplotlib seaborn nbformat`

## Key Findings

- **Engineering** is the most employable degree in 2026 (80.0%), up sharply from 64% in 2024.
- **B.A** has the lowest employability throughout 2024–2026 (55.55% in 2026).
- **Technical/professional degrees** (Engineering, MCA, MBA) consistently outperform
  **general degrees** (B.A, B.Sc, B.Com).
- **BCA** sits in the middle of the pack — but remember, this is an *estimated* figure.
- **Gender gap reversal:** female employability (54.0%) overtook male (51.5%) for the
  first time in 5 years in 2026.

## References

- Wheebox, CII, AICTE — *India Skills Report*, 2024, 2025, 2026 editions. [wheebox.com](https://wheebox.com)
- Statista — employability statistics citing Wheebox/ISR data
- News coverage of ISR 2025/2026 used to cross-verify degree-wise figures
