# Interview Preparation — SFEWS Project

Organized by theme. Each answer is written the way you should actually say it
out loud — specific, honest about tradeoffs, no hand-waving.

---

## 1. Data & Problem Framing

**Q: Why synthetic data? Doesn't that undermine the whole project?**

No — it's a disclosed, deliberate tradeoff, and I'd make the same call again.
Public startup datasets (Crunchbase exports, the Kaggle "startup success" set)
are static: one row per company, no monthly operating history. That structure
can only support "did this company eventually fail," which is a much weaker
problem than "is this company showing early warning signs *right now*." To
build a genuine early-warning system I needed a longitudinal panel, which
doesn't exist publicly at any real scale. So I built a structural simulation
calibrated against published industry statistics (CB Insights post-mortems,
Startup Genome Report, First Round Capital benchmarks) and disclosed it
everywhere — README, app, model card. The transferable part is the
methodology: leakage-safe windowing, company-level splitting, calibration,
explainability. That applies unchanged to a real licensed dataset.

**Q: What would change if you used real data?**

Three things: (1) I'd need a fairness audit — real founder/company data
often correlates with protected attributes (gender, ethnicity via name/location
proxies) in ways my synthetic generator doesn't model at all, since it has no
such attributes to begin with. (2) I'd expect messier features — missing
values, inconsistent reporting cadence, restated financials — so the feature
engineering would need real missing-data handling, not just clean rolling
windows. (3) I'd validate cohort stability against actual macro events (2008,
2020, 2022 rate hikes) rather than a simulated shock.

**Q: How did you decide on a 6-month prediction horizon?**

It's a judgment call balancing actionability against signal strength. Too
short (1 month) and there's no time to act on the warning. Too long (24
months) and the signal-to-noise ratio drops — too much can happen. Six months
roughly matches a typical board-meeting-to-board-meeting or fundraise-runway
review cadence, which is also where this kind of tool would actually get used.
This is exposed as a named constant (`HORIZON_MONTHS`) specifically so it's a
parameter, not a buried assumption.

---

## 2. Leakage & Validation Methodology (expect the hardest questions here)

**Q: Walk me through how you prevented data leakage.**

Three separate mechanisms, layered:
1. **Feature leakage**: every engineered feature (rolling slope, volatility,
   stress counters) is computed using only data up to and including month
   *t*. I verified this with a unit test that mutates a *future* month's raw
   value and asserts the feature at an earlier *t* doesn't change.
2. **Label leakage**: the label looks strictly forward — `fail_month > t AND
   fail_month <= t + HORIZON`. Rows at or after the actual failure month are
   dropped entirely, so the model never trains on a "company is already dead"
   row.
3. **Split leakage**: I split by `company_id`, not row. A row-level random
   split would put month 14 and month 15 of the same company on opposite
   sides of the split, and the model would partially memorize that company's
   trajectory — producing an artificially inflated test score that wouldn't
   hold up on a genuinely new company.

**Q: Why PR-AUC instead of accuracy or ROC-AUC as your primary metric?**

With ~9.8% positive class prevalence, a model that predicts "healthy" for
every company scores ~90% accuracy while being completely useless. ROC-AUC is
better but can still look deceptively good under heavy imbalance because it's
driven partly by the (easy) true-negative rate. PR-AUC focuses specifically on
how well the model finds the positive class without flooding false positives,
which is the actual operational concern here.

**Q: Why isotonic regression for calibration, and why does it matter?**

Tree ensembles are good *rankers* — they reliably order companies from
lowest to highest risk — but their raw probability outputs aren't reliably
*calibrated*: a raw score of 0.8 doesn't necessarily mean "80% of companies
like this fail." Isotonic regression fits a monotonic mapping from raw score
to empirical frequency on a held-out validation set, so the dashboard's
displayed percentage is actually meaningful to a non-technical stakeholder,
not just a relative ranking.

**Q: Can you SHAP-explain the calibrated model directly?**

Not cleanly — `shap.TreeExplainer` needs direct access to tree structure,
and the calibrated model is a wrapper (isotonic regression on top of the raw
tree model), not itself a tree. I compute SHAP on the *raw* model and use the
*calibrated* model for the displayed score. I documented this tradeoff
explicitly in the model card rather than silently glossing over it — the SHAP
attribution is a faithful explanation of what drives the *ranking*, just not
a literal decomposition of the final calibrated number.

---

## 3. Modeling Choices

**Q: Why not just use the best-performing model and skip the comparison?**

Two reasons. First, the logistic regression baseline tells me how much lift
the complexity of tree ensembles is actually buying me — if LR scored close
to XGBoost, I'd seriously consider shipping the simpler, more interpretable
model. Second, in this run the gap was real (LR's PR-AUC was meaningfully
lower), which justifies the added complexity with a number instead of "trust
me, gradient boosting is better."

**Q: Why didn't you use SMOTE for the class imbalance?**

SMOTE interpolates synthetic minority-class rows between nearest neighbors in
feature space. On i.i.d. tabular data that's reasonable. On this panel data,
two "nearest" company-months might belong to two *completely different
companies* with unrelated underlying trajectories — interpolating between them
produces a row that doesn't correspond to anything that could actually happen
to a real company. I used `class_weight`/`scale_pos_weight` instead, which
reweights the loss function without fabricating fake data points.

**Q: How did you pick the classification threshold?**

Not the default 0.5. I modeled it as a cost-minimization problem: missing a
real failure (false negative) is assumed to cost roughly 5x more than a false
alarm (false positive) — a missed warning could mean an investor or operator
discovers a cash crisis too late, while a false alarm just costs a review
meeting. That 5:1 ratio is an explicit, tunable business assumption, not a
statistical fact, and I say so directly in the app and model card.

---

## 4. Deployment & Product Thinking

**Q: How would you actually deploy this for a real fund?**

Batch-score nightly or weekly via the same GitHub Actions retrain workflow
pattern, write results to a database, and have Power BI / the Streamlit app
read from that rather than re-running inference live. For very fresh signals
(e.g. company self-reports a metric today) you'd want a lightweight API
endpoint wrapping the model for on-demand scoring, but for portfolio
monitoring, batch is simpler, cheaper, and sufficient — funds don't need
millisecond latency on this kind of decision.

**Q: What's the biggest risk if someone used this in production today?**

Believing the absolute numbers apply to real companies. I built guardrails
against that — the model card, the in-app disclosure, the "not causal"
warnings on the what-if simulator — specifically because the failure mode
I'm most worried about isn't a bug in the code, it's someone screenshotting a
risk score out of context and treating it as gospel.

**Q: Why both Streamlit AND Power BI? Isn't that redundant?**

They serve different audiences and update cadences. Streamlit is for
interactive, single-company exploration — a founder or analyst poking at one
company's trajectory, running what-if scenarios. Power BI is for the
recurring portfolio-level executive view — a partner meeting reviewing 50
companies at a glance, refreshed on a schedule, embedded in a broader BI
ecosystem the firm already uses. Building both also let me show I can ship
the same underlying model through two completely different consumption
patterns, which is a real skill gap between "I trained a model" and "I shipped
something people can act on."

---

## 5. Honest Weaknesses (have these ready — interviewers respect this more than pretending there are none)

- The What-If Simulator changes one feature at a time without re-deriving
  correlated downstream features (raising runway doesn't automatically lower
  burn multiple). It's explicitly labeled a "sensitivity tool," not a causal
  simulator, but a more rigorous version would need a structural causal model
  of feature dependencies, not just the predictive model.
- The composite heuristic risk index is a hand-tuned baseline I built for
  sanity-checking the ML model, not itself validated — I'd say that plainly if
  asked whether it's "real."
- No fairness/bias audit, because the synthetic data has no protected
  attributes to audit. This is a real gap that would need to be closed before
  any real-world use, and I'd lead with that unprompted.
- The narrative generator (`_build_narrative`) is a simple templated mapping
  from top SHAP features to phrases — it's good enough for a portfolio demo,
  but a production version would need a much larger, curated feature-to-language
  dictionary, and probably some guardrails against contradictory phrasing
  when SHAP values are close to zero.
