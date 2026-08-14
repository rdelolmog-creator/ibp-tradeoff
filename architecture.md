# IBP Trade-Off Engine — Project Architecture

**Week 6 Capstone · AI Finance Accelerator · Mastery Level** Source-of-truth document. Loaded as Claude Project knowledge and version-controlled in the repo. *v3 — revised at the Step 6 MVD gate. See Revision history at the end.*

## 1. The problem

*"**Every function optimises its own leg of the inventory trade-off — service, plant cost, obsolescence, cash — and nobody can see the total. So we're systematically choosing wrong.**"*

Demand Planning owns forecast quality. Supply Chain Planning owns inventory policy and MOQ. The plants own conversion cost. Commercial owns service expectations. Each optimises locally; the economic total has no owner, and no system produces it.

**What the prototype does:** converts four decision levers into four economic consequences on one screen, traced to a named owner, so the trade-off becomes visible and arguable.

**What it does not do:** replace the planning system, fix master data, or optimise the portfolio. It consumes standard IBP outputs and sits alongside any planning system.

## 2. Model logic

### Inputs

| **Input** | **Note** |
| --- | --- |
| Actual demand history (36 months) | By SKU, monthly |
| **Forecast history, by version** | The plan *as it was made*. Without this the bias decomposition is impossible. |
| Promotions calendar | Needed to exclude promo periods from bias estimation |
| Margin by SKU | Price vs standard cost |
| Supply variability | Schedule adherence, yield |
| Capacity by line, cost structure, WACC |  |

### Levers (decision points, with owners)

| **Lever** | **Owner** |
| --- | --- |
| Service target by ABC class | Supply Chain / Demand Planning |
| Inventory policy (cover / safety stock) | Supply Chain Planning |
| Forecast bias correction | Demand Planning / Commercial |
| MOQ / minimum run length | Supply Chain Planning / Plant |

### Outputs

**Analytical output — the decomposition.** What arrives is a noisy demand signal and a forecast. The model *infers* the split; it is never given it:

| **Component** | **Controllable?** | **Traceable to** |
| --- | --- | --- |
| Irreducible demand volatility | No | Market — absorb via buffer |
| Chronic bias (signed, persistent) | **Yes** | Named planner role / category |
| Random forecast error | Partly | Process capability |

This is a **finding you present**, not a parameter you assume — which is what makes the accountability story credible.

**Economic outputs:**

| **Output** | **Valuation** | **Granularity** |
| --- | --- | --- |
| Lost sales | Standard gross margin × substitution factor | SKU |
| Excess / slow-moving risk | Standard cost | SKU |
| Plant cost | Changeovers, run length, absorption, overtime step | Plant / line |
| Working capital | Standard COGS; cash shown separately, **only WACC carrying cost enters the total** | SKU |

**Excess / slow-moving risk is two mechanisms, not one** *(v3)*. Both value at standard cost: **(a) shelf-life write-off**, when stock ages past `shelf_life_days × (1 − writeoff_trigger_pct_life_left)`; **(b) a slow-moving provision** on stock held beyond `inventory_risk.excess_cover_months_threshold` months of forward demand, at `inventory_risk.excess_provision_pct`.

The output name is wider than obsolescence, and a write-off-only implementation is *exactly zero* on any long-shelf-life category inside a 12-month horizon — on personal care, stock would have to sit ~32 months. Mechanism (b) is what makes the output live there; (a) is confirmed live on chilled. The two assumption keys are additions to the original set and belong on the limitations slide as such.

Cover for mechanism (b) is measured against **demand, never against the plan**. Measuring against the plan lets `forecast_bias_correction` reclassify identical physical stock as excess through a definitional channel — a circularity leak found and fixed at Step 6.

**Objective:** minimise total economic cost **subject to a service floor**. Unconstrained minimisation would abandon service on low-margin SKUs and is commercially naive.

## 3. Scope

**3 plants, 4 lines, ~60 SKUs across 3 categories** — roughly 15 SKUs per line.

| **Plant** | **Lines** | **System** | **Role** |
| --- | --- | --- | --- |
| Plant 1 | 2 | A | Main site, multi-category |
| Plant 2 | 1 | A | Second site, same ERP |
| Plant 3 | 1 | **B** | Small site, **recent acquisition not yet migrated** |

- **History: 36 months.** Needed to estimate seasonality (≥2 full cycles, ideally 3) and to establish that a planner's bias is *chronic* rather than one bad quarter.

- **Forward horizon: 12 months rolling.** Beyond that, SKU-level granularity is false precision.

- **Two source systems.** Plant 3's separate system is not an arbitrary defect injection — an unmigrated acquisition *necessarily* carries its own SKU coding, UoM convention, reporting currency and date format. The cause is stated, so the mess is inevitable rather than contrived.

- Small scope is a **deliberate pedagogical choice**: one line traceable end-to-end on screen beats 1,200 SKUs nobody can follow. State this in the pitch; do not sound apologetic.

## 4. Tool allocation

| **Tool** | **Role** |
| --- | --- |
| **ChatGPT custom GPT** | Mechanism *design* + drafts generate_data.py |
| **Claude Project chats** | All reasoning, all Python via SNAKE, troubleshooting |
| **Google Colab** | Development and validation of the pipeline |
| **Claude Skills** | Agent swarm, scenario reporting |
| **GitHub (public) + Streamlit Cloud (free)** | Interactive application |

### Why the GPT writes a script, not the data

The GPT produces a **seeded, parameterised Python generator**; you run it in Colab; the data comes out of Colab.

- **Reproducibility** — seeded code regenerates identical data. Change a parameter at step 8, re-run one line; don't re-prompt into a subtly different world that breaks everything downstream.

- **Internal consistency** — ~60 SKUs × 36 months × 3 plants with stock balances rolling forward, capacity binding, and volumes reconciling is arithmetic. Chat-generated tables drift and the balances won't tie.

- **Auditability** — the circularity risk (§9) is that the generator secretly encodes the answer. With code you can *show* it doesn't. With chat output you can only assert it.

**Use the GPT where it is genuinely better:** proposing which mechanisms matter, how defects manifest, realistic parameter ranges, and a small illustrative sample to sanity-check shape. Then bring the drafted script to Claude for review against the circularity risk *before* running it.

### Colab vs Streamlit — different environments

Colab is where **you** run code interactively, cell by cell. Streamlit Cloud is a server that runs app.py continuously and serves it as a web page.

- SNAKE prompt in the Project → Claude produces app.py

- Commit to GitHub alongside src/, config/, requirements.txt

- Streamlit Cloud (connected to the repo) installs dependencies and serves it

- Edit → commit → auto-redeploy

Because app.py imports the same src/ modules validated in Colab, the app is thin — mostly layout and sliders.

**requirements.txt**** matters here in a way it doesn't in Colab.** Colab pre-installs pandas and scikit-learn; Streamlit Cloud installs only what you list. A missing line is the most common deployment failure. Ask for it explicitly in the SNAKE prompt.

## 5. Build steps

| **#** | **Step** | **Tool** | **Aim** | **Output** |
| --- | --- | --- | --- | --- |
| 0 | Foundations | Claude chat *(no code)* | Fix assumptions, schema, and the scalability/PII/ROI answers **before** design | assumptions.yaml, schema.yaml, decision log |
| 1 | Generator design | ChatGPT GPT → Claude review | Mechanism-based synthetic data, seeded | generate_data.py |
| 2 | Raw data + defect injection | Colab | Two inconsistent source files | plant_system_A.csv, plant_system_B.csv |
| 3 | **Cleaning logic** | Claude chat *(reasoning only — no SNAKE)* | Rubric explicitly rewards reasoning before code | Numbered cleaning spec |
| 4 | Ingestion + cleaning | SNAKE → Colab | Consolidate, clean, validate | src/ingest.py, src/cleaner.py, clean_master.parquet, DQ report |
| 5 | Demand characterisation | SNAKE → Colab | **Infer** volatility vs chronic bias vs random error from demand + forecast history | src/demand.py |
| 6 | **MVD checkpoint** | SNAKE → Colab | **One line, all its SKUs (~15), full 12-month horizon** — four outputs responding to one lever | Working vertical slice: src/engine.py (MVD scope), tests/test_engine.py |
| 7 | Cost-response model | SNAKE → Colab | Learn cost response to schedule disruption; **XGBoost vs linear benchmark + SHAP** | src/cost_model.py, comparison table |
| 8 | Scenario engine | SNAKE → Colab | Deterministic causal core: levers → four outputs, **broadened** to all lines and all levers together | src/engine.py *(extended, not created — see step 6)* |
| 9 | Reporter | SNAKE → Colab | Aggregation, owner mapping, artefacts | src/reporter.py |
| 10 | Streamlit app | SNAKE → GitHub → Streamlit Cloud | Sliders, assumptions panel, four outputs | app.py deployed |
| 11 | Agent swarm | Claude Skill | Analyst / Skeptic / CFO stress-test | Decision memo |
| 12 | Optimiser *(only if time)* | SNAKE → Colab | Constrained optimisation | src/optimiser.py |
| 13 | Pitch | Claude Skill (pptx) | 5-minute deck + limitations | Deck |

### Why the MVD is a line, not a SKU

A single SKU is misleading: capacity is consumed collectively, and changeovers are a property of the **sequence on a line**, not of any one SKU. One line with all its SKUs is the smallest unit where capacity binds, changeovers accumulate and mix effects are real — while still being traceable on screen.

**Step 6 is the checkpoint.** If the vertical slice works, the project is safe. Everything after is elaboration. Build in dependency order, not importance order — the optimiser sits at 12 because it is the enhancement most likely to eat the last week.

### Why the MVD is a full horizon, not one month *(v3)*

A single month cannot show inventory build, stock ageing, write-off, or the cost of a minimum-run batch that covers several months of demand. Three of the four economic outputs would be structurally inert and the gate would demonstrate nothing. The engine therefore projects the next `horizon_months: 12` rather than replaying history under counterfactual settings — the two answer different questions ("what would last quarter have cost under different rules" vs "what should I do next quarter"), and the tool exists for the second.

Scope discipline is unaffected. The MVD is still **one line, one lever at a time**. The restriction that matters is breadth across lines, not depth in time.

### What the MVD does *not* test *(v3)*

In the forward simulation the MVD line runs at roughly 35% utilisation, so `min_run_hours` never reaches the overtime step within its slider range. The gate therefore demonstrates the **changeover and inventory** legs of the conversion-cost mechanism but **not the capacity/overtime leg**, which stays untested until step 8 runs lines with tighter utilisation.

This is the static, capacity-blind SKU-to-line allocation surfacing exactly where it was predicted to, and it is evidence for that review rather than a defect in the engine. Fixed absorption is also ~94% of conversion cost, so that output responds correctly but looks nearly inert in level terms — step 9 should report deltas against a baseline alongside levels.

## 6. Repo structure

ibp-tradeoff/

  src/        ingest.py  cleaner.py  demand.py

              cost_model.py  engine.py  reporter.py

  config/     assumptions.yaml  schema.yaml

  data/       raw/  clean/

  notebooks/  01_cleaning  02_demand  03_model  04_scenarios

  app.py  generate_data.py  requirements.txt  README.md

src/ is simply the conventional folder name for source code — the .py files holding the classes, kept separate from notebooks, data and config.

### Notebooks vs modules — the development compromise

Develop in **separate notebooks**, one per step; debugging a 400-cell monolith is miserable. What must be avoided is the *logic* living inside notebook cells — that produces five scripts requiring manual shuttling of outputs, not a pipeline.

The rule: notebooks stay separate for development, but each one's logic goes into a module in src/. Notebook 01 develops and tests DataCleaner, which then lives in src/cleaner.py. Notebook 04 imports all modules and runs the chain end-to-end. Same development ergonomics; stitching at the end is one cell rather than a rewrite. It is also what makes app.py thin.

## 7. SNAKE adapted for this project

SNAKE as taught produces *notebook code*. The rubric requires a *class-based reusable pipeline*. Naive SNAKE gives five excellent standalone notebooks and a failed criterion.

**Two adaptations:**

- **The A-clause must name the target module and class.** Not *"**write code that cleans the file**"* but *"**generate a class **DataCleaner** in **src/cleaner.py**, plus a notebook cell that imports and calls it.**"*

- **The K-clause at step 1 is the target schema**, not a data sample — no data exists yet. From step 4 onward, paste real df.head() output.

**Worked example — step 4:**

**S:** Two raw CSVs in Drive — plant_system_A.csv (Plants 1–2), plant_system_B.csv (Plant 3), from generate_data.py. Known defects: duplicate SKU codes across systems, mixed UoM (cases vs units), one plant in GBP not EUR, inconsistent date formats, ~4% nulls in volume. **N:** Google Colab, Drive mounted, repo at /content/ibp-tradeoff/. **A:** Generate src/ingest.py with class DataIngestor (config-driven paths, schema validation that fails loudly on unexpected columns) and src/cleaner.py with class DataCleaner implementing the spec below. Then a notebook cell that imports both, runs them, prints a before/after reconciliation (rows in, dropped by reason, rows out) plus a data-quality DataFrame, and writes data/clean/clean_master.parquet. **K:** [paste df.head() from both files] **E:** No hard-coded paths or SKU lists — read from config/assumptions.yaml. Must be importable by both notebook and app.py. Cleaning spec: [paste from step 3].

**Step 10 differs:** the N-clause becomes *"**this is **app.py** for Streamlit Cloud, deployed from a GitHub repo; imports from **src/**, reads **config/assumptions.yaml**.**"* No display(), no cell-by-cell state — widget callbacks and st. components only. Request requirements.txt in the same prompt.

## 8. Rubric mapping

| **Mastery criterion** | **Where satisfied** | **Evidence** |
| --- | --- | --- |
| Complex problem identified | §1 | Trade-off with no owner; one-sentence statement |
| Cleaning logic defined | Step 3 | Numbered spec written before any code |
| Working Python code | Step 4 | clean_master.parquet + reconciliation output |
| Reusable pipeline | Steps 4–9 | Classes in src/, config-driven, re-run demo |
| Model trade-off explained | Step 7 | XGBoost vs linear, SHAP, architecture argument |
| Interactive application | Step 10 | Streamlit with sliders + assumptions panel |
| Scalability addressed | Step 0 | Written before build, so it shapes design |

**The architecture argument** — say it explicitly; do not let the grader infer it:

*"**In a decision system the model is not the product — the decision is. ML is used only where a relationship must be learned and cannot be derived: the demand volatility/bias decomposition, and the plant cost response to schedule disruption. Everything else is deterministic finance, because a CFO will not act on a black box that tells them their working capital number. Interpretability was not a constraint I worked around; it determined the architecture.**"*

## 9. Risks and mitigations

| **Risk** | **Mitigation** |
| --- | --- |
| **Circular synthetic data** *(project-killer)* | Generator encodes **mechanisms only** — run length, changeovers, absorption, expiry, lead-time variance. Never a direct forecast-error→cost coefficient. Include confounders (line speed, equipment age, complexity, labour rates) and genuine noise. R² of 0.98 is a confession, not a result. **Enforced from step 6 by a running test**, not by the `guardrails:` block in assumptions.yaml, which is only a statement of intent — see below. |
| **Pipeline not genuinely reusable** | Classes in src/ from the start; hold back month 36 and re-run the full pipeline in one cell to *demonstrate* it. Schema validation that fails loudly. |
| **Assumption opacity** | Single assumptions.yaml; assumptions panel on the main app screen, not buried; provenance label on every output; substitution factor as a live slider; tornado chart of sensitivity. |
| **ML looks peripheral** | One real benchmarked comparison at step 7. State the architecture argument aloud. Name what was rejected and why. |
| **Clean DataFrame not visible** | Named df_clean, before/after reconciliation printed, DQ report as a DataFrame, exported artefact. |
| **Scope creep** | Frozen at four levers, four outputs. Anything new displaces something existing. |
| **Deliverable questions forgotten** | Answered at step 0, before building. |

### The circularity guardrail is a running test *(v3)*

`tests/test_engine.py::test_bias_lever_acts_only_through_quantities` sweeps `forecast_bias_correction` across its full range with everything else held fixed, then rebuilds every economic output from the simulated quantities at unit rates read *before* the sweep. If any cost can only be reproduced by reference to the lever value, the test fails and the model is circular.

Two structural facts support it, each separately tested:

- the lever enters the engine in exactly one function, `project_plan()`, and it moves the **plan**, never demand;
- the demand path is asserted bit-identical across every lever setting.

This guardrail earned its place immediately: it caught a genuine leak where the slow-moving threshold was measured against the plan, letting the bias lever reclassify identical physical stock as excess. Worth saying aloud in the pitch — the guardrail is not decorative.

## 10. Open assumptions to challenge

- **Substitution factor** — not every stockout is a lost sale. Shortfall gets substituted within the portfolio, backordered, or genuinely lost, and the mix varies by category and channel. Valuing 100% of shortfall at full standard GM overstates the cost of low service and biases the model toward too much inventory — the exact error the tool exists to correct. Make it a slider.

- **Magnitudes are not findings.** With synthetic data, any number reported is whatever the generator encoded. The *method* is the deliverable; state this on the limitations slide.

- **Bias direction matters.** Under-forecasting → unplanned insertions, extra changeovers, overtime, premium freight. Over-forecasting → under-absorption, idle time, obsolescence risk. Both raise cost per unit through different mechanisms; treat signed bias and absolute error as separate features.

## 11. Final deliverable answers *(draft at step 0)*

- **Scalability:** what breaks first is per-SKU scenario computation, not model training. Fixes: vectorisation, caching, parallel scenario evaluation.

- **PII:** fully synthetic data, no personal information. Accountability assigned at **role level, not named individuals** — state this explicitly, since the framework assigns ownership.

- **ROI:** analyst hours per S&OP cycle × cycles per year. State the assumption openly rather than quoting an unsupportable number.

---

## Revision history

**v3 — Step 6 MVD gate.** Five changes, all arising from building and gating the vertical slice. Each carries a decision-log reference.

| # | Section | Change | Ref |
|---|---|---|---|
| 1 | §5, step 6 | MVD horizon changed from **one month to the full 12** — a single month leaves three of four economic outputs structurally inert and the gate proves nothing. Scope discipline is one line and one lever at a time, not one month. | D-045 |
| 2 | §2, Outputs | *Excess / slow-moving risk* specified as **two mechanisms** — shelf-life write-off plus a slow-moving provision — because write-off alone is exactly zero on a long-shelf-life category inside the horizon. Adds `inventory_risk.excess_cover_months_threshold` and `excess_provision_pct` to the assumption set. Cover measured against demand, never plan. | D-047, D-048 |
| 3 | §5, steps 6 and 8 | `src/engine.py` is **created at step 6** scoped to one line and **broadened at step 8**, rather than written twice. Step 8 widens the same class; it does not replace it. | D-046 |
| 4 | §9 | Circularity guardrail restated as an **enforced running test**, not a YAML comment. Added the mechanism and what it caught. | — |
| 5 | §5 | Added what the MVD **does not** test: the capacity/overtime leg, because the MVD line runs at ~35% utilisation. Plus the fixed-absorption reporting problem. | O-11, O-12 |

**v2** — revised after design review.

**v1** — initial architecture.

*This document is the source of truth. Where a later decision contradicts it, the document is amended here on the record rather than quietly overridden — otherwise the submission ships a source-of-truth document its own code disagrees with. The decision log carries the reasoning; this carries the current state.*
