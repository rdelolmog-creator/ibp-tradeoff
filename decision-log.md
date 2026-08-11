Decision Log — IBP Trade-Off Engine
One line per decision, in the form DECISION: X, because Y. Feeds the limitations slide and the pitch. Append as we go; never rewrite history — if a decision is reversed, add a new entry that supersedes it.


Step 0 — Foundations
D-001 · Schema layering DECISION: schema.yaml defines two layers — raw.* (as extracted, defects included) and canonical.* (post-clean contract) — because the cleaner's job is precisely the mapping between them, and writing it down makes the cleaning spec testable rather than narrative.

D-002 · Raw file count DECISION: the generator emits three raw CSVs — plant_system_A.csv, plant_system_B.csv, sku_master.csv — rather than the two named in §5, because margin and standard cost are SKU-level data, and holding 60 SKUs of master data in assumptions.yaml would breach the project's own no-hard-coded-SKU-list rule. Supersedes the §5 Step 2 output list.

D-003 · Clean master grain DECISION: clean_master is (sku_id, month) with forecast lags pivoted to columns (forecast_l1/l2/l3), because at ~2,160 rows long format triples table size for no analytical gain, and the bias decomposition reads lags side by side. Known cost: adding a fourth lag later is a schema change, not a data change.

D-004 · Imputation provenance DECISION: every imputed value carries a boolean provenance flag (actual_imputed), because ~4% of volume arrives null, and unflagged imputation would let generator noise be attributed to a named planner role — silently breaking the accountability claim the tool exists to make.

D-005 · Scalability answer (§11, revised after testing — see D-021) DECISION: the original Step 0 claim ("scenario computation breaks first, not model training") was an untested guess. Tested by doubling schema.meta.scope.skus to 120 with unchanged lines/plants: src/ingest.py and src/cleaner.py scale linearly with zero code changes (0.13s, 4,318 rows, same ~99% null-recovery rate, roll-forward still ties to the same tolerance) — the cleaning pipeline genuinely doesn't care about volume. What breaks first is upstream of cleaning: generate_data.py's fixed SKU-to-line allocation pushes L3 utilisation to 1.149 (over 100% of physical capacity) because SKU count doubled while line count did not. The real scalability constraint is physical capacity and portfolio-to-line allocation, not compute cost.

D-006 · PII answer (§11) DECISION: data is fully synthetic and accountability is assigned at role level, not to named individuals, because the framework's output is an attribution of chronic bias — and attributing that to a person rather than a role turns a diagnostic tool into a performance-management weapon.

D-007 · ROI answer (§11) DECISION: ROI is stated as analyst hours per S&OP cycle × cycles per year, with the hours assumption stated openly, because any single headline number here would be unsupportable and inviting challenge on it costs more credibility than it buys.

D-008 · Category differentiation DECISION: the three categories are ambient, chilled and personal care, with deliberately different shelf life (65 / 6 / 156 weeks) and margin (33% / 29% / 52%), because homogeneous categories would leave the obsolescence leg of the trade-off inert and produce a four-output model where one output never moves.

D-009 · Chronic bias structure DECISION: chronic forecast bias is drawn once at category level and held across all 36 months, with SKU-level idiosyncratic variation around it, because bias redrawn monthly is random error wearing a sign — the decomposition would correctly find nothing to attribute and the accountability claim would collapse.

D-010 · Bias control group DECISION: ambient carries zero chronic bias while chilled carries −7% and personal care +5%, because a control category demonstrates the model discriminates rather than manufacturing an attribution everywhere it looks.

D-011 · MAPE as validation, not input DECISION: forecast MAPE is a validation target rather than a generator parameter, because MAPE is a consequence of chronic bias plus random error, and specifying all three independently over-determines the system and silently distorts one of them.

D-012 · Capacity calibration DECISION: baseline line utilisation is 85% and line speeds are taken at the bottom of the proposed EMEA ranges, because capacity must bind for the MOQ and minimum-run levers to move any output, and the parameter table's central values implied a €700m business against a €220k/line absorption base — conversion cost would have landed near 2% of COGS against a 12–25% norm.


Closed — rulings made
Ref
Question
Ruling
O-01
Surrogate key vs. remapping table
Surrogate key {system}-{code}. Simpler, and collisions are resolved once at ingest.
O-02
Confounders: data or config?
line_master (data). Equipment age, complexity and line speed must be learnable features for the cost model, not fixed config.
O-03
Economic parameter set
Ranges sourced from a GPT prompted for EMEA FMCG plausibility, corrected here, recorded in assumptions.yaml with confidence labels carried through to the limitations slide. No parameter is claimed as fact.



Step 1 — Generator design
Generator reviewed against the §9 circularity risk before execution. Forecasts are built from the deterministic demand signal only — never from realised demand — and chronic bias is drawn once per SKU and held across all 36 months. No coefficient anywhere maps forecast error to cost or to write-off. Reviewed, corrected, then run: net sales €196.9m vs €200m target, conversion cost 12.7% of COGS, stock roll-forward ties to zero, MAPE inside range for all three ABC classes at lag 1.

D-013 · Capacity binds with a shortfall signal DECISION: overtime is capped at plant_economics.max_overtime_hours_month and any residual excess scales scheduled runs down proportionally, recording capacity_shortfall_units, because unbounded overtime meant capacity never physically bound and the minimum-run lever produced only a cost signal, never a shortage.

D-014 · Exported actuals are censored DECISION: raw files export shipped volume censored at available stock, and latent demand exists only in _truth/, because a planning system records what shipped, not what was wanted — with the known consequence that chronic bias is under-recovered on stocked-out SKUs.

D-015 · Bias must be estimated in log space DECISION: Step 5 estimates chronic bias as the mean of log(forecast / actual), not (forecast − actual) / actual, because demand noise CV exceeds forecast noise CV in every ABC class and the naive ratio shifts the estimate upward by (σd² − σf²) / 2 — measured at +1.8pp, enough to report +4.5% chronic bias in the control category that was encoded with zero.

D-016 · Ground truth carries raw-layer keys DECISION: ground_truth.csv carries source_system, source_sku_code and surrogate_sku_id alongside the internal sku_id, because the internal id appears in no raw file by design, and a truth file that cannot be joined cannot score the decomposition it exists to validate. The raw CSVs remain free of the internal id.

D-017 · Two datasets, not one tuned dataset DECISION: bias recovery is validated against two generated datasets — the primary at ~6.2% censoring and a control at ~1.6% — rather than tuning assumptions.yaml until recovery looks clean, because parameter selection until the answer appears is the circularity failure in another costume (§9). The control varies abc.*.target_cover_weeks only (12/12/14 vs 4/4/6); demand_noise_cv was deliberately left untouched because it drives both censoring and the variance asymmetry, and changing it would confound the two effects the control exists to separate.

D-018 · Constant-column drift check added DECISION: DataIngestor._validate_columns now asserts that any raw column the schema declares constant (e.g. system_b.ccy = GBP) actually holds only that value, because a value change that still parses as a valid string (GBP → USD) previously passed C-02 silently — the FX/UoM logic branches on the declared constant, not on what the column contains, so drift would have applied the wrong rate without raising anything. Verified: real data still passes; a simulated ccy drift is caught and raises SchemaViolation.
Open at Step 1 close
Ref
Question
Status
O-04
Realised line utilisation is 0.678 vs a 0.85 target. Merged into O-10.
Superseded — see O-10.
O-05
Which censored estimator at Step 5 — Tobit-style, or reconstruct latent demand from stockout periods?
Ruled: Tobit-style censored regression, to preserve the circularity boundary — reconstructing latent demand via the forecast risks the forecast leaking into the demand estimate.



Step 2 — Raw data + defect injection
Ran via generate_data.py against both assumptions.yaml (primary) and assumptions_lowcensoring.yaml (control). Validation passed on both: net sales €196.9m / €199.2m vs €200m target, conversion cost 12.7% / 12.4% of COGS, roll-forward ties to zero, MAPE inside range on all three ABC classes. Censoring confirmed at 6.16% (primary) vs 1.57% (control), isolating the intended variable.


Step 3 — Cleaning specification
cleaning-spec.md written before any cleaning code, per the rubric's reasoning-before-code criterion. 13 numbered steps (C-01 to C-13), dependency-ordered, each with a stated failure condition. Verified against real data: identifier dtype coercion drops the ground-truth join from 60/60 to 42/60 if not forced to string; null volumes are 100% exactly recoverable via the stock roll-forward identity on all 1,511 checkable rows.


Step 4 — Ingestion + cleaning
src/ingest.py (DataIngestor) and src/cleaner.py (DataCleaner) implement the spec. Run identically against both datasets with zero code changes — the reusable-pipeline evidence: 2,159 rows out, 60 SKUs, 86 of 87 nulls recovered exactly, 1 row dropped (irrecoverable first-month null), roll-forward max error 0.000007 units, both datasets.

D-019 · Human-review workbook is a pipeline output, not a reconstruction DECISION: Step4_Data_Quality_Review.xlsx is built by src/report_step4.py, called from inside the notebook immediately after DataCleaner.clean(), from the live df_clean/master/dq_report/dropped objects of that run — not rebuilt afterward from pasted console output — because the review artefact must be evidence of what the pipeline actually did, and a human sign-off (Approved / Approved with comments / Rejected) is required on this workbook before Step 5 proceeds.

D-020 · Regression and robustness test suite added DECISION: tests/test_pipeline.py (14 tests) locks in the current correct output (row counts, recovery counts, roll-forward tolerance) as regression tests, and separately proves the pipeline fails loudly rather than silently on defect types the spec never anticipated (new column, missing column, constant-value drift, unparseable date, out-of-range rate, null-rate spike, invalid case size, unknown category) — because a pipeline claimed as reusable needs a runnable yes/no answer to "does this still work," not a manual re-read of console output each cycle.

D-021 · Scalability stress-tested, not asserted DECISION: ran the actual pipeline at 2× SKU count (120) and 37-month history (up from 36) as concrete stress tests, rather than answering the §11 scalability question from first principles, because a Mastery-level claim about what breaks under volume should be demonstrated against real code, not guessed at Step 0 before any code existed. Findings: (1) src/ingest.py/src/cleaner.py scale linearly with no code change — this is genuinely a non-issue at any realistic scope for this project; (2) doubling SKUs without adding plants/lines overloads line capacity (L3 hit 114.9% utilisation) because build_portfolio() splits SKUs evenly across whatever lines are configured, with no capacity-aware allocation; (3) a longer history window (37 months) works unchanged because schema.meta.history_months is read from config, not hard-coded — confirming the full-re-extraction pattern in D-022 requires no code change for a growing window. Supersedes the untested D-005 claim.

D-022 · O-08 resolved: full re-extraction, not incremental append DECISION: the pipeline assumes full rolling-window re-extraction each IBP cycle (source systems re-queried for the entire window, not just the new month), not incremental append, because at this scale (2,160-4,318 rows tested) the efficiency gain from incremental is negligible while the correctness cost is real. Tested directly: simulating an incremental load (only the newest month passed to DataCleaner) breaks _derive_stock_open() — with no prior context, shift(1) returns null for every row in the new month, and C-11 would wrongly drop the entire month as irrecoverable. Full re-extraction also re-validates already-clean months every cycle, which incremental cannot do, so a source-system restatement of a prior period is caught rather than silently missed. schema.meta.history_months already reads from config, so a growing or shifting window requires no code change under this model — verified in D-021.

D-023 · Recovery correctness fixed for multi-field nulls DECISION: a row is only counted as actual_imputed = True if the roll-forward formula's result is itself non-null, not merely if stock_open is present, because the prior check let a row with a second missing input (e.g. stock_close also null the same month) be flagged as successfully recovered before being caught and dropped a few lines later — overstating nulls_recovered_by_identity in the DQ report even though no bad data reached clean_master. Verified with a double-null test case: recovered count stays at 86, the row is correctly dropped instead.

D-024 · Cascade failures classified separately from genuine first-month gaps DECISION: rows dropped at C-11 are split by calendar position — a genuine first month of a SKU's history vs. a later month whose stock_open is null because a prior month's stock_close was null and the gap cascaded forward via the roll-forward shift — because both cases share stock_open_units.isna() and an earlier version of this classification used exactly that check, silently mislabeling a real upstream data problem as an expected edge case. Fixed to use calendar position (is this the SKU's minimum month in the dataset) instead. Every dropped row's sku_id and month is now recorded individually (not just an aggregate count), and the Step 4 review workbook (sheet 7) flags cascade rows for investigation, separately from expected edge cases.

D-025 · FX rate made an explicit, auditable field DECISION: fx_rate_applied is added as a column on the cleaned sku_master output (the applied gbp_eur rate for System B rows, null for System A), because the review workbook could previously show the converted price but not the rate that produced it — a reviewer had to cross-reference assumptions.yaml by hand to verify the FX math. Now shown directly alongside the converted values on sheet 5.

D-026 · Review workbook rebuilt to show full detail, not samples DECISION: the Step 4 review workbook was rebuilt from 6 tabs sampling 2 SKUs to 9 tabs showing every row that was actually touched by cleaning — the full clean dataset (all 2,159 rows), every one of the 86 null recoveries with the roll-forward arithmetic shown, all 540 System B conversion rows, and all 60 SKU master rows — because a reviewer cannot approve what they cannot see, and 2 sample SKUs is a demonstration, not a review artefact.
Open at Step 4 close
Ref
Question
Status
O-06
sched_adherence and yield_rate are constant within source/category — a zero-variance feature teaches the Step 7 cost model nothing.
Open, revisit at Step 7 if the model needs it.
O-07
Opening inventory value exists in generator internal state but is not exported — none of the four economic outputs currently need it.
Open, not built. Revisit only if Step 8 reporting needs an average-position metric.
O-09
Doubling SKU count without adding lines overloads capacity. Merged into O-10.
Superseded — see O-10.


O-10 · SKU-to-line allocation is static and capacity-blind (merges O-04, O-09, and the multi-line question raised at Step 4 review)

Three symptoms, one root cause: build_portfolio() assigns each SKU to exactly one line, split evenly by count, with no regard for capacity, line speed, or cost. This produces: (a) realised utilisation of 0.678 vs an 0.85 target — the even split ignores real line-speed differences; (b) L3 hitting 114.9% utilisation when SKU count is doubled without adding lines (D-021); (c) no mechanism for a single SKU to be produced across two lines/plants, which becomes more likely exactly when simulating category growth within ambient/chilled/personal_care.

Building a fix now (dynamic, capacity-aware allocation, and/or multi-line SKU assignment) would change clean_master's grain from (sku_id, month) to (sku_id, line_id, month) — reopening Steps 1-4, which are signed off, for a problem the Step 6 MVD gate cannot even test (the MVD is deliberately scoped to one line in isolation, per §5, so it will never surface cross-line capacity pressure either way).

Decision: deferred, not skipped. Revisit immediately after Step 6, once the engine runs across all four lines and all four levers together — that is the first point real evidence exists (does any line breach capacity under realistic lever settings, or does the current static allocation hold up fine in practice). Committed: this gets a dedicated, unhurried review pass at that point, not a quick call folded into the next task.


Step 5 — Demand characterisation
(pending)
