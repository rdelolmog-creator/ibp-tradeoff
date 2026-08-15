# Decision Log — IBP Trade-Off Engine

One line per decision, in the form `DECISION: X, because Y`.
Feeds the limitations slide and the pitch. Append as we go; never rewrite history —
if a decision is reversed, add a new entry that supersedes it.

---

## Step 0 — Foundations

**D-001 · Schema layering**
DECISION: `schema.yaml` defines two layers — `raw.*` (as extracted, defects included) and `canonical.*` (post-clean contract) — because the cleaner's job is precisely the mapping between them, and writing it down makes the cleaning spec testable rather than narrative.

**D-002 · Raw file count**
DECISION: the generator emits three raw CSVs — `plant_system_A.csv`, `plant_system_B.csv`, `sku_master.csv` — rather than the two named in §5, because margin and standard cost are SKU-level data, and holding 60 SKUs of master data in `assumptions.yaml` would breach the project's own no-hard-coded-SKU-list rule.
*Supersedes the §5 Step 2 output list.*

**D-003 · Clean master grain**
DECISION: `clean_master` is `(sku_id, month)` with forecast lags pivoted to columns (`forecast_l1/l2/l3`), because at ~2,160 rows long format triples table size for no analytical gain, and the bias decomposition reads lags side by side.
*Known cost: adding a fourth lag later is a schema change, not a data change.*

**D-004 · Imputation provenance**
DECISION: every imputed value carries a boolean provenance flag (`actual_imputed`), because ~4% of volume arrives null, and unflagged imputation would let generator noise be attributed to a named planner role — silently breaking the accountability claim the tool exists to make.

**D-005 · Scalability answer** *(§11, revised after testing — see D-021)*
DECISION: the original Step 0 claim ("scenario computation breaks first, not model training") was an untested guess. Tested by doubling `schema.meta.scope.skus` to 120 with unchanged lines/plants: `src/ingest.py` and `src/cleaner.py` scale linearly with zero code changes (0.13s, 4,318 rows, same ~99% null-recovery rate, roll-forward still ties to the same tolerance) — the cleaning pipeline genuinely doesn't care about volume. What breaks first is upstream of cleaning: `generate_data.py`'s fixed SKU-to-line allocation pushes L3 utilisation to 1.149 (over 100% of physical capacity) because SKU count doubled while line count did not. The real scalability constraint is physical capacity and portfolio-to-line allocation, not compute cost.

**D-006 · PII answer** *(§11)*
DECISION: data is fully synthetic and accountability is assigned at **role level, not to named individuals**, because the framework's output is an attribution of chronic bias — and attributing that to a person rather than a role turns a diagnostic tool into a performance-management weapon.

**D-007 · ROI answer** *(§11)*
DECISION: ROI is stated as analyst hours per S&OP cycle × cycles per year, with the hours assumption stated openly, because any single headline number here would be unsupportable and inviting challenge on it costs more credibility than it buys.

**D-008 · Category differentiation**
DECISION: the three categories are ambient, chilled and personal care, with deliberately different shelf life (65 / 6 / 156 weeks) and margin (33% / 29% / 52%), because homogeneous categories would leave the obsolescence leg of the trade-off inert and produce a four-output model where one output never moves.

**D-009 · Chronic bias structure**
DECISION: chronic forecast bias is drawn once at category level and held across all 36 months, with SKU-level idiosyncratic variation around it, because bias redrawn monthly is random error wearing a sign — the decomposition would correctly find nothing to attribute and the accountability claim would collapse.

**D-010 · Bias control group**
DECISION: ambient carries zero chronic bias while chilled carries −7% and personal care +5%, because a control category demonstrates the model discriminates rather than manufacturing an attribution everywhere it looks.

**D-011 · MAPE as validation, not input**
DECISION: forecast MAPE is a validation target rather than a generator parameter, because MAPE is a consequence of chronic bias plus random error, and specifying all three independently over-determines the system and silently distorts one of them.

**D-012 · Capacity calibration**
DECISION: baseline line utilisation is 85% and line speeds are taken at the bottom of the proposed EMEA ranges, because capacity must bind for the MOQ and minimum-run levers to move any output, and the parameter table's central values implied a €700m business against a €220k/line absorption base — conversion cost would have landed near 2% of COGS against a 12–25% norm.

---

## Closed — rulings made

| Ref | Question | Ruling |
| --- | --- | --- |
| O-01 | Surrogate key vs. remapping table | **Surrogate key** `{system}-{code}`. Simpler, and collisions are resolved once at ingest. |
| O-02 | Confounders: data or config? | **`line_master` (data).** Equipment age, complexity and line speed must be learnable features for the cost model, not fixed config. |
| O-03 | Economic parameter set | **Ranges sourced from a GPT prompted for EMEA FMCG plausibility, corrected here, recorded in `assumptions.yaml` with confidence labels carried through to the limitations slide.** No parameter is claimed as fact. |

---

## Step 1 — Generator design

*Generator reviewed against the §9 circularity risk before execution. Forecasts are built from the deterministic demand signal only — never from realised demand — and chronic bias is drawn once per SKU and held across all 36 months. No coefficient anywhere maps forecast error to cost or to write-off. Reviewed, corrected, then run: net sales €196.9m vs €200m target, conversion cost 12.7% of COGS, stock roll-forward ties to zero, MAPE inside range for all three ABC classes at lag 1.*

**D-013 · Capacity binds with a shortfall signal**
DECISION: overtime is capped at `plant_economics.max_overtime_hours_month` and any residual excess scales scheduled runs down proportionally, recording `capacity_shortfall_units`, because unbounded overtime meant capacity never physically bound and the minimum-run lever produced only a cost signal, never a shortage.

**D-014 · Exported actuals are censored**
DECISION: raw files export shipped volume censored at available stock, and latent demand exists only in `_truth/`, because a planning system records what shipped, not what was wanted — with the known consequence that chronic bias is under-recovered on stocked-out SKUs.

**D-015 · Bias must be estimated in log space**
DECISION: Step 5 estimates chronic bias as the mean of `log(forecast / actual)`, not `(forecast − actual) / actual`, because demand noise CV exceeds forecast noise CV in every ABC class and the naive ratio shifts the estimate upward by `(σd² − σf²) / 2` — measured at +1.8pp, enough to report +4.5% chronic bias in the control category that was encoded with zero.

**D-016 · Ground truth carries raw-layer keys**
DECISION: `ground_truth.csv` carries `source_system`, `source_sku_code` and `surrogate_sku_id` alongside the internal `sku_id`, because the internal id appears in no raw file by design, and a truth file that cannot be joined cannot score the decomposition it exists to validate. The raw CSVs remain free of the internal id.

**D-017 · Two datasets, not one tuned dataset**
DECISION: bias recovery is validated against two generated datasets — the primary at ~6.2% censoring and a control at ~1.6% — rather than tuning `assumptions.yaml` until recovery looks clean, because parameter selection until the answer appears is the circularity failure in another costume (§9). The control varies `abc.*.target_cover_weeks` only (12/12/14 vs 4/4/6); `demand_noise_cv` was deliberately left untouched because it drives both censoring and the variance asymmetry, and changing it would confound the two effects the control exists to separate.

**D-018 · Constant-column drift check added**
DECISION: `DataIngestor._validate_columns` now asserts that any raw column the schema declares `constant` (e.g. `system_b.ccy = GBP`) actually holds only that value, because a value change that still parses as a valid string (GBP → USD) previously passed C-02 silently — the FX/UoM logic branches on the declared constant, not on what the column contains, so drift would have applied the wrong rate without raising anything. Verified: real data still passes; a simulated `ccy` drift is caught and raises `SchemaViolation`.

### Open at Step 1 close

| Ref | Question | Status |
| --- | --- | --- |
| O-04 | ~~Realised line utilisation is 0.678 vs a 0.85 target.~~ **Merged into O-10.** | Superseded — see O-10. |
| O-05 | Which censored estimator at Step 5 — Tobit-style, or reconstruct latent demand from stockout periods? | Ruled: Tobit-style censored regression, to preserve the circularity boundary — reconstructing latent demand via the forecast risks the forecast leaking into the demand estimate. |

---

## Step 2 — Raw data + defect injection

Ran via `generate_data.py` against both `assumptions.yaml` (primary) and `assumptions_lowcensoring.yaml` (control). Validation passed on both: net sales €196.9m / €199.2m vs €200m target, conversion cost 12.7% / 12.4% of COGS, roll-forward ties to zero, MAPE inside range on all three ABC classes. Censoring confirmed at 6.16% (primary) vs 1.57% (control), isolating the intended variable.

---

## Step 3 — Cleaning specification

`cleaning-spec.md` written before any cleaning code, per the rubric's reasoning-before-code criterion. 13 numbered steps (C-01 to C-13), dependency-ordered, each with a stated failure condition. Verified against real data: identifier dtype coercion drops the ground-truth join from 60/60 to 42/60 if not forced to string; null volumes are 100% exactly recoverable via the stock roll-forward identity on all 1,511 checkable rows.

---

## Step 4 — Ingestion + cleaning

`src/ingest.py` (`DataIngestor`) and `src/cleaner.py` (`DataCleaner`) implement the spec. Run identically against both datasets with zero code changes — the reusable-pipeline evidence: 2,159 rows out, 60 SKUs, 86 of 87 nulls recovered exactly, 1 row dropped (irrecoverable first-month null), roll-forward max error 0.000007 units, both datasets.

**D-019 · Human-review workbook is a pipeline output, not a reconstruction**
DECISION: `Step4_Data_Quality_Review.xlsx` is built by `src/report_step4.py`, called from inside the notebook immediately after `DataCleaner.clean()`, from the live `df_clean`/`master`/`dq_report`/`dropped` objects of that run — not rebuilt afterward from pasted console output — because the review artefact must be evidence of what the pipeline actually did, and a human sign-off (Approved / Approved with comments / Rejected) is required on this workbook before Step 5 proceeds.

**D-020 · Regression and robustness test suite added**
DECISION: `tests/test_pipeline.py` (14 tests) locks in the current correct output (row counts, recovery counts, roll-forward tolerance) as regression tests, and separately proves the pipeline fails loudly rather than silently on defect types the spec never anticipated (new column, missing column, constant-value drift, unparseable date, out-of-range rate, null-rate spike, invalid case size, unknown category) — because a pipeline claimed as reusable needs a runnable yes/no answer to "does this still work," not a manual re-read of console output each cycle.

**D-021 · Scalability stress-tested, not asserted**
DECISION: ran the actual pipeline at 2× SKU count (120) and 37-month history (up from 36) as concrete stress tests, rather than answering the §11 scalability question from first principles, because a Mastery-level claim about what breaks under volume should be demonstrated against real code, not guessed at Step 0 before any code existed. Findings: (1) `src/ingest.py`/`src/cleaner.py` scale linearly with no code change — this is genuinely a non-issue at any realistic scope for this project; (2) doubling SKUs without adding plants/lines overloads line capacity (L3 hit 114.9% utilisation) because `build_portfolio()` splits SKUs evenly across whatever lines are configured, with no capacity-aware allocation; (3) a longer history window (37 months) works unchanged because `schema.meta.history_months` is read from config, not hard-coded — confirming the full-re-extraction pattern in D-022 requires no code change for a growing window. Supersedes the untested D-005 claim.

**D-022 · O-08 resolved: full re-extraction, not incremental append**
DECISION: the pipeline assumes full rolling-window re-extraction each IBP cycle (source systems re-queried for the entire window, not just the new month), not incremental append, because at this scale (2,160-4,318 rows tested) the efficiency gain from incremental is negligible while the correctness cost is real. Tested directly: simulating an incremental load (only the newest month passed to `DataCleaner`) breaks `_derive_stock_open()` — with no prior context, `shift(1)` returns null for every row in the new month, and C-11 would wrongly drop the entire month as irrecoverable. Full re-extraction also re-validates already-clean months every cycle, which incremental cannot do, so a source-system restatement of a prior period is caught rather than silently missed. `schema.meta.history_months` already reads from config, so a growing or shifting window requires no code change under this model — verified in D-021.

**D-023 · Recovery correctness fixed for multi-field nulls**
DECISION: a row is only counted as `actual_imputed = True` if the roll-forward formula's result is itself non-null, not merely if `stock_open` is present, because the prior check let a row with a second missing input (e.g. `stock_close` also null the same month) be flagged as successfully recovered before being caught and dropped a few lines later — overstating `nulls_recovered_by_identity` in the DQ report even though no bad data reached `clean_master`. Verified with a double-null test case: recovered count stays at 86, the row is correctly dropped instead.

**D-024 · Cascade failures classified separately from genuine first-month gaps**
DECISION: rows dropped at C-11 are split by calendar position — a genuine first month of a SKU's history vs. a later month whose `stock_open` is null because a prior month's `stock_close` was null and the gap cascaded forward via the roll-forward shift — because both cases share `stock_open_units.isna()` and an earlier version of this classification used exactly that check, silently mislabeling a real upstream data problem as an expected edge case. Fixed to use calendar position (is this the SKU's minimum month in the dataset) instead. Every dropped row's `sku_id` and `month` is now recorded individually (not just an aggregate count), and the Step 4 review workbook (sheet 7) flags cascade rows for investigation, separately from expected edge cases.

**D-025 · FX rate made an explicit, auditable field**
DECISION: `fx_rate_applied` is added as a column on the cleaned `sku_master` output (the applied `gbp_eur` rate for System B rows, null for System A), because the review workbook could previously show the *converted* price but not the *rate* that produced it — a reviewer had to cross-reference `assumptions.yaml` by hand to verify the FX math. Now shown directly alongside the converted values on sheet 5.

**D-026 · Review workbook rebuilt to show full detail, not samples**
DECISION: the Step 4 review workbook was rebuilt from 6 tabs sampling 2 SKUs to 9 tabs showing every row that was actually touched by cleaning — the full clean dataset (all 2,159 rows), every one of the 86 null recoveries with the roll-forward arithmetic shown, all 540 System B conversion rows, and all 60 SKU master rows — because a reviewer cannot approve what they cannot see, and 2 sample SKUs is a demonstration, not a review artefact.

### Open at Step 4 close

| Ref | Question | Status |
| --- | --- | --- |
| O-06 | `sched_adherence` and `yield_rate` are constant within source/category — a zero-variance feature teaches the Step 7 cost model nothing. | Open, revisit at Step 7 if the model needs it. |
| O-07 | Opening inventory value exists in generator internal state but is not exported — none of the four economic outputs currently need it. | Open, not built. Revisit only if Step 8 reporting needs an average-position metric. |
| O-09 | ~~Doubling SKU count without adding lines overloads capacity.~~ **Merged into O-10.** | Superseded — see O-10. |

**O-10 · SKU-to-line allocation is static and capacity-blind (merges O-04, O-09, and the multi-line question raised at Step 4 review) — CLOSED, see D-070**

Three symptoms, one root cause: `build_portfolio()` assigns each SKU to exactly one line, split evenly by count, with no regard for capacity, line speed, or cost. This produces: (a) realised utilisation of 0.678 vs an 0.85 target — the even split ignores real line-speed differences; (b) L3 hitting 114.9% utilisation when SKU count is doubled without adding lines (D-021); (c) no mechanism for a single SKU to be produced across two lines/plants, which becomes more likely exactly when simulating category growth within ambient/chilled/personal_care.

Building a fix now (dynamic, capacity-aware allocation, and/or multi-line SKU assignment) would change `clean_master`'s grain from `(sku_id, month)` to `(sku_id, line_id, month)` — reopening Steps 1-4, which are signed off, for a problem the Step 6 MVD gate cannot even test (the MVD is deliberately scoped to one line in isolation, per §5, so it will never surface cross-line capacity pressure either way).

**Decision: deferred, not skipped.** Revisit immediately after Step 6, once the engine runs across all four lines and all four levers together — that is the first point real evidence exists (does any line breach capacity under realistic lever settings, or does the current static allocation hold up fine in practice). Committed: this gets a dedicated, unhurried review pass at that point, not a quick call folded into the next task.

**D-027 · Step 4 formally signed off**
DECISION: Step 4 (ingestion + cleaning) is approved by the human reviewer against the full-detail `Step4_Data_Quality_Review.xlsx` (9 tabs: full clean dataset, full null-recovery detail, full System B conversion detail, full SKU master detail with FX rate shown, corrections ledger, individually-listed dropped rows, full DQ report, known limitations). All 18 regression and robustness tests pass. `clean_master.parquet` for both `data_primary` and `data_control` is the approved input to Step 5.

**D-028 · Colab download mechanism switched to Google Drive**
DECISION: `01_ingest_and_clean.ipynb` copies review outputs to Google Drive rather than triggering `files.download()`, because the direct browser-download mechanism failed twice in the reviewer's environment (stuck mid-download, no error either side) — consistent with a network policy or security software blocking Colab's blob-download trick. Drive mount + Drive's own download button avoids that failure mode while still letting the reviewer choose their local destination folder. Verified end to end: workbook downloaded via Drive, reviewed, signed, uploaded, gate check passed, `clean_master_blind.csv` produced only after approval was confirmed programmatically.

**D-029 · Approval gate verified working end to end**
DECISION: the code-enforced approval boundary (D-019 revised) is confirmed functioning as designed — an unsigned upload correctly raised `RuntimeError` and blocked all downstream output; the properly signed re-upload passed the check and produced `clean_master_blind.csv`. Step 4 sign-off is not just a convention, it is a verified gate in the pipeline itself.

---

## Step 5 — Demand characterisation

**D-030 · Notebook naming convention: step-number prefix, not run-order**
DECISION: notebooks are renamed to `stepNN_description.ipynb`, matching the architecture's actual step numbering rather than the order they happened to be built — `00_generate_test_data.ipynb` → `step02_generate_test_data.ipynb` (Step 2: run the generator; Step 1 is the generator's design, reviewed in chat, no notebook of its own), `01_ingest_and_clean.ipynb` → `step04_ingest_and_clean.ipynb` (Step 4). Step 5 is split visibly in the filename: `step05a_demand_characterization_estimator.ipynb` (blind, external) and `step05b_scoring_and_portfolio_impact.ipynb` (sighted, built here). Verified the rename introduced zero content change — byte-identical MD5 checksums against the pre-rename files.

**D-031 · Step 5a (blind estimator) accepted as final**
DECISION: the blind chat's estimator (log-space bias, censored MLE, no forecast leakage into volatility) is accepted without modification. Scored against `ground_truth.csv`: all three category signs recovered correctly (ambient +0.045 vs a true small-sample average of +0.02 against a 0.00 category setting; chilled −0.058 vs −0.07; personal care +0.072 vs +0.05), 60/60 SKUs produced an estimate, per-SKU correlation 0.657, MAE 0.046, stable across both datasets (6.16% vs 1.57% censoring). The estimator's own added rigor this round (sensitivity test, shipment-ratio audit, standard errors, an INCONCLUSIVE verdict bucket) did not change the core numbers — it added a validated honesty layer on top of an unchanged method. Checked: the 14 SKUs it flagged INCONCLUSIVE correlate with ground truth *at least as well* as the confident 46 (0.45 vs 0.37) — the caution is well-calibrated, not overcautious.

**D-032 · The 14 low-confidence SKUs concentrate on one bottleneck, not scattered**
DECISION: logged as a finding, not yet acted on. The 14 SKUs Step 5a could not confidently sign are 14/14 in source system A, 14/14 in Plant 1, 11/14 on Line 2 (chilled) — 23.3% of SKUs carrying 30.8% of volume and 24.7% of revenue (€48.7M). This is mechanically the same finding as the stockout-concentration result (materiality for flagging requires censored months, and censoring is itself almost exclusively a system-A/Plant-1 phenomenon) — one physical bottleneck producing two visible symptoms, not two separate problems. Recommendation: hold `forecast_bias_correction` at zero for these 14 into Step 6 (the lever is already defined as "share of *detected* bias," so this is what the definition implies, not a workaround) and fold the bottleneck investigation into the O-10 review already scheduled after Step 6, rather than opening a second investigation now.

**D-033 · Step 5b built directly, not via the blind chat**
DECISION: `src/portfolio_impact.py` (scoring against ground truth + the concentration report) was written directly in this chat, not handed to the blind chat, because it performs no estimation — it joins Step 5a's already-final output against business context (category, plant, line, price) the estimator never needed and never saw. The blinding boundary protects against tuning an *estimate* toward a known answer; there is no estimate here to tune, only arithmetic on results that already exist. Kept structurally separate from Step 5a's notebook regardless, since that notebook's value rests on being provably self-contained (per its own N-clause) and adding to it later — even harmlessly — would break that property for anyone auditing it afterward. Tested against real data: reproduces the manually-verified 30.8%/24.7% concentration numbers exactly, and fails loudly (`ValueError`) on an unrecognised SKU ID rather than silently ignoring it.

**D-035 · Step 5 closed, conditionally**
DECISION: D-032's recommendation is approved — `forecast_bias_correction` held at zero for the 14 low-confidence SKUs entering Step 6. Approval is explicitly conditional, not final: it is tied to the O-10 review scheduled immediately after Step 6, since D-032's own finding (the 14 SKUs sit on the same Plant 1/Line 2 bottleneck as O-10's capacity question) means this treatment may need to change once that review has real evidence. Revisit together with O-10, not separately.

**D-036 · Manual copy-paste between Step 5a and 5b eliminated**
DECISION: `get_flagged_skus()` added to `src/portfolio_impact.py`, reading Step 5a's flagged-SKU list directly from `censoring_diagnostics.csv`'s `verdict` column, replacing a manual copy-paste step between the two notebooks — a workflow requiring the reviewer to hand-transcribe values between cells is not a reusable pipeline, it is two disconnected scripts. Verified: reproduces the same 14-SKU list and the same 30.8%/24.7% concentration numbers with zero manual input, in a fully isolated test.

**D-037 · Every Step 5 number independently reproduced by the reviewer's own Colab run**
DECISION: `step05b_scoring_and_portfolio_impact.ipynb` was run end-to-end by the reviewer, not just verified in the sandbox behind this chat. Output files (`step5b_scoring.csv`, `step5b_concentration_detail.csv`) checked directly: correlation 0.657, MAE 0.046, all three category recoveries (ambient +0.045, chilled −0.058, personal care +0.072), and the 14-SKU concentration (30.8% volume, 24.7% revenue, 14/14 in source system A and Plant 1) all match exactly. Step 5 is now genuinely closed, not merely sandbox-verified — matching the standard already applied to Steps 0–4.

---

## Step 6 — MVD checkpoint

**D-038 · Engine is forward-simulating, not history-replaying**
DECISION: the Step 6 engine projects forward (using `horizon_months: 12` from `schema.yaml`, already present but unresolved until now), not backward-replaying the 36 months of history under counterfactual lever settings. The two modes answer different questions — "what would this quarter have cost under different rules" vs. "what should I do next quarter" — and the tool's actual purpose (supporting forward decisions, e.g. whether to build inventory ahead of a promotional demand spike) requires the second. History is retained only as the input that characterizes demand behavior (Step 5's volatility/bias) for the forward simulation, not as the thing being recomputed. Working capital carrying cost (`stock_value × WACC/12`) is applied identically either way — the change is what months it's applied to, simulated future ones rather than actual past ones.

**D-039 · Warehousing/storage cost deferred, not built**
DECISION: a fifth output (external warehousing cost) is deferred rather than added now, despite the engine mechanism being a straightforward extension of the existing capacity/overflow pattern already required for conversion cost (shared constrained resource, overflow cost when exceeded). The blocking cost is not modeling complexity — it's that cases-per-pallet must live in `sku_master`, which is generated at Step 1 and locked at Step 4's sign-off; adding it means regenerating both datasets, re-cleaning, and re-running the full Step 4 human review and sign-off gate a second time. Explicitly a time/effort trade-off, not a scope-discipline default — revisit if a later step forces a Step 1/2 regeneration anyway for an unrelated reason.

**D-040 · Cross-lever interaction to check at the MVD gate**
DECISION: `min_run_hours` is expected to move service level *indirectly* — longer runs build larger average batches, which raises buffer stock and therefore service level, separately from the explicit `service_target` lever. The MVD should check for this as three outputs moving together from one lever (conversion cost down via fewer changeovers, carrying cost/obsolescence up via more buffer, lost sales down or flat) — a stronger demonstration of the trade-off engine than a single-lever/single-output pairing, and a real cross-lever coupling the model should reveal rather than hide.

**D-041 · Engine demand representation: deterministic, not stochastic simulation**
DECISION: Step 6's forward horizon uses one deterministic 12-month demand path (baseline + trend + seasonality, corrected by whatever `forecast_bias_correction` the lever specifies), with lost sales computed analytically from Step 5's `irreducible_volatility_cv` and the service-target lever via standard safety-stock-style formulas — not by simulating many random future paths and averaging. Chosen over stochastic simulation because it uses the same underlying technique real inventory-planning tools use in practice (not a simplification that weakens the pitch), needs no new seeding/reproducibility discipline, and is proportionate to an MVD gate meant to be minimum. Revisit only if the coach conversation specifically asks for uncertainty ranges the deterministic version can't show.

**D-042 · Strategy: run end-to-end before the coach meeting, decide scope changes after**
DECISION: proceeding with the simplest viable choice at every remaining open design point (D-041 here; same spirit as D-039's storage-cost deferral) specifically to reach a working Steps 6–13 pass before the scheduled coach meeting, rather than fully resolving every design question first. Rationale: a working model gives the coach something concrete to react to, and lets scope/robustness decisions (O-10, storage cost, deterministic-vs-stochastic, the 14 flagged SKUs) be made with real output in front of both parties rather than as abstract trade-offs — this is also the intended shape of the scalability section of the rubric, answered as informed next steps rather than unresolved gaps. Every deferred item remains logged and open, not abandoned.

**D-043 · `sku_master` is persisted, not held in memory**
DECISION: Step 4 writes `<data_root>/clean/sku_master.parquet` alongside `clean_master.parquet`, because Step 6's engine needs `price_eur`, `std_cost_eur`, `gross_margin_eur`, `shelf_life_days`, `moq_units`, `min_run_units` and `line_code` per SKU, and Step 10's Streamlit app cannot re-run the cleaner to obtain them. This does **not** reopen Step 4's sign-off: no cleaning logic changes, and all 60 rows were already reviewed on sheet 5 of the approved `Step4_Data_Quality_Review.xlsx`. Persisting reviewed content is not re-deciding it.

**D-044 · `canonical.line_master` is built at runtime, not stored**
DECISION: `build_line_master()` in `src/engine.py` constructs the `line_master` table that `schema.yaml` has declared since Step 0 but which no pipeline step ever emitted. Built from `assumptions.lines` + `assumptions.categories.<cat>` + `assumptions.plant_economics` rather than persisted as a generated artefact, because line economics (speed, changeover hours, labour rate, capacity) are *assumptions*, not observed data — storing them would create a second place to change them and a second thing to keep in sync. The function validates its output against the schema declaration so the two cannot drift apart silently.

**D-045 · MVD horizon is 12 months, superseding architecture §5's "one month"**
DECISION: the Step 6 gate simulates the full `horizon_months: 12` rather than the single month §5 specifies. A one-month horizon cannot show inventory build, stock ageing, write-off or the cost of a min-run batch that covers several months of demand — three of the four economic outputs would be structurally inert and the gate would prove nothing. This is a direct contradiction of the architecture document, which is the project's source of truth; §5 is therefore **amended**, not quietly overridden. See `architecture-amendments.md`, A-01. Consistent with D-038, which already required forward simulation.

**D-046 · Step 6 builds `src/engine.py` directly, not a throwaway MVD module**
DECISION: the MVD is built as the real `src/engine.py` scoped to one line, rather than a disposable `src/mvd.py` later replaced at Step 8. Step 8 is a broadening of the same class (more lines, all levers together), not a different object, and the Mastery rubric rewards a reusable, class-based pipeline over a one-off script. Scope discipline is preserved by *what the gate demonstrates* (one line), not by artificially crippling the module: `line_id` is a parameter, and `test_engine_runs_on_every_line_without_code_change` asserts the code is line-agnostic without any second line being claimed as a result.

**D-047 · The second economic output is excess/slow-moving risk, not shelf-life write-off alone**
DECISION: `excess_obsolescence_eur` carries two mechanisms — shelf-life write-off (age past `shelf_life_days × (1 − writeoff_trigger_pct_life_left)`) *and* a slow-moving provision on stock beyond `inventory_risk.excess_cover_months_threshold`. Two new keys are added to both assumption files: `excess_cover_months_threshold: 4.0` and `excess_provision_pct: 0.25`, both LOW confidence.

Rationale: architecture §2 names this output "Excess / **slow-moving** risk", which is wider than obsolescence, and a write-off-only mechanism is **structurally dead** on the MVD line — personal care's shelf life is 1,092 days with a 10% trigger, so stock must sit ~32 months before write-off and the horizon is 12. Not "moves least by design" (D-008) but exactly zero, which would leave the gate demonstrating three outputs, not four. The write-off half is confirmed live on L2 (6-week shelf life) as a code-path liveness check, not as a second-line result. This is a disclosed addition to the assumption set, and it feeds the limitations slide as such.

**D-048 · Excess cover is measured against demand, never against the plan**
DECISION: the slow-moving threshold in `_excess_units()` is `threshold × demand_units`, not `threshold × plan_units`. Measuring against the plan created a real circularity leak found during Step 6 build: lowering the plan via `forecast_bias_correction` shrank the "allowed" cover and reclassified *identical physical stock* as excess, moving the cost through a purely definitional channel rather than a physical one (excess swung €179k → €261k on the bias lever alone). After the fix the same sweep moves excess by ~0.001%. This is precisely the failure mode §9's guardrail exists to catch, and it was caught by building the engine, not by inspecting the design.

**D-049 · The `constant:` schema check is implemented, correcting D-027**
DECISION: `DataIngestor._validate_columns()` now enforces `constant:` and `values:` declarations from `schema.yaml`. `tests/test_pipeline.py::test_currency_drift_raises` asserted this behaviour but the check was never implemented — the test has been failing since it was written.

**D-027 is therefore incorrect as recorded**: it states "All 18 regression and robustness tests pass" when 17 passed and one failed. Verified by stashing all Step 6 changes and re-running the suite on the committed code. Corrected here rather than edited in place, so the error and its correction both stay on the record. Step 4's *sign-off* stands — the failing test concerned an unseen-defect robustness scenario, not any row of the reviewed output — but the claim attached to it did not. Now 18/18.

**D-050 · MVD gate pass criterion, set before the run**
DECISION: the gate passes on all four outputs being **computed, finite, non-zero and directionally correct**, plus an interior optimum on at least one lever — not on all four outputs moving materially. Fixed in advance of running, because two outputs are known to be structurally muted on L3 (write-off by D-008/D-047; conversion cost because fixed absorption is ~94% of it, O-12) and a criterion set after seeing results is not a criterion.

### Open at Step 6 close

| Ref | Question | Status |
|---|---|---|
| O-11 | L3 realised utilisation is ~35% in the forward simulation, so `min_run_hours` never reaches the overtime step and the **capacity leg of the conversion-cost mechanism is untested at the gate**. The MVD demonstrates changeover and inventory mechanisms, not capacity pressure. This is O-10 surfacing exactly where it was predicted to. | Open — feeds the O-10 review, now due. |
| O-12 | `fixed_absorption_eur_line_month: 300_000` is ~94% of conversion cost, so the lever moves the total by ~3% and the signal is swamped in level terms. Reporting should show deltas vs baseline alongside levels, or the pitch audience will read "conversion cost barely responds". | Open — a reporting decision for Step 9, not a model change. |
| O-13 | `src/demand_characterization.py` and `tests/test_demand_characterization_recovery.py` are referenced in `handoff_notes.md` but **do not exist in the GitHub repo**. Step 5a exists only as `step05a_demand_characteristics_estimator.ipynb`. Either the module was never extracted from the blind chat's notebook, or it was lost before commit. | Open — resolve before Step 10, which needs importable estimator logic. |

**D-051 · Step 6 MVD gate passed**
DECISION: the gate is recorded as passed on 2026-08-14 against the D-050 criteria fixed in advance, because all four outputs are live and directional, `min_run_hours` produces the predicted three-way coupling, `inventory_cover_weeks` has an interior optimum at 4 weeks, and both suites pass on the reviewer's own Colab run (17/17 engine, 18/18 pipeline).

**D-055 · Storage and handling cost deferred, and specified as a mechanism when built**
DECISION: inventory continues to be priced at WACC only for now, and when storage is added it will be driven by **pallet positions** (`units_per_pallet` in sku_master × EUR/pallet/month), not by a percentage of stock value. A value-based rate is a coefficient rather than a mechanism and inverts the real incentive — a bulky low-value SKU costs more to store than a compact high-value one. Correct form requires master data additions spanning steps 1-4 (generator, schema, cleaner), which is out of scope at this point in the timeline. Carried as a limitation: holding cost is understated, so every lever is structurally biased toward more inventory.

**D-056 · Commercial consequences of poor service are documented, not modelled**
DECISION: substitution margin dilution, expedite cost and OTIF penalties are recorded as an explicit limitation rather than implemented as null-valued parameters. They are master and commercial data (substitution matrix, channel freight, customer contracts), not assumptions. A structurally-present but permanently-unpriced parameter is assumption opacity with extra steps — the model would appear to price penalties while pricing none. Carried as a limitation: the cost of low service is understated. Note this pulls opposite to D-055 and the two do not cancel.

**D-057 · Stockout disposition stays a flat share; the order-book mechanism is rejected**
DECISION: the existing `stockout_disposition` shares (ambient 0.40, chilled 0.35, personal care 0.50 lost) remain the reordering adjustment, and a shortfall decomposition with backlog carry-forward was considered and is **not implemented**. The flat share already *is* the percentage reduction for reordering; layering a further reduction on top would double-count the same reasoning.

The order-book version was built in exploration and reverted. It required a backlog service rule, a priority rule against current demand, and a cancellation/expiry rule — the last being a new ungrounded parameter, which D-056 had just ruled out. Disproportionate to the remaining timeline and to the question being answered. The substituted and backordered remainder is stated as absorbed outside the model: a documented simplification, not a claim that the model tracks all volume.

**D-058 · Capacity backlog is consumed by the request that carries it**
DECISION: `backlog[s]` resets to zero once folded into a production request, because the request that carries the deferred volume also satisfies it. Without the reset it accumulates every month and production diverges from demand — observed at over 2× annual demand in testing.

This bug was **latent in the passed-gate code**, not introduced by the revision: `backlog[s] += cut` had no matching decrement since Step 6 was written. It never fired because L3 never breaches capacity, so `cut` was always zero. It would have surfaced the moment a tighter line was run — which is precisely what the O-10 review does. The gate result stands: on L3 the buggy path was never executed. `test_production_does_not_diverge_from_demand` now asserts the invariant at three lever settings rather than only the default.

**D-059 · O-14 closed — safety stock off demand; the bias lever's direction is unchanged**
DECISION: safety stock is sized off demand, not the bias-carrying plan (`safety = z × cv × demand_units`), and the change does NOT flip the forecast-bias lever. On L3, correcting the full 6.4% detected over-forecast costs €52,617 in lost margin and saves €7,370 in working capital, so total cost rises €4.309m → €4.354m. The over-forecast buys fill rate (99.30% → 99.65%) worth more than the inventory it ties up at a 52% margin.

Robust to the known holding-cost gap: even at a realistic 18-25% annual carrying rate the working-capital saving roughly doubles to ~€17k, still far short of €52k. This is a result about a HIGH-MARGIN, LONG-SHELF-LIFE line, not a general finding — on chilled, where write-off actually fires, it may reverse. Do not state it as a general conclusion in the pitch.

**D-060 · Gate result restated on the revised engine**
DECISION: with O-14 and D-058 applied, the Step 6 gate figures are lost sales €42,014, excess/slow-moving €261,364, conversion €3,820,954, working capital €184,700, total €4,309,032; service achieved 0.9965, mean utilisation 0.352. All six D-050 criteria pass; 20/20 engine tests and 18/18 pipeline tests. Interior optima unchanged at `min_run_hours` 9 and `inventory_cover_weeks` 4 weeks. D-051 stands on the pre-revision engine; this is the current figure set.

Process note: the first post-revision report was invalid. The gate report cell reused a cached in-memory engine built from the previous `src/engine.py` while its pytest lines read the new code from disk, producing pre-revision numbers beside post-revision test counts. Fixed by always rebuilding and printing the `engine.py` hash plus an explicit "safety stock off: demand / PLAN" provenance line in the report header.

**D-062 · The four outputs are reported as AVOIDABLE cost**
DECISION: conversion cost is reported excluding `fixed_absorption_eur_line_month`, which no lever changes. Including €3.6m of unavoidable absorption compresses a €110k lever effect into noise and misrepresents which output the decision turns on: at the gate default the reported set becomes excess €261,364, conversion (avoidable) €220,954, working capital €184,700, lost sales €42,014 — four outputs of comparable magnitude rather than one large number plus rounding.

Fixed absorption is DISPLAYED separately and labelled, never silently dropped, and the output is labelled "conversion cost — avoidable". A plant controller who sees €221k unlabelled will conclude the model has lost €3.6m of their P&L. No interior optimum moves: a constant shifts every scenario equally. Implemented at Step 9 (reporter); no engine change. Limitation for the pitch: optimising avoidable cost means the tool will never recommend closing a line — that decision needs the absorption back in.

**D-063 · The O-10 review moves to after Step 8**
DECISION: O-10's review is rescheduled from "immediately after Step 6" to immediately after Step 8, because its own trigger condition requires the engine to have run across all four lines and all four levers together — which is the definition of Step 8. Step 6 is deliberately one line, so the evidence O-10 asked for could not exist at the Step 6 gate. The original commitment was made before the evidence requirement was understood. D-034's conditional treatment of the 14 flagged SKUs travels with it, unchanged in the meantime.

**D-064 · Step 7 redefined — optimal-policy model, not cost-response model**
DECISION: Step 7 no longer learns cost response to schedule disruption. The engine already computes conversion cost causally, so a model fitted to engine output would be learning the engine — circular, and §9 territory.

Step 7 instead (a) sweeps the engine per SKU for the cost-minimising inventory cover and service target — deterministic, no ML, and this is the operational answer — then (b) fits SKU ATTRIBUTES to those optimal parameters, benchmarked (XGBoost vs ridge vs a naive ABC-class baseline) and explained with SHAP.

Purpose: set an opening policy for a SKU the engine cannot sweep. The principal case is a NEW LAUNCH — no demand history, so Step 5a cannot characterise it and the engine cannot simulate it, but every attribute the policy model needs is known before first shipment. Two feature sets are fitted and reported separately, FULL and LAUNCH, the latter excluding `irreducible_volatility_cv` and `chronic_bias_l1`; reporting only FULL would overstate real-world performance, since the most predictive features are exactly the ones a launch lacks.

`min_run_hours` is excluded from the per-SKU optimum — it is a line decision, and changeovers are a property of the sequence on a line. `forecast_bias_correction` is excluded as a planner-accountability lever, not a per-SKU policy parameter.

For the pitch and the limitations slide: **the optimisation is deterministic; machine learning explains and generalises the policy rather than producing it.** Output renamed `src/cost_model.py` → `src/policy_model.py`. Architecture amended to v4.

**D-065 · The policy model cannot be fitted on a single line**
DECISION: Step 7's sweep runs across all four lines (60 SKUs), not the MVD line alone. Measured on L3: of the ten attribute features, **only `abc_class` has any variance**. Category, line speed, shelf life, price, standard cost, case size, MOQ and minimum run are each constant within a line, because one line carries one category on one machine.

The consequence is not a weak result but an uninterpretable one — the model is mathematically forced to reduce to the ABC-class mean it is benchmarked against, and therefore cannot beat it. The single-line run confirmed this exactly: SHAP returned non-zero importance for `abc_class` and precisely 0.0000 for every other feature, MAE margins were ~2% on 15 rows, and the LAUNCH set outscored FULL — removing informative features improved performance, which happens only when there is no signal.

The optimiser was already line-agnostic, so this is `optimise_all_lines()`, a loop, not new logic. Two tests now assert the fact rather than leaving it in prose: `test_single_line_training_has_almost_no_feature_variance` and `test_portfolio_training_restores_feature_variance`. The single-line run stands as a demonstration of the pipeline, not as a result.

**D-066 · O-16 resolved — `separable` is the per-SKU objective**
DECISION: the per-SKU optimum is taken over the three separable outputs (lost sales, excess/slow-moving, working capital). Conversion cost is excluded and remains a line-level decision through `min_run_hours`. `delta` attribution is retained for comparison only.

Evidence: on the MVD line, **8 of 15 SKUs chose a different optimum** under `delta` versus `separable`. Pro-rata by run hours is one defensible split among several, so more than half the per-SKU answers moved with an arbitrary modelling choice. An answer that depends on that choice is not an answer.

It also matches the architecture's own division of levers — `min_run_hours` is the line lever, cover and service are the SKU levers — so letting a per-SKU optimum depend on a joint line cost contradicts the design.

Cost of the decision, for the limitations slide and stated rather than hidden: a SKU that wants small frequent batches genuinely does impose changeover cost on its neighbours, and `separable` cannot see that interaction.

**D-067 · `total_cost_at_default_eur` was computed from class A's policy for every SKU**
DECISION: fixed a genuine bug found on the reviewer's own Colab run, not in sandbox testing. `PolicyOptimiser.optimise()` computed each SKU's default cost using `self.base_levers` uniformly — which is always class A's defaults (service floor 0.985, cover 4.0 weeks) — regardless of the SKU's own ABC class. Class B and C SKUs were therefore scored against a policy that was never theirs. Symptom: 4 SKUs, all class B, showed negative `saving_eur` — an optimum "worse than the default" is structurally impossible when the default is itself an admissible grid point, so a negative value is proof of a computation error, not a modelling result.

Fixed on two levels: (a) each SKU's own class default (from `LeverSettings.defaults()` on its actual `abc_class`) is now added as an explicit evaluation point, not just used as the comparison baseline; (b) `total_cost_at_default_eur` is looked up per SKU against its own class default, with a direct fallback evaluation if that point is somehow absent from the sweep. A secondary, related cause is grid coarseness: a class default may not land exactly on the shared cover/service grid, and because the cost surface is not smooth (min-run rounding, the excess-cover step), a coarse grid can score worse everywhere it samples than an untested default point — the class-default-as-candidate fix addresses this too.

Effect on the portfolio total: saving vs default rose from EUR 458,144 to EUR 626,976 once B/C-class SKUs were correctly compared against their own policy rather than A's tighter one. `test_optimum_never_worse_than_default_across_all_abc_classes` added, requiring more than one ABC class present — the MVD line alone cannot guarantee this and did not catch the bug in earlier sandbox runs.

**D-069 · Step 8 confirms D-059 generalises; the real inconsistencies are mechanistic, not noise**
DECISION: on the real Colab run, only 4 of 20 lever/output pairs are inconsistent across the four lines — none involving `forecast_bias_correction`, correcting a speculation raised after the earlier sandbox run (contaminated data had suggested the bias lever might flip by line). D-059's finding — the bias lever's direction on total cost does not flip sign — holds across the full portfolio, not just L3.

The four pairs that ARE inconsistent all involve `inventory_cover_weeks` or `min_run_hours` against `excess_obsolescence_eur` or `total_economic_cost_eur`, and in every case **L2 is the line that moves opposite the other three**. This is mechanistic, not noise: L2 is the only line with a short enough shelf life (6 weeks) for write-off to actually bind, so additional cover — which buffers volatility everywhere else — buffers stock straight into obsolescence there instead.

**D-070 · O-10 resolved — static SKU-to-line allocation is not rebuilt**
DECISION: the SKU-to-line allocation stays static. Evidence, from the full Step 8 sweep (1,024 scenarios): only L2 breaches capacity, and only at 64 of 256 grid points for that line (25%) — every breaching row requires `inventory_cover_weeks = 10.0`, the top of the admissible range. L2's own cost-optimal setting (cover 4.0, min-run 2.0) has **zero** capacity shortfall.

The risk is real — a planner who overrides the tool toward high cover on L2 specifically can push it into overtime — but it does not occur at the setting the model itself recommends, and reallocating SKUs across lines to hedge against a combination the optimiser would never choose is not proportionate given the remaining timeline. O-10 closes as: **documented limitation, not rebuilt.** Step 9's reporter must carry this forward as an explicit warning on the L2 + high-cover combination, not bury it in a footnote — see the Step 9 SNAKE prompt.

O-10 status: **CLOSED.**

**D-071 · Per-SKU policy was never jointly simulated — engine redesigned to per-class levers, verified in one combined simulation**
DECISION: `LeverSettings` is redesigned to accept a per-ABC-class value for ALL FOUR levers (`service_target`, `inventory_cover_weeks`, `forecast_bias_correction`, AND `min_run_hours` — not just the first three), with capacity checked ONCE, in aggregate, on the real combined batch schedule that results.

**The defect this fixes.** `optimal_policy.csv` (D-068) reported a per-SKU optimum found by sweeping the WHOLE LINE at one uniform lever setting per grid point, then picking — for each SKU — whichever uniform-line run happened to be cheapest for it. The combination "SKU A at its own optimum, SKU B at its own different optimum, simultaneously" was never simulated, because `run_scenario()` only accepted one `LeverSettings` per line. Capacity was verified for each individual uniform run, never for the actual mixed policy the €622,610 saving implied. This was found through direct questioning of the Step 7 mockup, not through testing — the model was never wrong in a way any existing test could have caught, because no test asserted joint feasibility of a combined policy that the engine had no way to construct.

**Why `min_run_hours` moved from line-level to per-class (correcting D-064/D-066's stated position).** Those decisions held minimum-run policy at the line level on the reasoning that changeovers are a shared, physical line resource. That reasoning is still correct for the CAPACITY CHECK, but wrong for the POLICY ITSELF: a batch-size floor ("don't run a C-class batch under 2 hours, don't run an A-class batch under 9") is exactly the same kind of class-level policy decision as cover or service, and real plants set it that way. The only genuinely line-level fact is the aggregate constraint — total run hours plus changeovers versus scheduled hours plus max overtime — which is checked once, in the same joint simulation, regardless of how each class's batch floor was set.

**Why `forecast_bias_correction` also moved to per-class (correcting D-064's "planner-accountability lever" framing).** That framing was invented to justify keeping the lever uniform: convenient, not required. `chronic_bias_l1` is already per-SKU in `demand_characteristics.csv`; only how much of it to act on was forced uniform. No structural reason survives scrutiny for keeping it that way.

**Scope of the change.** `LeverSettings` accepts either a single float (backward-compatible with every existing Step 6/8 result, which used one uniform value) or a `{A: x, B: y, C: z}` mapping. `simulate()`'s existing per-SKU loop looks up each SKU's own class value — production, shipment and capacity math were already per-SKU, so this is a contained change to the lever-lookup, not a rewrite of the mechanism. Step 7 is redone in full: it now searches per-class combinations directly, with each grid point one real joint simulation, so capacity is genuinely verified for the combination reported as optimal. Step 8's uniform-line sweep is unaffected — it answers a different, still-valid question. Step 9 gains a by-class breakdown view. Step 10's sliders and cost bridge are redesigned to match (dashboard layout validated in a Claude-artifact mockup before any Streamlit code was touched, to avoid iterating on layout through slow redeploys).

**Consequence for D-068.** The €622,610 saving figure is retracted pending the redone Step 7. It may turn out to be smaller once genuinely joint capacity constraints bind on the combined per-class policy — that is itself a legitimate finding, not a failure, and will be reported honestly either way.

**Time-budget decision, made explicit rather than left implicit.** The Step 6 engine change and the Step 7 redo are treated as mandatory regardless of remaining time or final grading level — the alternative is leaving a known-unverified figure in the submitted record, which is not an acceptable trade at any level. Step 9's two label corrections (see D-072 note in that step) are cheap and will be done if time allows. Step 10 (Streamlit app) and Step 11 (agent stress-test) are the explicit, pre-agreed fallback: if time runs out after Steps 6/7/9, the submission proceeds at Proficiency-level scope rather than Mastery, dropping the deployed app and multi-agent review, with this decision on record as a deliberate trade-off made under real time constraint — not an oversight.

**D-072 · Step 9 redone against Step 7's real output — `policy_brief` retired, not patched**
DECISION: `policy_brief()` is removed and replaced by two functions that match what Step 7's joint search actually produces. `line_policy_brief()` flattens `search_all_lines()` into one row per (line, class) — the direct "what to set" table, carrying the line's own `saving_eur` once per class rather than inventing a per-class share of it. `class_breakdown_view()` aggregates REAL per-class economics from `expand_to_sku_level()`'s per-SKU rows (genuinely summed, not the illustrative split the dashboard mockup used as a placeholder) — and deliberately does NOT split conversion cost per class, since D-066's finding that conversion isn't separable per SKU still holds; it is shown once per line, unsplit, rather than manufacturing a number this project has already found misleading when done the other way.

`avoidable_cost_view`, `capacity_warning`, `owner_view`, `portfolio_brief` are untouched — none depend on Step 7's output shape.

**Known, deliberate, temporary breakage:** `src/app_logic.py` still imports the retired `policy_brief`, so `tests/test_app_logic.py` fails at collection and `app.py` would not run. Left broken on purpose rather than shallow-patched: `app_logic.py`'s `top_policy_moves()`, the committed `app_data/optimal_policy.csv` contract, and the fingerprint check all need to change together as part of Step 10's already-planned full rebuild (new sliders, new bridge, new `app_data` shape) — patching the import alone now would produce a shallow fix thrown away within the same work session. `tests/test_app_logic.py` is excluded from the "all tests pass" claim until Step 10 lands; every other suite (114 tests across engine, policy_model, portfolio_sweep, reporter, pipeline) passes.

**D-073 · Step 10 rebuilt against the real per-class engine and Step 9's reporter — verified beyond HTTP 200**
DECISION: `app.py`/`src/app_logic.py` fully rewritten to match the dashboard mockup agreed with the user before any Streamlit code was touched. Twelve sliders (four levers x three ABC classes), a multi-select line picker, Base/Optimal preset buttons, a cost bridge with both delta rows, and — the substantive change — capacity warnings computed LIVE from the scenario actually being viewed (`run_scenario_for_lines()` + `live_capacity_warnings()`), not looked up in a static Step 8 table as the retired design did.

**"Base" is redefined to match D-072**: each class at its own true default (`LeverSettings.defaults_per_class()`), not class A applied to the whole line. A dedicated test (`test_base_levers_are_not_class_a_applied_to_everyone`) exists specifically to stop the old convention resurfacing silently.

**Two real bugs found and fixed during this build**, same discipline as every prior step:
1. `by_class_breakdown()`'s first draft passed the engine's raw `sku_month` (one row per SKU **per month**) straight into `class_breakdown_view()`'s groupby, which would have counted SKU-months as SKUs. Fixed by summing to one row per SKU first.
2. The same function initially had no `cover_weeks`/`service_target` columns at all — `class_breakdown_view()` needs them but the engine's `sku_month` doesn't carry the lever value itself, only its effects. Fixed by deriving them from the current `LeverSettings` via `levers.resolve()` before aggregation.

**Verification method upgraded, not just repeated.** `HTTP 200` and an empty terminal log — the standard used for the first two app.py builds — do NOT prove a Streamlit script executed without error; Streamlit can serve 200 while showing an error boundary client-side. This build was verified with `streamlit.testing.v1.AppTest`, which executes the script directly and exposes real exceptions and real widget state (confirmed: 12 sliders, 7 buttons, 6 dataframes, zero exceptions), plus simulated interaction (button clicks, `.set_value()` on sliders) confirming the capacity warning fires on L2 under stress and stays silent on L1 under the identical stress — proving the live capacity check is genuinely live, not coincidentally correct. Also re-verified under a genuine read-only bind mount (not `chmod`, which a root-run sandbox ignores and would have given a false pass) to reproduce Streamlit Cloud's exact deploy condition from the first failed attempt.

`app_data/optimal_policy.csv` is deleted (superseded by `line_results.csv` + `sku_level.csv`, matching Step 7/9's real output shape). 132/132 tests pass across every module — the `test_app_logic.py` collection failure noted as a known gap in D-072 is now closed.

**D-074 · Sandbox data contamination caught before merge — app_data corrected to real Step 5a output**
DECISION: the `app_data/line_results.csv` and `sku_level.csv` committed in D-073's initial push were built from a contaminated sandbox copy of `demand_characteristics.csv`/`censoring_diagnostics.csv` (5 flagged SKUs, not the real 14) — the same sandbox contamination flagged repeatedly earlier in this project's build. Caught by a direct check before considering D-073 closed, not discovered later. The real files (14 flagged, matching every Colab-confirmed run) were recovered from the ORIGINAL `main` branch commit, where the user's genuine Step 5a upload had already been committed during the first Step 10 build.

Sandbox's `data_primary/clean/demand_characteristics.csv` and `censoring_diagnostics.csv` were overwritten with the real files at the source, and `search_all_lines(n=3)` was re-run end to end (206s). Corrected total saving vs true per-class default: **EUR 615,470** (previously reported, on contaminated data: EUR 619,889 — a ~4.4k difference, not large, but the contaminated figure was never the number to trust regardless of magnitude). All four lines remain fully capacity-feasible at the corrected optimum. App re-verified via AppTest against the corrected `app_data/` — clean, 132/132 tests.

**D-075 · Colab notebooks rebuilt for Steps 7 and 9; a second contamination catch in committed Step 8 data**
DECISION: `step07_policy_model.ipynb` and `step09_reporter.ipynb` were still calling retired functions (`optimise_all_lines()`, the deleted `policy_brief()`) — Step 9's notebook would have crashed outright on import. Both rebuilt against `search_all_lines()`/`expand_to_sku_level()`/`line_policy_brief()`/`class_breakdown_view()`. `step06_mvd_engine.ipynb` needed no change — it only uses scalar `LeverSettings.defaults()`, which D-071 kept fully backward compatible.

A new function, `sku_level_for_policy_model()`, reshapes `expand_to_sku_level()`'s output into the column names `PolicyModel` (the ML/SHAP half of Step 7) still expects — `PolicyModel`'s own logic is unaffected by D-071 and was not rewritten, only the column names feeding it. `edge_optimum_flag` is set `False` for every row: the old flag existed because the retired per-SKU search commonly landed on grid boundaries as an artefact of attributing a uniform-line run to one SKU; these rows are genuinely joint-optimal, feasibility-filtered points, so there is no reason to exclude any of them from training.

**Honest, expected result surfaced by the rebuilt notebook, not hidden:** `optimal_cover_weeks`'s naive-baseline MAE is exactly 0.0 in this run — the n=3 grid found the same corner solution optimal on every line, so cover is literally constant within each class across the whole portfolio, and no model can beat a baseline that is already exact. Reported as what it is, not smoothed over.

**A second contamination catch, same root cause as D-074:** `app_data/step08_portfolio_summary.csv` and `step08_lever_consistency.csv` were ALSO built from the sandbox's earlier-contaminated data, giving 8 of 20 inconsistent lever/output pairs instead of the real 4 — caught by comparing the rebuilt Step 9 notebook's `portfolio_brief()` output against the number already confirmed from the user's actual Step 8 Colab run. Both files recovered from `main`, where the user's genuine Step 8 upload had already been committed during the original Step 10 build. Re-verified: `portfolio_brief()` now reports "4 of 20 ... concentrated in: inventory_cover_weeks, min_run_hours, service_target" — `forecast_bias_correction` correctly absent, confirming D-069 held.

Both rebuilt notebooks executed end to end in the sandbox (Step 7: ~340s including its own test suite; Step 9: fast, reusing Step 7's cached search result) with all consolidated-report checks PASS. 132/132 tests across every module; `app.py` re-verified via `AppTest` against the fully corrected `app_data/`.

**D-078 · "Jump to base"/"Jump to optimal" did not move the sliders — a Streamlit session_state trap, and a gap in my own testing**
DECISION: fixed a real bug the user caught by using the app, not by anything in the automated test suite. Each slider is created with a `key=` (e.g. `svc_A`). Streamlit's actual rule: once a widget's key exists in `session_state`, the widget's `value=` argument is IGNORED on every subsequent rerun — only `session_state[key]` is read. The original preset-button code updated a separate `current_levers` dict and passed it into `value=`, which never touched the sliders' own keys — clicking the button registered (no exception, no visible error) but the sliders never moved.

Fixed by writing directly into each slider's own `session_state` key (`svc_A`, `cov_A`, ... 12 keys total) inside the button handler, followed by `st.rerun()`. A second issue surfaced during the fix: passing both `value=` and a pre-existing `session_state[key]` in the same run triggers a Streamlit policy warning (not yet a hard error, but flagged as discouraged and liable to become one). Resolved properly — `session_state[key]` is seeded once, before the widget is first created, and `value=` is dropped from every slider call thereafter, matching Streamlit's own recommended pattern.

**Why this passed my own verification the first time.** Prior `AppTest` checks confirmed clicking the button caused no exception, and confirmed capacity warnings fired correctly under manually-set slider values via `.set_value()` — but never checked that the PRESET BUTTONS themselves changed slider VALUES. That is a real gap in the verification method itself, not just the code: "no exception" is necessary but not sufficient evidence a UI action did what it claims. A permanent regression test now asserts a slider's value changes after a preset click, not merely that the app survives the click.

Re-verified end to end: manually move a slider, confirm it moved; click "Jump to base", confirm it returns to the true per-class default (`svc_A=0.985`, `cov_C=6.0`); click "Jump to optimal", confirm it moves to Step 7's actual per-line optimum (`svc_A=0.88`, `cov_C=1.0` for L1) — genuinely different values, not a repeat of Base. 133/133 tests pass.

**D-079 · Four levers reduced to three — service reclassified from decision to reported outcome, min_run promoted to dominant lever**
DECISION: following a full-grid diagnostic review (external, verified line-by-line against real sweep data before acceptance — see below), `service_target` is removed as a decision lever and reported as an achieved output instead, because both service and cover terminate in a single `target_stock` scalar (`cover_months × plan + z(service) × cv × demand`) that every cost formula downstream reads only as a total. Two different (service, cover) pairs producing the same total stock are numerically indistinguishable to the model — the split between them was never identifiable, and D-071 through D-078's "88% beats 98.5%" framing, while numerically real, implied a service recommendation the model cannot actually support.

DECISION: `service_target` is retained as a frozen field on `LeverSettings` rather than deleted, because removing it breaks the Step 6 test suite inside the delivery window. It is set to each SKU's ABC-class default on every scenario and no longer appears in the search grid, the app's slider set, or any results table as a decision — only as `service_achieved` (per line, per class) and `unit_fill_rate` (aggregate: `1 − Σlost/Σdemand`, never a mean of per-row ratios).

DECISION: removal sequenced after confirming, on real data, that `bias_correction` and `min_run_hours` — previously held at 0.0 and a per-line category default respectively, across all 2,920 rows of the original Step 7 grid — actually move cost within a single line. Confirmed: both do. `min_run_hours` dominates by roughly two orders of magnitude (€1.21M spread on L2 vs a few thousand euros for bias correction). Reflected in the lever ordering in `architecture.md` §2 and in this log.

DECISION: the bias-correction × min-run interaction is reported explicitly, including cells where bias has exactly zero effect (confirmed real on L2 at min_run ∈ {9, 12}: identical cost to the euro across every tested bias value) — the batch floor exceeds the plan adjustment at those settings, so forecast accuracy cannot reach the P&L. Not suppressed or smoothed.

DECISION: all results reported as delta vs base, with fixed absorption (67.7% of total cost, confirmed by sum/sum against the full 2,920-row grid) stripped out before the delta is taken — not just noted alongside it. Verified directly: an apparent discrepancy in reported L2 cost levels (this session's own figures vs an external reviewer's) turned out to be pure conditioning artefact — three different ways of holding the other five dimensions fixed gave three different LEVELS (€4,684k / €4,613k / €5,057k at service=88%) but an IDENTICAL delta (+€2.0k, then +€114k) as service moved, reproduced across all three conditionings to the euro. Levels are conditioning-sensitive; deltas are not.

**Verification discipline applied before any code changed:** every quantitative claim in the external review — grid values, min_run/line confounding, conversion's 67.7% share and 8.3–8.6% variation, the cover interior optimum (L1+L4: cost falls then rises as cover moves 1.0→5.5→10.0), service saturation, the specific L2 reconciliation table — was independently checked against this session's own real engine output before Change 1 began, and again before Changes 2–5. All confirmed, several to the exact euro. One genuine factual correction was surfaced and accepted in return: the originally-stated premise that "5.5 and 1.0 are the maximum and minimum" of the cover grid was wrong — 5.5 is the midpoint; 10.0 is the true maximum and was never selected.

**New engine-facing functions (`src/policy_model.py`):** `search_three_lever_policy()` (cover × bias × min_run, service frozen, replaces the retracted 2-lever search as the live Step 7 search), `best_feasible_three_lever()`, `search_all_lines_three_lever()`, `bias_min_run_interaction()`. The retired 2-lever functions (`search_joint_policy`, `search_all_lines`) are left in the file, unremoved — Step 6's tests exercise scalar `LeverSettings` independently and are unaffected. `src/reporter.py` gains `delta_vs_base_view()`.

**Real result on real data, all four lines, fully feasible:** total saving vs true base is **€695,268** (revised up from D-071–078's €615,470, since `min_run_hours` is now genuinely optimized rather than held at a per-line-confounded default). Achieved service lands at 0.94–1.00 across every class on every line — the saturation finding holds at the joint optimum, not only in the isolated diagnostic. 141/141 tests pass.

**Not done this turn, stated plainly rather than left ambiguous:** the HTML dashboard (service sliders, results tables) has not yet been updated to match — it still presents service as a live decision lever. This is the next piece of work, not a completed item.


*(pending)*
