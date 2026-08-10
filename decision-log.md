Decision Log — IBP Trade-Off Engine
One line per decision, in the form DECISION: X, because Y. Feeds the limitations slide and the pitch. Append as we go; never rewrite history — if a decision is reversed, add a new entry that supersedes it.


Step 0 — Foundations
D-001 · Schema layering DECISION: schema.yaml defines two layers — raw.* (as extracted, defects included) and canonical.* (post-clean contract) — because the cleaner's job is precisely the mapping between them, and writing it down makes the cleaning spec testable rather than narrative.

D-002 · Raw file count DECISION: the generator emits three raw CSVs — plant_system_A.csv, plant_system_B.csv, sku_master.csv — rather than the two named in §5, because margin and standard cost are SKU-level data, and holding 60 SKUs of master data in assumptions.yaml would breach the project's own no-hard-coded-SKU-list rule. Supersedes the §5 Step 2 output list.

D-003 · Clean master grain DECISION: clean_master is (sku_id, month) with forecast lags pivoted to columns (forecast_l1/l2/l3), because at ~2,160 rows long format triples table size for no analytical gain, and the bias decomposition reads lags side by side. Known cost: adding a fourth lag later is a schema change, not a data change.

D-004 · Imputation provenance DECISION: every imputed value carries a boolean provenance flag (actual_imputed), because ~4% of volume arrives null, and unflagged imputation would let generator noise be attributed to a named planner role — silently breaking the accountability claim the tool exists to make.

D-005 · Scalability answer (§11) DECISION: the first thing to break under volume is per-SKU scenario computation, not model training, because scenarios are evaluated combinatorially across levers while the cost model trains once. Mitigations: vectorisation, caching of unchanged legs, parallel scenario evaluation.

D-006 · PII answer (§11) DECISION: data is fully synthetic and accountability is assigned at role level, not to named individuals, because the framework's output is an attribution of chronic bias — and attributing that to a person rather than a role turns a diagnostic tool into a performance-management weapon.

D-007 · ROI answer (§11) DECISION: ROI is stated as analyst hours per S&OP cycle × cycles per year, with the hours assumption stated openly, because any single headline number here would be unsupportable and inviting challenge on it costs more credibility than it buys.

D-008 · Category differentiation DECISION: the three categories are ambient, chilled and personal care, with deliberately different shelf life (65 / 6 / 156 weeks) and margin (33% / 29% / 52%), because homogeneous categories would leave the obsolescence leg of the trade-off inert and produce a four-output model where one output never moves.

D-009 · Chronic bias structure DECISION: chronic forecast bias is drawn once at category level and held across all 36 months, with SKU-level idiosyncratic variation around it, because bias redrawn monthly is random error wearing a sign — the decomposition would correctly find nothing to attribute and the accountability claim would collapse.

D-010 · Bias control group DECISION: ambient carries zero chronic bias while chilled carries −7% and personal care +5%, because a control category demonstrates the model discriminates rather than manufacturing an attribution everywhere it looks.

D-011 · MAPE as validation, not input DECISION: forecast MAPE is a validation target rather than a generator parameter, because MAPE is a consequence of chronic bias plus random error, and specifying all three independently over-determines the system and silently distorts one of them.

D-012 · Capacity calibration DECISION: baseline line utilisation is 85% and line speeds are taken at the bottom of the proposed EMEA ranges, because capacity must bind for the MOQ and minimum-run levers to move any output, and the parameter table's central values implied a €700m business against a €220k/line absorption base — conversion cost would have landed near 2% of COGS against a 12–25% norm.


Closed — rulings made



Step 1 — Generator design
Generator reviewed against the §9 circularity risk before execution. Forecasts are built from the deterministic demand signal only — never from realised demand — and chronic bias is drawn once per SKU and held across all 36 months. No coefficient anywhere maps forecast error to cost or to write-off. Reviewed, corrected, then run: net sales €196.9m vs €200m target, conversion cost 12.7% of COGS, stock roll-forward ties to zero, MAPE inside range for all three ABC classes at lag 1.

D-013 · Capacity binds with a shortfall signal DECISION: overtime is capped at plant_economics.max_overtime_hours_month and any residual excess scales scheduled runs down proportionally, recording capacity_shortfall_units, because unbounded overtime meant capacity never physically bound and the minimum-run lever produced only a cost signal, never a shortage.

D-014 · Exported actuals are censored DECISION: raw files export shipped volume censored at available stock, and latent demand exists only in _truth/, because a planning system records what shipped, not what was wanted — with the known consequence that chronic bias is under-recovered on stocked-out SKUs.

D-015 · Bias must be estimated in log space DECISION: Step 5 estimates chronic bias as the mean of log(forecast / actual), not (forecast − actual) / actual, because demand noise CV exceeds forecast noise CV in every ABC class and the naive ratio shifts the estimate upward by (σd² − σf²) / 2 — measured at +1.8pp, enough to report +4.5% chronic bias in the control category that was encoded with zero.

D-016 · Ground truth carries raw-layer keys DECISION: ground_truth.csv carries source_system, source_sku_code and surrogate_sku_id alongside the internal sku_id, because the internal id appears in no raw file by design, and a truth file that cannot be joined cannot score the decomposition it exists to validate. The raw CSVs remain free of the internal id.

D-017 · Two datasets, not one tuned dataset DECISION: bias recovery is validated against two generated datasets — the primary at ~6.2% censoring and a control at ~1.6% — rather than tuning assumptions.yaml until recovery looks clean, because parameter selection until the answer appears is the circularity failure in another costume (§9). The control varies abc.*.target_cover_weeks only (12/12/14 vs 4/4/6); demand_noise_cv was deliberately left untouched because it drives both censoring and the variance asymmetry, and changing it would confound the two effects the control exists to separate.
Open at Step 1 close



Step 2 — Raw data + defect injection
(pending)
