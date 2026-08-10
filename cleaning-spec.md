Cleaning Specification — IBP Trade-Off Engine
Step 3. Written before any cleaning code exists. Implemented by src/ingest.py (DataIngestor) and src/cleaner.py (DataCleaner). Maps schema.yaml raw.* to schema.yaml canonical.*.

Every step states what it does, why, and what makes it fail loudly. Nothing coerces silently — per schema.validation, unexpected columns, missing columns and dtype mismatches raise.


Ordering principle
The order is a dependency chain, not a preference:

keys must be strings before any join, or padding is lost
case_size must come from sku_master before UoM conversion
stock_open must exist before null volumes can be derived
business rules can only be checked after all normalisation

Reordering C-01, C-03, C-06 or C-10 produces silently wrong output rather than an error. That is why the order is specified rather than left to the implementer.


C-01 · Read with explicit dtypes
Read all three raw CSVs with dtype forced for identifier columns: MATERIAL, item_no, source_sku_code, PLANT, site, plant_code, line_code → str. Never rely on inference.

Why. Source codes are zero-padded (01, 02). Pandas infers int64 and drops the padding. Measured consequence: the join to ground_truth.csv matches 42 of 60 SKUs on inferred dtypes and 60 of 60 on forced strings. It fails silently — no error, no warning, 30% of the portfolio quietly absent from the validation at Step 5.

Fails loudly if. Any identifier column arrives numeric after read.
C-02 · Validate raw schema on arrival
For each raw file, compare the column set against schema.raw.<table>.columns. Raise on unexpected columns, missing columns, or dtype mismatch. Do not proceed on partial success.

Why. §7 requires schema validation that fails loudly. A silently dropped column would propagate to the canonical layer as a null column and be indistinguishable from missing data.
C-03 · Build the surrogate key
sku_id = {source_system}-{source_sku_code}, e.g. A-01, B-01. System A carries source_system = "A" for PLANT1/PLANT2; system B carries "B" for PLANT3. Zero-pad codes to a consistent width before concatenation.

Why. All 15 system-B codes collide with system-A codes for different materials. Joining on the bare code silently merges unrelated SKUs — the single most damaging defect in this dataset, because the result looks plausible. Ruled at O-01: surrogate key, not a remapping table.

Fails loudly if. sku_id is not unique in sku_master, or any clean_master.sku_id has no parent in sku_master.
C-04 · Normalise dates to month start
System A: PERIOD is int YYYYMM → parse to YYYY-MM-01. System B: month_end is DD/MM/YYYY month-end → parse with format="%d/%m/%Y" explicitly, then snap to that month's first day.

Why. Two conventions for the same period. Never use dayfirst inference — 03/04/2024 is ambiguous and pandas will resolve it inconsistently across a column. Snapping to month start makes the two sources joinable and matches canonical.clean_master.month.

Fails loudly if. Any date fails to parse, or the calendar has gaps: every sku_id must have exactly schema.meta.history_months rows.
C-05 · Normalise rate scales
System A: SCHED_ADH_PCT, YIELD_PCT are 0–100 → divide by 100. System B: adherence, yield_rate are already 0–1 → leave.

Why. Same quantity, two scales. Detect by declared source rather than by inspecting values — a 0–100 column that happens to contain only small values would be misread by a range heuristic.

Fails loudly if. Any resulting rate falls outside [0, 1].
C-06 · Convert UoM to units
System A is EA (already units). System B is CS: multiply shipped, fcst_m1/m2/m3, produced, stock_eom by that SKU's case_size from sku_master. case_size varies by SKU (observed 7, 8, …), so this is a join, not a constant.

Why. schema.meta.base_uom is units. Applying a single case size, or converting before the master join, corrupts every volume on Plant 3.

Fails loudly if. Any system-B SKU has no case_size, or case_size <= 0.
C-07 · Convert currency to EUR
sku_master only. System B reports GBP per case: price_eur = price_gbp × assumptions.finance.fx.gbp_eur ÷ case_size. Same for std_cost. Also divide moq and min_run_qty by case_size — those are quantities, not money, and take no FX.

Why. The GBP values are per case, so FX and UoM must both be undone, in that order. Transaction files carry no money and need no FX — only the master does. The rate is a stated assumption with LOW confidence, not a market view.

Fails loudly if. Any price_eur <= std_cost_eur (negative margin), which would indicate FX or case-size applied twice.
C-08 · Normalise free-text category
Strip whitespace, lowercase, collapse internal spaces to underscore. Nine observed variants (AMBIENT, Ambient , ambient, Personal_care , …) must resolve to exactly three values. Validate against the keys of assumptions.categories rather than against whatever the data contains.

Why. Validating against observed values would accept a tenth variant as legitimate. The assumption file is the authority.

Fails loudly if. Distinct categories after normalisation ≠ 3, or any value is absent from assumptions.categories.
C-09 · Derive stock_open
Sort by sku_id, month; stock_open_units = stock_close_units.shift(1) within each sku_id. First month has no prior — leave null for now, resolve at C-11.
C-10 · Recover null volumes from the roll-forward identity
~4% of SALES_QTY / shipped arrive null. Do not interpolate. Derive:

actual_units = stock_open_units + production_units - stock_close_units

production_units and stock_close_units are never null. Set actual_imputed = True on every derived row.

Why. This is arithmetic, not estimation. Verified against the 1,511 rows where the actual is reported: maximum absolute error 0.000000, 100% within the 0.5-unit tolerance. Of 65 nulls, 64 are exactly recoverable; only the one first-month case is not. Interpolating would inject fabricated variance into the demand series, and D-004 exists precisely because unflagged imputation lets generator noise be attributed to a named planner role.

Fails loudly if. A derived value is negative beyond the 0.5-unit tolerance — that means the identity is broken upstream, not that demand was negative.
C-11 · Handle irrecoverable nulls
Nulls in month 1, where stock_open is unknown, cannot be derived. Drop the row, record it in the reconciliation report under rows_dropped_by_reason, and do not backfill.

Why. One row of 2,160. Fabricating an opening balance to rescue it would introduce an assumption with no basis, and the loss is immaterial. State the count, don't hide it.
C-12 · Validate business rules
After all normalisation, enforce schema.validation.business_rules: non-negativity; stock roll-forward within 0.5 units; exactly history_months rows per sku_id with no calendar gaps; capacity feasibility by line-month; referential integrity on sku_id and line_id.

Why. These are the invariants everything downstream assumes. Checking them here means a Step 6 failure is a modelling error, not a data error — which is the difference between debugging in an hour and debugging in a day.
C-13 · Emit the reconciliation and DQ report
Per schema.validation.reconciliation_report, as a DataFrame: rows_in_by_source, rows_dropped_by_reason, rows_out, null_rate_by_column, fx_applied, uom_converted_rows, sku_collisions_resolved, plus the count of rows where actual_imputed = True.

Why. §9 lists "clean DataFrame not visible" as a risk. A named df_clean, a printed before/after reconciliation and a DQ DataFrame are what make the Mastery "working Python code" criterion evidenced rather than asserted.


Known limitations of this spec
sched_adherence and yield_rate are constant in the generated data (0.92 everywhere; yield fixed per category). Cleaning cannot fix this, but it matters at Step 7: a feature with zero variance teaches the cost model nothing. Logged as O-06.
Derived actuals are censored actuals. C-10 recovers what shipped, not what was demanded (D-014). The roll-forward identity cannot see lost sales, by construction. Step 5 must handle censoring separately.
The 0.5-unit tolerance is inherited from schema.validation, chosen to absorb the generator's half-unit rounding. It is not a statement about real-world measurement error.
