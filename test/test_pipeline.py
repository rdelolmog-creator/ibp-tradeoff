"""
Regression tests for src/ingest.py and src/cleaner.py.

Purpose: answer "does this still work if we run it again next month" with a
yes/no, not a manual re-read of console output. Two kinds of test:

  1. REGRESSION  — the real generated data still produces the same shape of
     result (row counts, recovery counts) it did when Step 4 was signed off.
  2. ROBUSTNESS  — deliberately corrupted / novel-defect fixtures, to prove
     the pipeline fails LOUDLY on things it wasn't built for, rather than
     silently producing wrong output. This is the direct answer to
     "what happens next month with an error type not in the spec".

Run from repo root:  python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from src.ingest import DataIngestor, SchemaViolation
from src.cleaner import DataCleaner, CleaningViolation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def ingestor():
    return DataIngestor(repo_root=REPO, data_root="data_primary")


@pytest.fixture(scope="module")
def raw(ingestor):
    return ingestor.load()


@pytest.fixture(scope="module")
def cleaned(ingestor, raw):
    cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
    df, master, dq = cleaner.clean(raw)
    return df, master, dq, cleaner


# ---------------------------------------------------------------------------
# REGRESSION — known-good numbers, so a future change that alters them fails
# the suite instead of silently shipping.
# ---------------------------------------------------------------------------

class TestRegressionAgainstKnownGood:
    def test_row_counts(self, cleaned):
        df, master, dq, cleaner = cleaned
        assert len(df) == 2159
        assert df.sku_id.nunique() == 60
        assert df.month.nunique() == 36

    def test_null_recovery(self, cleaned):
        df, master, dq, cleaner = cleaned
        d = dict(zip(dq.metric, dq.value))
        assert int(d["nulls_in_volume"]) == 87
        assert int(d["nulls_recovered_by_identity"]) == 86
        assert sum(x["rows"] for x in cleaner.dropped) == 1

    def test_rollforward_within_tolerance(self, cleaned):
        df, master, dq, cleaner = cleaned
        d = dict(zip(dq.metric, dq.value))
        assert d["stock_rollforward_max_abs_diff"] < 0.5

    def test_no_orphan_skus(self, cleaned):
        df, master, dq, cleaner = cleaned
        assert set(df.sku_id).issubset(set(master.sku_id))

    def test_categories_collapse_to_three(self, cleaned):
        df, master, dq, cleaner = cleaned
        assert master.category.nunique() == 3

    def test_identical_result_on_control_dataset(self, ingestor):
        """The reusability claim: same code, different data_root, same shape
        of result. This is what Step 4 is supposed to demonstrate."""
        ing2 = DataIngestor(repo_root=REPO, data_root="data_control")
        raw2 = ing2.load()
        cl2 = DataCleaner(ing2.schema, ing2.assumptions)
        df2, master2, dq2 = cl2.clean(raw2)
        assert len(df2) == 2159
        assert df2.sku_id.nunique() == 60


# ---------------------------------------------------------------------------
# ROBUSTNESS — novel/corrupted inputs the spec never saw. Each test proves
# the pipeline raises loudly rather than producing silently wrong output.
# ---------------------------------------------------------------------------

class TestRobustnessToUnseenDefects:

    def test_unexpected_new_column_raises(self, ingestor, raw):
        bad = raw["system_a"].copy()
        bad["WAREHOUSE_CODE"] = "WH1"
        with pytest.raises(SchemaViolation):
            ingestor._validate_columns("system_a", bad)

    def test_missing_expected_column_raises(self, ingestor, raw):
        bad = raw["system_a"].drop(columns=["PROMO_FLAG"])
        with pytest.raises(SchemaViolation):
            ingestor._validate_columns("system_a", bad)

    def test_currency_drift_raises(self, ingestor, raw):
        """A source system silently starts reporting a new currency. This is
        the gap identified in review: a declared 'constant' column changing
        value still parses fine as a string and must be caught explicitly."""
        bad = raw["system_b"].copy()
        bad.loc[0, "ccy"] = "USD"
        with pytest.raises(SchemaViolation):
            ingestor._validate_columns("system_b", bad)

    def test_unparseable_date_format_raises(self, ingestor, raw):
        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        bad = raw["system_b"].copy()
        bad.loc[0, "month_end"] = "2023-09-30"  # ISO instead of DD/MM/YYYY
        with pytest.raises(Exception):
            cleaner._month_from_ddmmyyyy(bad["month_end"])

    def test_rate_outside_unit_interval_raises(self, ingestor, raw):
        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        bad = pd.DataFrame({
            "sched_adherence": [1.4],  # would come from a source no longer
                                       # on the 0-100 scale being divided by 100
            "yield_rate": [0.98],
        })
        with pytest.raises(CleaningViolation):
            cleaner._check_rates(bad, "synthetic_bad_source")

    def test_null_rate_spike_is_visible_not_silent(self, ingestor, raw):
        """Simulates next month's data arriving with far more nulls than the
        ~4% the spec was tuned around. The pipeline must not crash AND must
        not silently absorb it — it should show up honestly in dropped rows
        or the DQ report, large enough that a human notices."""
        bad_a = raw["system_a"].copy()
        rng = np.random.default_rng(0)
        # null out 40% instead of ~4%
        idx = rng.choice(bad_a.index, size=int(len(bad_a) * 0.4), replace=False)
        bad_a.loc[idx, "SALES_QTY"] = np.nan

        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        a_clean = cleaner._normalise_system_a(bad_a)
        a_clean = cleaner._derive_stock_open(a_clean)
        n_before = a_clean["actual_units"].isna().sum()
        recovered = cleaner._recover_nulls(a_clean.copy())
        # every dropped row must be counted, not silently lost
        assert sum(d["rows"] for d in cleaner.dropped) > 1  # more than baseline
        assert n_before > 0.3 * len(a_clean)  # confirms the injected spike was real

    def test_negative_case_size_raises(self, ingestor, raw):
        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        bad_master = raw["sku_master"].copy()
        bad_master["case_size"] = bad_master["case_size"].astype(object)
        bad_master.loc[bad_master.index[0], "case_size"] = -1
        with pytest.raises(CleaningViolation):
            cleaner._clean_master(bad_master)

    def test_unknown_category_raises(self, ingestor, raw):
        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        bad_master = raw["sku_master"].copy()
        bad_master.loc[0, "category"] = "FROZEN"  # a 4th category, never
                                                    # seen at Step 0/1
        with pytest.raises(CleaningViolation):
            cleaner._clean_master(bad_master)

    def test_double_null_not_falsely_counted_as_recovered(self, ingestor, raw):
        """Step 4 review, item 2. If BOTH actual_units and stock_close_units
        are null on the same row, the roll-forward identity cannot derive a
        value (NaN + x - NaN = NaN). This must NOT be counted as a successful
        recovery, and the row must be dropped, not silently left null."""
        bad_a = raw["system_a"].copy()
        sub_idx = bad_a[(bad_a.MATERIAL == "01") & (bad_a.PLANT == "PLANT1")].index
        target = sub_idx[3]  # a mid-series month, not the first
        bad_a.loc[target, "SALES_QTY"] = None
        bad_a.loc[target, "STOCK_CLOSE"] = None

        cleaner_good = DataCleaner(ingestor.schema, ingestor.assumptions)
        good = cleaner_good.clean(raw)[0]
        baseline_recovered = int(cleaner_good.stats["nulls_recovered_by_identity"])

        cleaner_bad = DataCleaner(ingestor.schema, ingestor.assumptions)
        bad_raw = {**raw, "system_a": bad_a}
        cleaner_bad.clean(bad_raw)
        # recovered count must NOT increase — the double-null row cannot be
        # genuinely recovered, so it must fall through to being dropped
        assert cleaner_bad.stats["nulls_recovered_by_identity"] == baseline_recovered
        assert sum(d["rows"] for d in cleaner_bad.dropped) == 2  # baseline 1 + this new one

    def test_cascade_failure_classified_separately_from_first_month(self, ingestor, raw):
        """Step 4 review, item 2. A null stock_close in month N nulls
        stock_open in month N+1 via the shift. If month N+1 ALSO has a null
        actual, it cannot be recovered — and must be classified as a cascade
        failure, NOT bucketed under 'expected first-month edge case', even
        though both cases have stock_open_units == NaN. Distinguishing them
        requires calendar position, not just whether stock_open is null."""
        bad_a = raw["system_a"].copy()
        sub_idx = bad_a[(bad_a.MATERIAL == "03") & (bad_a.PLANT == "PLANT1")].index
        bad_a.loc[sub_idx[4], "STOCK_CLOSE"] = None  # breaks month 5's close
        bad_a.loc[sub_idx[5], "SALES_QTY"] = None    # month 6 now unrecoverable

        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        cleaner.clean({**raw, "system_a": bad_a})

        reasons = [d["reason"] for d in cleaner.dropped]
        assert any("cascad" in r.lower() for r in reasons), (
            "A genuine mid-series cascade failure must be labelled as a "
            "cascade, not silently folded into the first-month bucket."
        )
        # confirm the cascade entry names the correct SKU/month, not month 1
        cascade_entries = [d for d in cleaner.dropped if "cascad" in d["reason"].lower()]
        cascade_skus = {row["sku_id"] for d in cascade_entries for row in d["detail"]}
        assert "A-03" in cascade_skus

    def test_fx_rate_applied_column_present_and_correct(self, ingestor, raw):
        """Step 4 review, item 4. The review workbook must be able to show
        the exact FX rate used per row, not just the abstract rate in
        assumptions.yaml. System A needs no conversion (NaN); System B must
        show the configured gbp_eur rate."""
        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        _, master, _ = cleaner.clean(raw)
        assert "fx_rate_applied" in master.columns
        expected_fx = float(ingestor.assumptions["finance"]["fx"]["gbp_eur"])
        b_rates = master.loc[master.source_system == "B", "fx_rate_applied"]
        a_rates = master.loc[master.source_system == "A", "fx_rate_applied"]
        assert (b_rates == expected_fx).all()
        assert a_rates.isna().all()

    def test_dropped_rows_have_individual_detail(self, ingestor, raw):
        """Step 4 review, item 5. Every dropped-row entry must name the
        specific sku_id and month, not just an aggregate count — a reviewer
        cannot investigate 'N rows dropped' without knowing which N."""
        cleaner = DataCleaner(ingestor.schema, ingestor.assumptions)
        cleaner.clean(raw)
        assert len(cleaner.dropped) > 0
        for entry in cleaner.dropped:
            assert "detail" in entry
            assert len(entry["detail"]) == entry["rows"]
            for row in entry["detail"]:
                assert "sku_id" in row and "month" in row


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
