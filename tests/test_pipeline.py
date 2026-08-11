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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
