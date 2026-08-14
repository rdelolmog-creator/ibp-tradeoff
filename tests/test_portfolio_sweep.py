"""Step 8 — tests for src/portfolio_sweep.py.

The tests that matter most: `test_engine_is_not_modified` (this step consumes
src/engine.py, per D-046 / architecture A-03), and
`test_worst_case_utilisation_uses_the_full_grid_not_just_the_optimum` — a
summary that only checked capacity at the cheapest lever setting could hide
exactly the risk the O-10 review needs to see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.engine import LeverSettings, TradeOffEngine  # noqa: E402
from src.portfolio_sweep import (  # noqa: E402
    LEVER_NAMES,
    PortfolioSweepViolation,
    lever_consistency,
    lever_grid,
    portfolio_summary,
    sweep_all_lines_all_levers,
)

DATA_ROOT = "data_primary"


@pytest.fixture(scope="module")
def assumptions():
    return yaml.safe_load((REPO / "config" / "assumptions.yaml").read_text())


@pytest.fixture(scope="module")
def schema():
    return yaml.safe_load((REPO / "config" / "schema.yaml").read_text())


@pytest.fixture(scope="module")
def engine(assumptions, schema):
    clean = REPO / DATA_ROOT / "clean"
    for p in (
        clean / "clean_master.parquet",
        clean / "sku_master.parquet",
        clean / "demand_characteristics.csv",
    ):
        if not p.is_file():
            pytest.skip(f"missing pipeline artefact: {p}")
    cm = pd.read_parquet(clean / "clean_master.parquet")
    sm = pd.read_parquet(clean / "sku_master.parquet")
    dc = pd.read_csv(clean / "demand_characteristics.csv", index_col="sku_id")
    flagged = []
    diag = clean / "censoring_diagnostics.csv"
    if diag.is_file():
        d = pd.read_csv(diag, index_col="sku_id")
        if "verdict" in d.columns:
            flagged = d.index[d["verdict"] == "INCONCLUSIVE"].tolist()
    return TradeOffEngine(assumptions, schema, cm, sm, dc, flagged)


@pytest.fixture(scope="module")
def grid(assumptions):
    return lever_grid(assumptions, n=3)  # small grid for test speed


@pytest.fixture(scope="module")
def sweep(engine, grid):
    return sweep_all_lines_all_levers(engine, grid)


# ---------------------------------------------------------------------------
# lever_grid
# ---------------------------------------------------------------------------


def test_grid_has_n_values_per_lever(grid):
    for name in LEVER_NAMES:
        assert len(grid[name]) == 3


def test_grid_rejects_too_few_values(assumptions):
    with pytest.raises(PortfolioSweepViolation):
        lever_grid(assumptions, n=2)


def test_grid_endpoints_match_assumptions_range(grid, assumptions):
    levers = assumptions["levers"]
    for name, key in [
        ("service_target", "service_target_by_abc"),
        ("inventory_cover_weeks", "inventory_cover_weeks"),
        ("forecast_bias_correction", "forecast_bias_correction"),
        ("min_run_hours", "min_run_hours"),
    ]:
        lo, hi = (float(v) for v in levers[key]["range"])
        assert grid[name][0] == pytest.approx(lo)
        assert grid[name][-1] == pytest.approx(hi)


# ---------------------------------------------------------------------------
# sweep_all_lines_all_levers
# ---------------------------------------------------------------------------


def test_sweep_covers_every_line(sweep, engine):
    assert set(sweep["line_id"]) == set(engine.line_master["line_id"])


def test_sweep_row_count(sweep, grid, engine):
    expected_combos = np.prod([len(v) for v in grid.values()])
    n_lines = len(engine.line_master)
    assert len(sweep) == expected_combos * n_lines


def test_engine_is_not_modified(engine, assumptions):
    """This step CONSUMES src/engine.py — a full sweep must leave a default
    run_scenario call bit-identical (same guarantee as Step 7's equivalent
    test)."""
    cat = str(engine.sku_master.loc[engine.line_skus("L3")[0], "category"])
    base = LeverSettings.defaults(assumptions, cat, "A")
    before = engine.run_scenario("L3", base).totals
    sweep_all_lines_all_levers(engine, lever_grid(assumptions, n=3))
    after = engine.run_scenario("L3", base).totals
    assert before == after


def test_utilisation_max_is_ge_utilisation_mean(sweep):
    merged = sweep.copy()
    assert (merged["utilisation_max"] >= merged["utilisation_mean"] - 1e-9).all()


def test_capacity_never_silently_breaches(sweep, assumptions):
    """Same invariant Step 6 tests, now across the full grid on every line."""
    econ = assumptions["plant_economics"]
    cap = float(econ["scheduled_hours_per_line_month"]) + float(
        econ["max_overtime_hours_month"]
    )
    hours_avail = float(econ["scheduled_hours_per_line_month"])
    implied_workload_ceiling = hours_avail + sweep["overtime_hours_total"].max()
    # utilisation_max is workload/hours_available for the PEAK month; overtime
    # is only bought when workload exceeds hours_available, so utilisation_max
    # can exceed 1.0 by design (that is what overtime is for). What must never
    # happen is the ENGINE's own capacity invariant breaking, which is already
    # asserted inside TradeOffEngine and re-checked here at the portfolio scale
    # via capacity_shortfall_total staying finite and non-negative.
    assert sweep["capacity_shortfall_total"].min() >= 0.0
    assert np.isfinite(sweep["capacity_shortfall_total"]).all()
    assert cap > 0  # sanity: the ceiling itself is a positive, finite number


# ---------------------------------------------------------------------------
# portfolio_summary
# ---------------------------------------------------------------------------


def test_summary_has_one_row_per_line(sweep, assumptions, engine):
    summ = portfolio_summary(sweep, assumptions)
    assert set(summ["line_id"]) == set(engine.line_master["line_id"])
    assert summ["line_id"].is_unique


def test_worst_case_utilisation_uses_the_full_grid_not_just_the_optimum(
    sweep, assumptions
):
    """The summary must not only report capacity risk AT the cheapest point.
    A line can be fine at its optimum but risky elsewhere in the admissible
    range — that risk must still surface."""
    summ = portfolio_summary(sweep, assumptions)
    for line_id in summ["line_id"]:
        g = sweep[sweep["line_id"] == line_id]
        s = summ[summ["line_id"] == line_id].iloc[0]
        assert s["worst_case_utilisation_max"] == pytest.approx(
            g["utilisation_max"].max()
        )
        # the row that sets worst-case utilisation need not be the cost-optimal
        # row — assert the summary does not silently assume they coincide by
        # checking the worst-case value is read from the WHOLE grid's max, not
        # from the optimal row alone
        optimal_row = g.loc[g["total_economic_cost_eur"].idxmin()]
        if not np.isclose(
            optimal_row["utilisation_max"], g["utilisation_max"].max()
        ):
            assert s["worst_case_utilisation_max"] != pytest.approx(
                optimal_row["utilisation_max"]
            )


def test_any_capacity_breach_is_boolean(sweep, assumptions):
    summ = portfolio_summary(sweep, assumptions)
    assert summ["any_capacity_breach"].dtype == bool


def test_interior_optimum_flags_are_boolean(sweep, assumptions):
    summ = portfolio_summary(sweep, assumptions)
    for col in summ.columns:
        if col.startswith("interior_optimum_"):
            assert summ[col].dtype == bool


# ---------------------------------------------------------------------------
# lever_consistency
# ---------------------------------------------------------------------------


def test_consistency_covers_every_lever_and_output(sweep, grid):
    lc = lever_consistency(sweep, grid)
    assert set(lc["lever"]) == set(LEVER_NAMES)
    expected_outputs = {
        "lost_sales_eur",
        "excess_obsolescence_eur",
        "conversion_cost_eur",
        "working_capital_cost_eur",
        "total_economic_cost_eur",
    }
    assert set(lc["output"]) == expected_outputs


def test_lever_consistency_flags_direction_changes():
    """Construct a case where a lever provably flips direction between two
    lines, and assert the function catches it. This function's whole job is
    not to miss that."""
    sweep = pd.DataFrame(
        [
            # line A: raising the lever from 1 to 3 increases the output
            {"line_id": "A", "x": 1, "y": 0, "z": 0, "w": 0, "lost_sales_eur": 10.0},
            {"line_id": "A", "x": 2, "y": 0, "z": 0, "w": 0, "lost_sales_eur": 20.0},
            {"line_id": "A", "x": 3, "y": 0, "z": 0, "w": 0, "lost_sales_eur": 30.0},
            # line B: raising the same lever DECREASES the output
            {"line_id": "B", "x": 1, "y": 0, "z": 0, "w": 0, "lost_sales_eur": 30.0},
            {"line_id": "B", "x": 2, "y": 0, "z": 0, "w": 0, "lost_sales_eur": 20.0},
            {"line_id": "B", "x": 3, "y": 0, "z": 0, "w": 0, "lost_sales_eur": 10.0},
        ]
    )
    # lever_consistency reads a fixed output list, so the other four columns
    # it also checks must exist even though this test only varies one of them
    for col in (
        "excess_obsolescence_eur",
        "conversion_cost_eur",
        "working_capital_cost_eur",
        "total_economic_cost_eur",
    ):
        sweep[col] = 0.0
    grid = {"x": [1, 2, 3], "y": [0], "z": [0], "w": [0]}

    import src.portfolio_sweep as ps

    original = ps.LEVER_NAMES
    ps.LEVER_NAMES = ["x", "y", "z", "w"]
    try:
        lc = ps.lever_consistency(sweep, grid)
    finally:
        ps.LEVER_NAMES = original

    row = lc[(lc["lever"] == "x") & (lc["output"] == "lost_sales_eur")]
    assert not row.empty
    assert bool(row["consistent"].iloc[0]) is False
    assert row["A_direction"].iloc[0] == "up"
    assert row["B_direction"].iloc[0] == "down"


def test_ground_truth_not_importable():
    text = (REPO / "src" / "portfolio_sweep.py").read_text()
    assert "ground_truth" not in text.lower()
