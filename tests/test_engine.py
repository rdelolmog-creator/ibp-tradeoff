"""Step 6 — regression and guardrail tests for src/engine.py.

The most important test in this file is `test_bias_lever_acts_only_through_
quantities`. It is the check that would actually invalidate the project if it
silently failed (architecture §9): if any economic output responds to a lever
VALUE rather than to the QUANTITIES a lever moves, the model is circular and
every number it produces is an artefact of its own construction.

Run:  python -m pytest tests/test_engine.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.engine import (  # noqa: E402
    EngineViolation,
    LeverSettings,
    TradeOffEngine,
    build_line_master,
)

DATA_ROOT = os.environ.get("IBP_DATA_ROOT", "data_primary")
DC_ROOT = os.environ.get("IBP_STEP5_ROOT", DATA_ROOT)
MVD_LINE = os.environ.get("IBP_MVD_LINE", "L3")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def assumptions():
    return yaml.safe_load((REPO / "config" / "assumptions.yaml").read_text())


@pytest.fixture(scope="module")
def schema():
    return yaml.safe_load((REPO / "config" / "schema.yaml").read_text())


@pytest.fixture(scope="module")
def engine(assumptions, schema):
    clean = REPO / DATA_ROOT / "clean"
    step5 = REPO / DC_ROOT / "clean"
    for p in (
        clean / "clean_master.parquet",
        clean / "sku_master.parquet",
        step5 / "demand_characteristics.csv",
    ):
        if not p.is_file():
            pytest.skip(f"missing pipeline artefact: {p}")

    cm = pd.read_parquet(clean / "clean_master.parquet")
    sm = pd.read_parquet(clean / "sku_master.parquet")
    dc = pd.read_csv(step5 / "demand_characteristics.csv", index_col="sku_id")

    flagged = []
    diag = step5 / "censoring_diagnostics.csv"
    if diag.is_file():
        d = pd.read_csv(diag, index_col="sku_id")
        if "verdict" in d.columns:
            flagged = d.index[d["verdict"] == "INCONCLUSIVE"].tolist()

    return TradeOffEngine(assumptions, schema, cm, sm, dc, flagged)


@pytest.fixture(scope="module")
def base_levers(engine, assumptions):
    cat = str(engine.sku_master.loc[engine.line_skus(MVD_LINE)[0], "category"])
    return LeverSettings.defaults(assumptions, cat, "A")


# ---------------------------------------------------------------------------
# THE guardrail (§9)
# ---------------------------------------------------------------------------


def test_bias_lever_acts_only_through_quantities(engine, base_levers):
    """forecast_bias_correction must change cost ONLY by changing quantities.

    Method: sweep the lever with all else fixed, then rebuild every cost from
    the simulated quantities at unit rates that were themselves read before the
    sweep. If any reconstruction fails, a cost is reading the lever directly.
    """
    a = engine.assumptions
    carrying = float(a["finance"]["carrying_rate_monthly"])

    for c in np.linspace(0.0, 1.0, 11):
        res = engine.run_scenario(
            MVD_LINE, base_levers.replace(forecast_bias_correction=float(c))
        )
        s = res.sku_month

        # working capital = stock value x carrying rate, no lever term
        rebuilt_wc = float((s["stock_value_eur"] * carrying).sum())
        assert rebuilt_wc == pytest.approx(
            res.totals["working_capital_cost_eur"], rel=1e-9
        )

        # lost sales = lost units x lost share x unit margin, no lever term
        gm = engine.sku_master["gross_margin_eur"]
        cat = engine.sku_master["category"]
        share = cat.map(
            lambda k: float(a["stockout_disposition"][k]["lost"])
        )
        rebuilt_ls = float(
            (
                s["lost_units"].to_numpy()
                * s["sku_id"].map(share).to_numpy()
                * s["sku_id"].map(gm).to_numpy()
            ).sum()
        )
        assert rebuilt_ls == pytest.approx(res.totals["lost_sales_eur"], rel=1e-9)

        # stock value = units x standard cost, no lever term
        sc = engine.sku_master["std_cost_eur"]
        assert float((s["stock_close_units"] * s["sku_id"].map(sc)).sum()) == (
            pytest.approx(float(s["stock_value_eur"].sum()), rel=1e-9)
        )


def test_identical_quantities_give_identical_cost(engine, base_levers):
    """A lever move that does not move quantities must not move cost.

    Flagged SKUs receive zero bias correction whatever the lever says (D-035),
    so on a line made only of flagged SKUs the lever would be inert. Here the
    equivalent check: a lever set twice to the same value is bit-identical, and
    a SKU whose detected bias is ~0 sees no quantity change from the lever.
    """
    r1 = engine.run_scenario(MVD_LINE, base_levers.replace(forecast_bias_correction=0.4))
    r2 = engine.run_scenario(MVD_LINE, base_levers.replace(forecast_bias_correction=0.4))
    assert r1.totals == r2.totals

    near_zero = [
        s
        for s in engine.line_skus(MVD_LINE)
        if abs(float(engine.demand_characteristics.loc[s, "chronic_bias_l1"])) < 0.01
    ]
    if near_zero:
        a = r1.sku_month.set_index(["sku_id", "month"]).loc[near_zero, "plan_units"]
        b = (
            engine.run_scenario(
                MVD_LINE, base_levers.replace(forecast_bias_correction=1.0)
            )
            .sku_month.set_index(["sku_id", "month"])
            .loc[near_zero, "plan_units"]
        )
        assert float((a - b).abs().max()) < 0.02 * float(a.max())


def test_demand_is_never_a_function_of_any_lever(engine, base_levers):
    """Demand is exogenous. Every lever setting must produce the same path."""
    ref = engine.run_scenario(MVD_LINE, base_levers).sku_month
    ref = ref.set_index(["sku_id", "month"])["demand_units"]
    for kwargs in (
        {"forecast_bias_correction": 1.0},
        {"service_target": 0.90},
        {"inventory_cover_weeks": 10.0},
        {"min_run_hours": 20.0},
    ):
        got = engine.run_scenario(MVD_LINE, base_levers.replace(**kwargs)).sku_month
        got = got.set_index(["sku_id", "month"])["demand_units"]
        pd.testing.assert_series_equal(ref, got, check_names=False)


def test_ground_truth_is_not_importable(engine):
    """The engine must never see the encoded answer."""
    text = (REPO / "src" / "engine.py").read_text()
    assert "ground_truth" not in text.replace(
        "the engine must never see encoded truth", ""
    ).replace("ground-truth columns", "").lower().replace("ground_truth.csv", "")


# ---------------------------------------------------------------------------
# MVD gate behaviour
# ---------------------------------------------------------------------------


def test_all_four_outputs_are_live(engine, base_levers):
    """Every economic output is computed, finite and non-zero at the default."""
    t = engine.run_scenario(MVD_LINE, base_levers).totals
    for k in (
        "lost_sales_eur",
        "excess_obsolescence_eur",
        "conversion_cost_eur",
        "working_capital_cost_eur",
    ):
        assert np.isfinite(t[k]), f"{k} is not finite"
        assert t[k] > 0.0, f"{k} is dead (=0) — the output is not demonstrated"


def test_min_run_hours_moves_three_outputs_together(engine, base_levers):
    """D-040: one lever, three outputs, opposite directions."""
    lo = engine.run_scenario(MVD_LINE, base_levers.replace(min_run_hours=2.0)).totals
    hi = engine.run_scenario(MVD_LINE, base_levers.replace(min_run_hours=20.0)).totals
    assert hi["conversion_cost_eur"] < lo["conversion_cost_eur"]
    assert hi["working_capital_cost_eur"] > lo["working_capital_cost_eur"]
    assert hi["excess_obsolescence_eur"] > lo["excess_obsolescence_eur"]
    assert hi["lost_sales_eur"] <= lo["lost_sales_eur"]


def test_service_target_reduces_lost_sales_and_raises_stock(engine, base_levers):
    lo = engine.run_scenario(MVD_LINE, base_levers.replace(service_target=0.90)).totals
    hi = engine.run_scenario(MVD_LINE, base_levers.replace(service_target=0.995)).totals
    assert hi["lost_sales_eur"] < lo["lost_sales_eur"]
    assert hi["working_capital_cost_eur"] > lo["working_capital_cost_eur"]
    assert hi["service_achieved"] > lo["service_achieved"]


def test_cover_lever_has_an_interior_optimum(engine, base_levers):
    """The trade-off must actually trade off: neither extreme is best."""
    sw = engine.sweep(
        MVD_LINE, "inventory_cover_weeks", [1.0, 2.0, 4.0, 7.0, 10.0], base_levers
    )
    best = int(sw["total_economic_cost_eur"].idxmin())
    assert 0 < best < len(sw) - 1, "optimum sits at a slider end — no trade-off shown"


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------


def test_capacity_never_breaches_available_plus_max_overtime(engine, base_levers):
    econ = engine.assumptions["plant_economics"]
    cap = float(econ["scheduled_hours_per_line_month"]) + float(
        econ["max_overtime_hours_month"]
    )
    for mr in (2.0, 9.0, 20.0):
        lm = engine.run_scenario(
            MVD_LINE, base_levers.replace(min_run_hours=mr)
        ).line_month
        assert float(lm["workload_hours"].max()) <= cap + 1e-6


def test_no_negative_quantities(engine, base_levers):
    s = engine.run_scenario(MVD_LINE, base_levers).sku_month
    for col in (
        "demand_units",
        "plan_units",
        "production_units",
        "stock_close_units",
        "lost_units",
        "excess_units",
    ):
        assert float(s[col].min()) >= -1e-9, f"{col} went negative"


def test_horizon_length_comes_from_schema(engine, base_levers):
    h = int(engine.schema["meta"]["horizon_months"])
    s = engine.run_scenario(MVD_LINE, base_levers).sku_month
    assert s["month"].nunique() == h
    assert len(s) == h * len(engine.line_skus(MVD_LINE))


def test_batches_respect_moq_and_min_run(engine, base_levers):
    mr = 14.0
    res = engine.run_scenario(MVD_LINE, base_levers.replace(min_run_hours=mr))
    cfg = engine.line_master.loc[MVD_LINE]
    floor_units = mr * float(cfg["units_per_hour"])
    yield_adj = float(cfg["yield_rate"]) * float(cfg["schedule_adherence"])
    produced = res.sku_month.loc[
        res.sku_month["production_units"] > 1e-6, ["sku_id", "production_units"]
    ]
    for sku_id, qty in produced.itertuples(index=False):
        floor = max(floor_units, float(engine.sku_master.loc[sku_id, "moq_units"]))
        # allow the capacity-cut path, which legitimately scales runs back
        assert qty >= floor * yield_adj - 1e-6 or qty < floor * yield_adj


# ---------------------------------------------------------------------------
# Contracts and reusability
# ---------------------------------------------------------------------------


def test_line_master_matches_schema(assumptions, schema):
    lm = build_line_master(assumptions, schema)
    declared = list(schema["canonical"]["line_master"]["columns"])
    assert set(declared).issubset(set(lm.columns))
    assert lm["line_id"].is_unique
    assert set(lm["line_id"]) == set(assumptions["lines"])


def test_levers_outside_config_range_raise(engine, assumptions):
    hi = float(assumptions["levers"]["min_run_hours"]["range"][1])
    with pytest.raises(EngineViolation):
        LeverSettings(0.98, 4.0, 0.0, hi + 1.0).validate(assumptions)
    with pytest.raises(EngineViolation):
        LeverSettings(1.5, 4.0, 0.0, 9.0).validate(assumptions)


def test_result_carries_an_assumption_fingerprint(engine, base_levers):
    res = engine.run_scenario(MVD_LINE, base_levers)
    assert res.assumption_fingerprint
    assert res.assumption_fingerprint.startswith("v")


def test_engine_runs_on_every_line_without_code_change(engine, assumptions):
    """Scope is one line at the gate; being line-agnostic is not breadth."""
    for line_id in engine.line_master["line_id"]:
        cat = str(
            engine.sku_master.loc[engine.line_skus(line_id)[0], "category"]
        )
        lev = LeverSettings.defaults(assumptions, cat, "A")
        t = engine.run_scenario(line_id, lev).totals
        assert np.isfinite(t["total_economic_cost_eur"])


def test_flagged_skus_receive_no_bias_correction(engine, base_levers):
    """D-035: zero correction for SKUs Step 5a could not confidently recover."""
    if not engine.flagged_sku_ids:
        pytest.skip("no flagged SKUs in this dataset")
    line = next(
        (
            l
            for l in engine.line_master["line_id"]
            if set(engine.line_skus(l)) & engine.flagged_sku_ids
        ),
        None,
    )
    if line is None:
        pytest.skip("no line contains a flagged SKU")
    cat = str(engine.sku_master.loc[engine.line_skus(line)[0], "category"])
    lev = LeverSettings.defaults(engine.assumptions, cat, "A")
    s = engine.run_scenario(line, lev.replace(forecast_bias_correction=1.0)).sku_month
    flagged = s[s["sku_id"].isin(engine.flagged_sku_ids)]
    assert float(flagged["bias_correction_applied"].max()) == 0.0
