"""Step 9 — tests for src/reporter.py.

The tests that matter most: `test_avoidable_and_reported_totals_both_present`
(the exact omission D-062 exists to prevent) and
`test_capacity_warning_does_not_hardcode_a_line_name` (proves the D-070
warning generalises rather than encoding today's one finding as a permanent
special case).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.engine import LeverSettings, TradeOffEngine  # noqa: E402
from src.reporter import (  # noqa: E402
    line_policy_brief_three_lever,
    ReporterViolation,
    avoidable_cost_view,
    capacity_warning,
    export_artefacts,
    owner_view,
    line_policy_brief,
    class_breakdown_view,
    portfolio_brief,
    require_artefact,
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
def scenario_result(engine, assumptions):
    cat = str(engine.sku_master.loc[engine.line_skus("L3")[0], "category"])
    base = LeverSettings.defaults(assumptions, cat, "A")
    return engine.run_scenario("L3", base)


@pytest.fixture
def portfolio_summary_row():
    return pd.Series(
        {
            "line_id": "L2",
            "best_inventory_cover_weeks": 4.0,
            "worst_case_utilisation_max": 1.15,
            "worst_case_overtime_hours": 720.0,
            "any_capacity_breach": True,
        }
    )


# ---------------------------------------------------------------------------
# avoidable_cost_view — D-062
# ---------------------------------------------------------------------------


def test_avoidable_and_reported_totals_both_present(scenario_result, assumptions, schema):
    """The exact omission D-062 exists to prevent."""
    av = avoidable_cost_view(scenario_result, assumptions, schema)
    assert "total_avoidable_cost_eur" in av
    assert "total_reported_cost_eur" in av
    assert av["total_reported_cost_eur"] > av["total_avoidable_cost_eur"]
    assert av["total_reported_cost_eur"] == pytest.approx(
        av["total_avoidable_cost_eur"] + av["conversion_cost_fixed_eur"]
    )


def test_fixed_absorption_read_from_assumptions_not_hardcoded(
    scenario_result, assumptions, schema
):
    import copy

    a1 = avoidable_cost_view(scenario_result, assumptions, schema)

    modified = copy.deepcopy(assumptions)
    modified["plant_economics"]["fixed_absorption_eur_line_month"] = (
        float(assumptions["plant_economics"]["fixed_absorption_eur_line_month"]) * 1.05
    )
    a2 = avoidable_cost_view(scenario_result, modified, schema)

    assert a2["conversion_cost_fixed_eur"] == pytest.approx(
        a1["conversion_cost_fixed_eur"] * 1.05
    )
    delta_fixed = a2["conversion_cost_fixed_eur"] - a1["conversion_cost_fixed_eur"]
    assert a2["conversion_cost_avoidable_eur"] == pytest.approx(
        a1["conversion_cost_avoidable_eur"] - delta_fixed
    )


def test_avoidable_totals_reconstruct_from_components(scenario_result, assumptions, schema):
    av = avoidable_cost_view(scenario_result, assumptions, schema)
    rebuilt = (
        av["lost_sales_eur"]
        + av["excess_obsolescence_eur"]
        + av["working_capital_cost_eur"]
        + av["conversion_cost_avoidable_eur"]
    )
    assert rebuilt == pytest.approx(av["total_avoidable_cost_eur"], rel=1e-9)


def test_carries_assumption_fingerprint(scenario_result, assumptions, schema):
    av = avoidable_cost_view(scenario_result, assumptions, schema)
    assert av["assumption_fingerprint"] == scenario_result.assumption_fingerprint


# ---------------------------------------------------------------------------
# capacity_warning — D-070
# ---------------------------------------------------------------------------


def test_capacity_warning_fires_near_the_known_breach_condition(portfolio_summary_row):
    msg = capacity_warning("L2", 10.0, portfolio_summary_row)
    assert msg is not None
    assert "L2" in msg


def test_capacity_warning_silent_at_the_lines_own_optimum(portfolio_summary_row):
    assert capacity_warning("L2", 4.0, portfolio_summary_row) is None


def test_capacity_warning_silent_when_no_breach_recorded():
    row = pd.Series(
        {
            "line_id": "L3",
            "best_inventory_cover_weeks": 4.0,
            "worst_case_utilisation_max": 0.7,
            "worst_case_overtime_hours": 0.0,
            "any_capacity_breach": False,
        }
    )
    assert capacity_warning("L3", 10.0, row) is None


def test_capacity_warning_does_not_hardcode_a_line_name():
    """Proves the logic generalises rather than special-casing 'L2' as a
    string — the whole point of D-070's warning design."""
    row = pd.Series(
        {
            "line_id": "Z9",
            "best_inventory_cover_weeks": 2.0,
            "worst_case_utilisation_max": 1.3,
            "worst_case_overtime_hours": 900.0,
            "any_capacity_breach": True,
        }
    )
    msg = capacity_warning("Z9", 8.0, row)
    assert msg is not None
    assert "Z9" in msg
    assert "L2" not in msg


# ---------------------------------------------------------------------------
# owner_view
# ---------------------------------------------------------------------------


def test_owner_view_uses_config_mapping_not_inline_guess(
    scenario_result, assumptions, schema
):
    av = avoidable_cost_view(scenario_result, assumptions, schema)
    ov = owner_view(av, assumptions)
    assert set(ov["owner_role"]) <= set(assumptions["owner_mapping"].values())
    assert len(ov) > 0


def test_owner_view_raises_without_config_mapping(scenario_result, assumptions, schema):
    import copy

    av = avoidable_cost_view(scenario_result, assumptions, schema)
    stripped = copy.deepcopy(assumptions)
    stripped.pop("owner_mapping", None)
    with pytest.raises(ReporterViolation):
        owner_view(av, stripped)


def test_owner_mapping_is_role_level_not_named_individuals(assumptions):
    """architecture §11: PII answer is role-level accountability."""
    for owner in assumptions["owner_mapping"].values():
        assert isinstance(owner, str) and len(owner) > 0
        # a weak but real check: role titles read as titles, not "John Smith"
        assert any(
            kw in owner for kw in ("Lead", "Manager", "Planner", "Finance", "Director")
        ), f"{owner!r} does not read as a role title"


# ---------------------------------------------------------------------------
# require_artefact
# ---------------------------------------------------------------------------


def test_missing_upstream_artefact_raises_with_the_producing_step_named(tmp_path):
    missing = str(tmp_path / "optimal_policy.csv")
    with pytest.raises(ReporterViolation) as exc:
        require_artefact(missing)
    assert "Step 7" in str(exc.value)


def test_present_artefact_passes_through(tmp_path):
    p = tmp_path / "clean_master.parquet"
    p.write_text("x")
    assert require_artefact(str(p)) == str(p)


# ---------------------------------------------------------------------------
# line_policy_brief / class_breakdown_view / portfolio_brief
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_line_results():
    """One row per line — the search_all_lines() output shape."""
    return pd.DataFrame(
        {
            "line_id": ["L1", "L2"],
            "cover_A": [3.5, 2.0], "cover_B": [2.5, 4.0], "cover_C": [1.5, 6.0],
            "service_A": [0.995, 0.96], "service_B": [0.95, 0.94], "service_C": [0.88, 0.90],
            "total_economic_cost_eur": [380000.0, 500000.0],
            "default_total_cost_eur": [409000.0, 550000.0],
            "saving_eur": [29000.0, 50000.0],
        }
    )


@pytest.fixture
def toy_sku_level():
    """The expand_to_sku_level() output shape."""
    return pd.DataFrame(
        {
            "sku_id": ["S1", "S2", "S3", "S4"],
            "line_id": ["L1", "L1", "L1", "L2"],
            "abc_class": ["A", "B", "C", "A"],
            "cover_weeks": [3.5, 2.5, 1.5, 2.0],
            "service_target": [0.995, 0.95, 0.88, 0.96],
            "lost_sales_eur": [100.0, 200.0, 300.0, 150.0],
            "excess_obsolescence_eur": [50.0, 80.0, 400.0, 60.0],
            "working_capital_cost_eur": [70.0, 90.0, 60.0, 40.0],
            "line_conversion_cost_eur": [160000.0, 160000.0, 160000.0, 200000.0],
        }
    )


def test_line_policy_brief_has_one_row_per_line_per_class(toy_line_results):
    lpb = line_policy_brief(toy_line_results)
    assert len(lpb) == len(toy_line_results) * 3
    assert set(lpb["abc_class"]) == {"A", "B", "C"}


def test_line_policy_brief_carries_the_lines_own_saving_not_a_per_sku_one(
    toy_line_results,
):
    """Saving is a LINE fact now — every class row for a line shows the SAME
    line_saving_eur, because attributing it per class/SKU was exactly the
    unverified assumption D-071 retracted."""
    lpb = line_policy_brief(toy_line_results)
    l1_savings = lpb[lpb.line_id == "L1"]["line_saving_eur"]
    assert l1_savings.nunique() == 1
    assert float(l1_savings.iloc[0]) == 29000.0


def test_class_breakdown_sums_real_sku_level_costs(toy_sku_level):
    cb = class_breakdown_view(toy_sku_level)
    l1_a = cb[(cb.line_id == "L1") & (cb.abc_class == "A")].iloc[0]
    assert l1_a["lost_sales_eur"] == 100.0
    assert l1_a["n_skus"] == 1


def test_class_breakdown_does_not_split_conversion_per_class(toy_sku_level):
    """D-066 still holds: conversion is not separable per class. It must
    appear identically across every class row for a line, not be divided."""
    cb = class_breakdown_view(toy_sku_level)
    l1_conv = cb[cb.line_id == "L1"]["line_conversion_cost_eur"]
    assert l1_conv.nunique() == 1
    assert float(l1_conv.iloc[0]) == 160000.0


def test_portfolio_brief_mentions_breaching_line():
    summary = pd.DataFrame(
        [
            {"line_id": "L1", "any_capacity_breach": False,
             "worst_case_utilisation_max": 0.6, "worst_case_overtime_hours": 0.0},
            {"line_id": "L2", "any_capacity_breach": True,
             "worst_case_utilisation_max": 1.15, "worst_case_overtime_hours": 720.0},
        ]
    )
    consistency = pd.DataFrame(
        [{"lever": "min_run_hours", "output": "total_economic_cost_eur", "consistent": True}]
    )
    text = portfolio_brief(summary, consistency)
    assert "L2" in text
    assert "L1" in text


# ---------------------------------------------------------------------------
# export_artefacts
# ---------------------------------------------------------------------------


def test_export_writes_expected_files(
    scenario_result, assumptions, schema, toy_line_results, toy_sku_level, tmp_path
):
    av = avoidable_cost_view(scenario_result, assumptions, schema)
    lpb = line_policy_brief(toy_line_results)
    cb = class_breakdown_view(toy_sku_level)
    brief = portfolio_brief(
        pd.DataFrame([{"line_id": "L1", "any_capacity_breach": False,
                       "worst_case_utilisation_max": 0.6,
                       "worst_case_overtime_hours": 0.0}]),
        pd.DataFrame([{"lever": "x", "output": "y", "consistent": True}]),
    )
    paths = export_artefacts(av, lpb, cb, brief, str(tmp_path))
    for p in paths.values():
        assert os.path.isfile(p)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_reporter_does_not_recompute_engine_outputs():
    """This module may only reshape totals it is handed — it must not
    re-derive lost_sales_eur etc. from raw quantities, which would create a
    second place these numbers could diverge from their source of truth."""
    text = (REPO / "src" / "reporter.py").read_text()
    forbidden = ["_expected_shortfall", "np.expm1", "irreducible_volatility_cv"]
    for token in forbidden:
        assert token not in text, f"reporter.py appears to recompute via {token!r}"


def test_ground_truth_not_importable():
    text = (REPO / "src" / "reporter.py").read_text()
    assert "ground_truth" not in text.lower()


# ---------------------------------------------------------------------------
# D-079 — line_policy_brief_three_lever
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_line_results_three_lever():
    return pd.DataFrame({
        "line_id": ["L1", "L2"],
        "cover_A": [5.5, 5.5], "cover_B": [5.5, 5.5], "cover_C": [1.0, 1.0],
        "bias_correction": [1.0, 0.5], "min_run_hours": [6.0, 4.0],
        "service_achieved_A": [1.0, 0.9996], "service_achieved_B": [0.9995, 0.9981],
        "service_achieved_C": [0.9753, 0.9395],
        "unit_fill_rate": [0.9977, 0.9960], "overhang_cost_share": [0.7119, 0.7722],
        "total_economic_cost_eur": [4011794.88, 4500825.65],
        "default_total_cost_eur": [4056457.21, 5108118.14],
        "saving_eur": [44662.33, 607292.49],
    })


def test_line_policy_brief_three_lever_no_service_decision_column(toy_line_results_three_lever):
    """service_target must not appear as a settable decision column — only
    service_achieved, an outcome."""
    lpb = line_policy_brief_three_lever(toy_line_results_three_lever)
    assert "service_target" not in lpb.columns
    assert "service_achieved" in lpb.columns


def test_line_policy_brief_three_lever_bias_and_minrun_repeat_per_class(toy_line_results_three_lever):
    """bias_correction and min_run_hours are line-level, not class-level —
    every class row for a line must show the identical value."""
    lpb = line_policy_brief_three_lever(toy_line_results_three_lever)
    l1 = lpb[lpb.line_id == "L1"]
    assert l1["bias_correction"].nunique() == 1
    assert l1["min_run_hours"].nunique() == 1


def test_line_policy_brief_three_lever_row_count(toy_line_results_three_lever):
    lpb = line_policy_brief_three_lever(toy_line_results_three_lever)
    assert len(lpb) == len(toy_line_results_three_lever) * 3
