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
    ReporterViolation,
    avoidable_cost_view,
    capacity_warning,
    export_artefacts,
    owner_view,
    policy_brief,
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
# policy_brief / portfolio_brief
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_optimal_policy():
    return pd.DataFrame(
        {
            "sku_id": ["S1", "S2", "S3"],
            "line_id": ["L1", "L1", "L2"],
            "optimal_cover_weeks": [2.0, 6.0, 1.0],
            "optimal_service_target": [0.9, 0.99, 0.88],
            "saving_eur": [500.0, 5000.0, 50.0],
        }
    )


@pytest.fixture
def toy_sku_master():
    return pd.DataFrame(
        {
            "sku_id": ["S1", "S2", "S3"],
            "category": ["ambient", "ambient", "chilled"],
            "abc_class": ["A", "B", "C"],
        }
    ).set_index("sku_id")


@pytest.fixture
def toy_defaults():
    return {
        "A": {"inventory_cover_weeks": 4.0, "service_target": 0.985},
        "B": {"inventory_cover_weeks": 4.0, "service_target": 0.96},
        "C": {"inventory_cover_weeks": 6.0, "service_target": 0.93},
    }


def test_policy_brief_sorted_by_saving_descending(
    toy_optimal_policy, toy_sku_master, toy_defaults
):
    pb = policy_brief(toy_optimal_policy, toy_sku_master, toy_defaults, top_n=3)
    assert list(pb["saving_eur"]) == sorted(pb["saving_eur"], reverse=True)
    assert pb.iloc[0]["sku_id"] == "S2"


def test_policy_brief_respects_top_n(toy_optimal_policy, toy_sku_master, toy_defaults):
    pb = policy_brief(toy_optimal_policy, toy_sku_master, toy_defaults, top_n=2)
    assert len(pb) == 2


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
    scenario_result, assumptions, schema, toy_optimal_policy, toy_sku_master,
    toy_defaults, tmp_path
):
    av = avoidable_cost_view(scenario_result, assumptions, schema)
    pb = policy_brief(toy_optimal_policy, toy_sku_master, toy_defaults)
    brief = portfolio_brief(
        pd.DataFrame([{"line_id": "L1", "any_capacity_breach": False,
                       "worst_case_utilisation_max": 0.6,
                       "worst_case_overtime_hours": 0.0}]),
        pd.DataFrame([{"lever": "x", "output": "y", "consistent": True}]),
    )
    paths = export_artefacts(av, pb, brief, str(tmp_path))
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
