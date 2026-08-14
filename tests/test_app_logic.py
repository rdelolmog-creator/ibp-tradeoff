"""Step 10 — tests for src/app_logic.py.

The tests that matter most: `test_no_streamlit_import_in_app_logic` (keeps
this module testable without a runtime), and
`test_capacity_warnings_are_live_not_static` — proves capacity checking
reflects the ACTUAL scenario being viewed, not a canned Step 8 lookup, which
is the whole reason this module was rebuilt (D-071/D-072).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.app_logic import (  # noqa: E402
    ABC_CLASSES,
    APP_DATA_FILES,
    AppLogicViolation,
    base_levers_for_line,
    build_clean_data,
    build_engine,
    by_class_breakdown,
    fingerprint_status,
    line_policy_table,
    live_capacity_warnings,
    load_app_data,
    load_config,
    optimal_levers_for_line,
    owner_table,
    run_scenario_for_lines,
)
from src.engine import LeverSettings  # noqa: E402
from src.reporter import avoidable_cost_view, owner_view  # noqa: E402

DATA_ROOT = "data_primary"


@pytest.fixture(scope="module")
def config():
    return load_config(str(REPO))


@pytest.fixture(scope="module")
def clean_data():
    for p in (
        REPO / DATA_ROOT / "clean" / "clean_master.parquet",
        REPO / DATA_ROOT / "clean" / "sku_master.parquet",
    ):
        if not p.is_file():
            pytest.skip(f"missing pipeline artefact: {p}")
    return build_clean_data(str(REPO))


@pytest.fixture(scope="module")
def app_data():
    for rel in APP_DATA_FILES.values():
        if not (REPO / rel).is_file():
            pytest.skip(f"missing app_data artefact: {rel}")
    return load_app_data(str(REPO))


@pytest.fixture(scope="module")
def engine(config, clean_data, app_data):
    assumptions, schema = config
    clean_master, sku_master = clean_data
    return build_engine(assumptions, schema, clean_master, sku_master, app_data)


# ---------------------------------------------------------------------------
# Streamlit isolation
# ---------------------------------------------------------------------------


def test_no_streamlit_import_in_app_logic():
    text = (REPO / "src" / "app_logic.py").read_text()
    code_lines, in_doc = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('"""'):
            in_doc = not in_doc
            continue
        if not in_doc:
            code_lines.append(line)
    joined = "\n".join(code_lines)
    assert "import streamlit" not in joined
    assert not any(l.strip().startswith("st.") for l in code_lines)


# ---------------------------------------------------------------------------
# build_clean_data — read-only repo safety (regression, see D-072/deploy fix)
# ---------------------------------------------------------------------------


def test_build_clean_data_never_writes_inside_repo_root(tmp_path):
    import shutil

    repo_copy = tmp_path / "repo_copy"
    shutil.copytree(
        REPO, repo_copy,
        ignore=shutil.ignore_patterns("data_primary", "data_control", "__pycache__", ".git"),
    )

    def _relevant(root):
        return {p.relative_to(root) for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts}

    before = _relevant(repo_copy)
    build_clean_data(str(repo_copy), data_root=str(tmp_path / "writable_data"))
    after = _relevant(repo_copy)
    assert after == before


# ---------------------------------------------------------------------------
# Lever presets — D-071/D-072's real Base and Optimal
# ---------------------------------------------------------------------------


def test_base_levers_are_true_per_class_defaults(engine, config):
    assumptions, _ = config
    base = base_levers_for_line(engine, assumptions, "L3")
    for c in ABC_CLASSES:
        expected = float(assumptions["abc"][c]["service_floor"])
        assert base.resolve("service_target", c) == pytest.approx(expected)


def test_base_levers_are_not_class_a_applied_to_everyone(engine, config):
    """The retired convention — this must NOT be what 'Base' means anymore."""
    assumptions, _ = config
    base = base_levers_for_line(engine, assumptions, "L3")
    a_service = base.resolve("service_target", "A")
    c_service = base.resolve("service_target", "C")
    assert a_service != c_service, "Base collapsed back to a single class's policy"


def test_optimal_levers_come_from_committed_line_results(app_data):
    line_results = app_data["line_results"]
    line_id = line_results["line_id"].iloc[0]
    levers = optimal_levers_for_line(line_results, line_id)
    assert isinstance(levers.service_target, dict)
    assert set(levers.service_target.keys()) == set(ABC_CLASSES)


def test_optimal_levers_raise_for_unknown_line(app_data):
    with pytest.raises(AppLogicViolation):
        optimal_levers_for_line(app_data["line_results"], "NOT_A_LINE")


# ---------------------------------------------------------------------------
# fingerprint_status
# ---------------------------------------------------------------------------


def test_fingerprint_mismatch_is_detected(engine, app_data):
    import copy

    tampered = copy.deepcopy(app_data)
    tampered["line_results"] = tampered["line_results"].copy()
    tampered["line_results"]["assumption_fingerprint"] = "v0.0:deadbeef0000"
    status = fingerprint_status(engine, tampered)
    assert status["match"] is False
    assert status["static"] == "v0.0:deadbeef0000"


# ---------------------------------------------------------------------------
# run_scenario_for_lines — the live core
# ---------------------------------------------------------------------------


def test_single_line_matches_direct_engine_and_reporter_call(engine, config):
    assumptions, schema = config
    base = base_levers_for_line(engine, assumptions, "L3")
    via_app = run_scenario_for_lines(engine, ["L3"], base, assumptions, schema)

    direct_result = engine.run_scenario("L3", base)
    direct_view = avoidable_cost_view(direct_result, assumptions, schema)

    for key in ("lost_sales_eur", "excess_obsolescence_eur",
                "working_capital_cost_eur", "total_reported_cost_eur"):
        assert via_app["avoidable_view"][key] == pytest.approx(direct_view[key])


def test_multi_line_sums_two_independent_single_line_calls(engine, config):
    assumptions, schema = config
    base = base_levers_for_line(engine, assumptions, "L1")  # same category as L4
    combined = run_scenario_for_lines(engine, ["L1", "L4"], base, assumptions, schema)

    r1 = run_scenario_for_lines(engine, ["L1"], base, assumptions, schema)
    r4 = run_scenario_for_lines(engine, ["L4"], base, assumptions, schema)

    expected_total = (
        r1["avoidable_view"]["total_reported_cost_eur"]
        + r4["avoidable_view"]["total_reported_cost_eur"]
    )
    assert combined["avoidable_view"]["total_reported_cost_eur"] == pytest.approx(
        expected_total
    )
    assert len(combined["sku_month"]) == len(r1["sku_month"]) + len(r4["sku_month"])


def test_run_scenario_raises_with_no_lines(engine, config):
    assumptions, schema = config
    base = base_levers_for_line(engine, assumptions, "L3")
    with pytest.raises(AppLogicViolation):
        run_scenario_for_lines(engine, [], base, assumptions, schema)


def test_scenario_view_does_not_reimplement_economics():
    """Static guardrail, same style as Step 9's recompute check."""
    text = (REPO / "src" / "app_logic.py").read_text()
    forbidden = ["gross_margin_eur *", "* gross_margin_eur",
                 "carrying_rate_monthly", "_expected_shortfall",
                 "writeoff_trigger_pct_life_left"]
    for token in forbidden:
        assert token not in text, f"app_logic.py appears to recompute via {token!r}"


# ---------------------------------------------------------------------------
# live_capacity_warnings — the whole reason this module was rebuilt
# ---------------------------------------------------------------------------


def test_capacity_warnings_are_live_not_static(engine, config):
    """Proves capacity checking reflects the ACTUAL scenario just simulated,
    not a canned Step 8 lookup — the defect the old design carried."""
    assumptions, schema = config
    base = base_levers_for_line(engine, assumptions, "L2")
    calm = run_scenario_for_lines(engine, ["L2"], base, assumptions, schema)
    assert live_capacity_warnings(calm["capacity"]) == []

    stress = LeverSettings(
        service_target=0.995, inventory_cover_weeks=10.0,
        forecast_bias_correction=0.0, min_run_hours=20.0,
    ).validate(assumptions)
    stressed = run_scenario_for_lines(engine, ["L2"], stress, assumptions, schema)
    warnings = live_capacity_warnings(stressed["capacity"])
    assert len(warnings) == 1
    assert "L2" in warnings[0]


def test_capacity_warning_names_the_breaching_line_in_multiline(engine, config):
    assumptions, schema = config
    stress = LeverSettings(
        service_target=0.995, inventory_cover_weeks=10.0,
        forecast_bias_correction=0.0, min_run_hours=20.0,
    ).validate(assumptions)
    res = run_scenario_for_lines(engine, ["L2", "L3"], stress, assumptions, schema)
    warnings = live_capacity_warnings(res["capacity"])
    assert any("L2" in w for w in warnings)


# ---------------------------------------------------------------------------
# by_class_breakdown
# ---------------------------------------------------------------------------


def test_by_class_breakdown_counts_skus_not_sku_months(engine, config):
    """Regression: sku_month has 12 rows per SKU. n_skus must count distinct
    SKUs, not SKU-months."""
    assumptions, schema = config
    base = base_levers_for_line(engine, assumptions, "L3")
    res = run_scenario_for_lines(engine, ["L3"], base, assumptions, schema)
    cb = by_class_breakdown(res["sku_month"], base)
    expected_a = len(
        [s for s in engine.line_skus("L3")
         if engine.sku_master.loc[s, "abc_class"] == "A"]
    )
    a_row = cb[cb.abc_class == "A"]
    if not a_row.empty:
        assert int(a_row.iloc[0]["n_skus"]) == expected_a


def test_by_class_breakdown_does_not_fabricate_conversion(engine, config):
    assumptions, schema = config
    base = base_levers_for_line(engine, assumptions, "L3")
    res = run_scenario_for_lines(engine, ["L3"], base, assumptions, schema)
    cb = by_class_breakdown(res["sku_month"], base)
    assert cb["line_conversion_cost_eur"].isna().all()


# ---------------------------------------------------------------------------
# owner_table / line_policy_table — thin wrappers
# ---------------------------------------------------------------------------


def test_owner_table_matches_direct_reporter_call(engine, config):
    assumptions, schema = config
    base = base_levers_for_line(engine, assumptions, "L3")
    res = run_scenario_for_lines(engine, ["L3"], base, assumptions, schema)
    via_app = owner_table(res["avoidable_view"], assumptions)
    direct = owner_view(res["avoidable_view"], assumptions)
    pd.testing.assert_frame_equal(via_app, direct)


def test_line_policy_table_has_one_row_per_class(app_data):
    lpt = line_policy_table(app_data["line_results"])
    n_lines = app_data["line_results"]["line_id"].nunique()
    assert len(lpt) == n_lines * 3


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_ground_truth_not_importable():
    assert "ground_truth" not in (REPO / "src" / "app_logic.py").read_text().lower()
