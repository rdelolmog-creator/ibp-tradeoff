"""Step 10 — tests for src/app_logic.py and static checks on app.py.

The tests that matter most: `test_no_streamlit_import_in_app_logic` (what
keeps app_logic testable without a Streamlit runtime), and
`test_scenario_view_does_not_reimplement_economics` /
`test_app_py_contains_no_inline_economic_arithmetic` — together these are the
guardrail against economic logic leaking out of engine.py/reporter.py and
into the UI layer, where it would be a second, untested place these numbers
could diverge.
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
    APP_DATA_FILES,
    AppLogicViolation,
    build_clean_data,
    build_engine,
    capacity_check,
    fingerprint_status,
    load_app_data,
    load_config,
    owner_table,
    scenario_view,
    top_policy_moves,
)
from src.engine import LeverSettings  # noqa: E402
from src.reporter import ReporterViolation, capacity_warning, owner_view  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config(str(REPO))


@pytest.fixture(scope="module")
def clean_data():
    for p in (
        REPO / "data_primary" / "clean" / "clean_master.parquet",
        REPO / "data_primary" / "clean" / "sku_master.parquet",
    ):
        if not p.is_file():
            pytest.skip(f"missing pipeline artefact: {p}")
    return build_clean_data(str(REPO))


@pytest.fixture(scope="module")
def app_data():
    for rel in APP_DATA_FILES.values():
        if not (REPO / rel).is_file():
            pytest.skip(f"missing app_data artefact: {rel} — run the Step 10 setup cell")
    return load_app_data(str(REPO))


@pytest.fixture(scope="module")
def engine(config, clean_data, app_data):
    assumptions, schema = config
    clean_master, sku_master = clean_data
    return build_engine(assumptions, schema, clean_master, sku_master, app_data)


@pytest.fixture(scope="module")
def base_levers(engine, config):
    assumptions, _ = config
    cat = str(engine.sku_master.loc[engine.line_skus("L3")[0], "category"])
    return LeverSettings.defaults(assumptions, cat, "A")


# ---------------------------------------------------------------------------
# Streamlit isolation — the thing that makes app_logic testable at all
# ---------------------------------------------------------------------------


def test_no_streamlit_import_in_app_logic():
    text = (REPO / "src" / "app_logic.py").read_text()
    code_lines = [
        line for line in text.splitlines() if not line.strip().startswith("#")
    ]
    # exclude the module docstring, which legitimately mentions the string
    # "import streamlit" while explaining why it must not appear as code
    in_docstring = False
    checked = []
    for line in code_lines:
        stripped = line.strip()
        if stripped.startswith('"""'):
            in_docstring = not in_docstring
            continue
        if not in_docstring:
            checked.append(line)
    joined = "\n".join(checked)
    assert "import streamlit" not in joined
    assert "st.cache" not in joined
    assert not any(l.strip().startswith("st.") for l in checked)


# ---------------------------------------------------------------------------
# build_clean_data — no divergent cleaning path
# ---------------------------------------------------------------------------


def test_build_clean_data_matches_step4_output(clean_data):
    from src.ingest import DataIngestor
    from src.cleaner import DataCleaner

    ing = DataIngestor(repo_root=str(REPO), data_root="data_primary")
    ref_cm, ref_sm, _ = DataCleaner(ing.schema, ing.assumptions).clean(ing.load())

    app_cm, app_sm = clean_data
    assert app_cm.shape == ref_cm.shape
    assert app_sm.shape == ref_sm.shape
    assert set(app_cm.columns) == set(ref_cm.columns)
    assert set(app_sm.columns) == set(ref_sm.columns)


# ---------------------------------------------------------------------------
# load_app_data
# ---------------------------------------------------------------------------


def test_load_app_data_raises_on_missing_file_naming_it(tmp_path):
    # an empty repo_root — every app_data file is "missing"
    with pytest.raises((ReporterViolation, FileNotFoundError)) as exc:
        load_app_data(str(tmp_path))
    assert "app_data" in str(exc.value) or "demand_characteristics" in str(exc.value)


def test_load_app_data_returns_all_five(app_data):
    assert set(app_data.keys()) == set(APP_DATA_FILES.keys())
    for df in app_data.values():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# fingerprint_status
# ---------------------------------------------------------------------------


def test_fingerprint_matches_for_freshly_generated_app_data(engine, app_data):
    """app_data was regenerated from the current assumptions in this sandbox
    run, so fingerprints should agree here."""
    status = fingerprint_status(engine, app_data)
    assert status["live"] == status["static"]
    assert status["match"] is True


def test_fingerprint_mismatch_is_detected(engine, app_data):
    import copy

    tampered = copy.deepcopy(app_data)
    tampered["optimal_policy"] = tampered["optimal_policy"].copy()
    tampered["optimal_policy"]["assumption_fingerprint"] = "v0.0:deadbeef0000"
    status = fingerprint_status(engine, tampered)
    assert status["match"] is False
    assert status["live"] != status["static"]
    assert status["static"] == "v0.0:deadbeef0000"


# ---------------------------------------------------------------------------
# scenario_view — must not reimplement economics
# ---------------------------------------------------------------------------


def test_scenario_view_matches_direct_engine_and_reporter_call(
    engine, base_levers, config
):
    assumptions, schema = config
    via_app = scenario_view(engine, "L3", base_levers, assumptions, schema)

    from src.reporter import avoidable_cost_view

    result = engine.run_scenario("L3", base_levers)
    direct = avoidable_cost_view(result, assumptions, schema)

    assert via_app == direct


def test_scenario_view_does_not_reimplement_economics():
    """Static guardrail: app_logic.py must not contain arithmetic that
    belongs in engine.py or reporter.py — same style of check as Step 9's
    recompute guardrail (test_reporter_does_not_recompute_engine_outputs)."""
    text = (REPO / "src" / "app_logic.py").read_text()
    forbidden = [
        "gross_margin_eur *",
        "* gross_margin_eur",
        "carrying_rate_monthly",
        "_expected_shortfall",
        "writeoff_trigger_pct_life_left",
    ]
    for token in forbidden:
        assert token not in text, f"app_logic.py appears to recompute via {token!r}"


# ---------------------------------------------------------------------------
# capacity_check / owner_table — thin wrappers
# ---------------------------------------------------------------------------


def test_capacity_check_matches_direct_reporter_call(app_data):
    summary = app_data["portfolio_summary"]
    line_id = summary["line_id"].iloc[0]
    row = summary.set_index("line_id").loc[line_id]
    cover = float(row["best_inventory_cover_weeks"]) + 10.0  # push toward a warning

    via_app = capacity_check(line_id, cover, summary)
    direct = capacity_warning(line_id, cover, row)
    assert via_app == direct


def test_owner_table_matches_direct_reporter_call(engine, base_levers, config):
    assumptions, schema = config
    from src.reporter import avoidable_cost_view

    result = engine.run_scenario("L3", base_levers)
    view = avoidable_cost_view(result, assumptions, schema)

    via_app = owner_table(view, assumptions)
    direct = owner_view(view, assumptions)
    pd.testing.assert_frame_equal(via_app, direct)


# ---------------------------------------------------------------------------
# top_policy_moves
# ---------------------------------------------------------------------------


def test_top_policy_moves_returns_sorted_rows(app_data, engine, config):
    assumptions, _ = config
    moves = top_policy_moves(
        app_data["optimal_policy"], engine.sku_master, assumptions, top_n=5
    )
    assert len(moves) <= 5
    assert list(moves["saving_eur"]) == sorted(moves["saving_eur"], reverse=True)


# ---------------------------------------------------------------------------
# app.py — static checks (no Streamlit runtime needed)
# ---------------------------------------------------------------------------


def test_app_py_imports_only_from_app_logic_for_computation():
    text = (REPO / "app.py").read_text()
    assert "from src.app_logic import" in text
    assert "from src.engine import LeverSettings" in text  # LeverSettings is a
    # dataclass constructor, not economic logic — permitted


def test_app_py_contains_no_inline_economic_arithmetic():
    """If app.py computes a cost rather than displaying one already computed
    by app_logic/reporter/engine, that is a scope violation."""
    text = (REPO / "app.py").read_text()
    forbidden = [
        "gross_margin_eur *",
        "* gross_margin_eur",
        "carrying_rate_monthly",
        "_expected_shortfall",
        "writeoff_trigger_pct_life_left",
        "std_cost_eur *",
    ]
    for token in forbidden:
        assert token not in text, f"app.py appears to compute economics via {token!r}"


def test_app_py_sliders_reference_assumptions_ranges_not_literals():
    """Lightweight static guard: every st.slider call's bounds should trace to
    a variable built from assumptions['levers'], not a bare literal pair typed
    directly into the slider call."""
    text = (REPO / "app.py").read_text()
    assert "levers_cfg = assumptions" in text
    for lever_key in (
        "service_target_by_abc",
        "inventory_cover_weeks",
        "forecast_bias_correction",
        "min_run_hours",
    ):
        assert lever_key in text, f"{lever_key} range not read from assumptions in app.py"


def test_app_py_shows_both_avoidable_and_fixed_cost():
    """D-062's exact requirement, checked at the UI layer too — the fixed
    absorption figure must be visible, not only the avoidable total."""
    text = (REPO / "app.py").read_text()
    assert "conversion_cost_fixed_eur" in text
    assert "total_reported_cost_eur" in text


def test_app_py_syntax_is_valid():
    import ast

    ast.parse((REPO / "app.py").read_text())


def test_requirements_txt_lists_streamlit_and_pyarrow():
    text = (REPO / "requirements.txt").read_text().lower()
    for pkg in ("streamlit", "pandas", "numpy", "pyyaml", "pyarrow"):
        assert pkg in text, f"{pkg} missing from requirements.txt"


def test_ground_truth_not_importable():
    for f in ("app_logic.py",):
        text = (REPO / "src" / f).read_text()
        assert "ground_truth" not in text.lower()
    assert "ground_truth" not in (REPO / "app.py").read_text().lower()
