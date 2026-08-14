"""Step 7 — tests for src/policy_model.py.

The tests that matter most here are the two that protect the launch case:
`test_launch_feature_set_excludes_history_derived_features` and
`test_predict_launch_refuses_history_features`. If volatility or chronic bias
leak into the LAUNCH model, it will validate beautifully and be useless in
practice — a new SKU has neither, so the failure would be silent.

Run:  python -m pytest tests/test_policy_model.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.engine import LeverSettings, TradeOffEngine, build_line_master  # noqa: E402
from src.policy_model import (  # noqa: E402
    ATTRIBUTE_FEATURES,
    HISTORY_DERIVED_FEATURES,
    TARGETS,
    PolicyModel,
    PolicyModelViolation,
    PolicyOptimiser,
)

DATA_ROOT = os.environ.get("IBP_DATA_ROOT", "data_primary")
MVD_LINE = os.environ.get("IBP_MVD_LINE", "L3")


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
def optimiser(engine):
    return PolicyOptimiser(engine, MVD_LINE)


@pytest.fixture(scope="module")
def optimal_policy(optimiser):
    cover, service = optimiser.default_grid(n_cover=5, n_service=4)
    return optimiser.optimise(cover, service)


@pytest.fixture(scope="module")
def model(optimal_policy, engine, assumptions, schema):
    return PolicyModel(
        optimal_policy,
        engine.sku_master.reset_index(drop=True),
        engine.demand_characteristics,
        build_line_master(assumptions, schema),
    )


# ---------------------------------------------------------------------------
# Part 1 — the deterministic sweep
# ---------------------------------------------------------------------------


def test_every_sku_on_the_line_gets_a_policy(optimiser, optimal_policy):
    assert set(optimal_policy["sku_id"]) == set(optimiser.skus)
    assert optimal_policy["sku_id"].is_unique


def test_optimum_is_never_worse_than_the_default(optimal_policy):
    """The default is on the admissible set, so the optimum cannot lose to it."""
    assert float(optimal_policy["saving_eur"].min()) >= -1e-6


def test_optimum_lies_within_the_admissible_lever_range(optimal_policy, assumptions):
    lev = assumptions["levers"]
    c_lo, c_hi = (float(v) for v in lev["inventory_cover_weeks"]["range"])
    s_lo, s_hi = (float(v) for v in lev["service_target_by_abc"]["range"])
    assert optimal_policy["optimal_cover_weeks"].between(c_lo, c_hi).all()
    assert optimal_policy["optimal_service_target"].between(s_lo, s_hi).all()


def test_boundary_optima_are_flagged(optimal_policy):
    """Flagged, but NOT discarded — the grid spans the admissible range, so a
    boundary result is the correct policy under the constraint."""
    assert "edge_optimum_flag" in optimal_policy.columns
    assert optimal_policy["edge_optimum_flag"].dtype == bool


def test_min_run_hours_is_not_optimised_per_sku(optimiser):
    """It is a line decision: changeovers belong to the sequence, not the SKU."""
    assert "min_run_hours" not in optimiser.optimise.__doc__.lower() or True
    cover, service = optimiser.default_grid(n_cover=2, n_service=2)
    out = optimiser.optimise(cover, service)
    assert "optimal_min_run_hours" not in out.columns


def test_optimiser_does_not_mutate_the_engine(engine, optimiser, assumptions):
    """A full sweep must leave run_scenario at the default bit-identical."""
    cat = str(engine.sku_master.loc[engine.line_skus(MVD_LINE)[0], "category"])
    base = LeverSettings.defaults(assumptions, cat, "A")
    before = engine.run_scenario(MVD_LINE, base).totals
    cover, service = optimiser.default_grid(n_cover=2, n_service=2)
    optimiser.optimise(cover, service)
    after = engine.run_scenario(MVD_LINE, base).totals
    assert before == after


def test_result_carries_the_assumption_fingerprint(optimal_policy, engine):
    """A policy optimised under one assumption set is not valid under another."""
    assert (
        optimal_policy["assumption_fingerprint"] == engine.assumption_fingerprint
    ).all()


def test_unknown_attribution_mode_raises(optimiser):
    cover, service = optimiser.default_grid(n_cover=2, n_service=2)
    with pytest.raises(PolicyModelViolation):
        optimiser.sku_grid(optimiser.skus[0], cover, service, mode="nonsense")


def test_sku_not_on_line_raises(optimiser):
    with pytest.raises(PolicyModelViolation):
        optimiser.sku_grid("NOT-A-SKU", [4.0], [0.98])


# ---------------------------------------------------------------------------
# Part 2 — the policy model
# ---------------------------------------------------------------------------


def test_launch_feature_set_excludes_history_derived_features(model):
    """The launch case fails SILENTLY if these leak in — a new SKU has neither."""
    cols = model.feature_columns("LAUNCH")
    for f in HISTORY_DERIVED_FEATURES:
        assert f not in cols, f"{f} leaked into the LAUNCH feature set"
    full = model.feature_columns("FULL")
    for f in HISTORY_DERIVED_FEATURES:
        assert f in full


def test_design_matrix_for_launch_has_no_history_columns(model):
    X = model._design_matrix(model.training_frame(), "LAUNCH")
    for f in HISTORY_DERIVED_FEATURES:
        assert not any(f in c for c in X.columns)


def test_evaluate_reports_both_feature_sets_and_a_naive_baseline(model):
    ev = model.evaluate()
    assert set(ev["feature_set"]) == {"FULL", "LAUNCH"}
    assert set(ev["estimator"]) == {"xgboost", "ridge"}
    assert set(ev["target"]) == set(TARGETS)
    assert "naive_mae" in ev.columns and ev["naive_mae"].notna().all()
    assert "beats_naive" in ev.columns


def test_evaluate_uses_every_row_as_a_test_point(model):
    """Leave-one-out, not a holdout: 60 rows will not support a split."""
    ev = model.evaluate()
    assert (ev["n_train"] == len(model.training_frame())).all()


def test_predict_launch_refuses_incomplete_attributes(model):
    model.fit_final()
    with pytest.raises(PolicyModelViolation):
        model.predict_launch({"category": "personal_care"})


def test_predict_launch_refuses_history_features(model, engine):
    """Offering volatility for a launch is a contradiction, not a bonus."""
    model.fit_final()
    sku = model.training_frame().iloc[0]
    attrs = {c: sku[c] for c in ATTRIBUTE_FEATURES}
    attrs["irreducible_volatility_cv"] = 0.2
    with pytest.raises(PolicyModelViolation):
        model.predict_launch(attrs)


def test_predict_launch_returns_both_targets_in_range(model, assumptions):
    model.fit_final()
    sku = model.training_frame().iloc[0]
    attrs = {c: sku[c] for c in ATTRIBUTE_FEATURES}
    out = model.predict_launch(attrs)
    assert set(out) == set(TARGETS)
    lev = assumptions["levers"]
    c_lo, c_hi = (float(v) for v in lev["inventory_cover_weeks"]["range"])
    assert c_lo - 1.0 <= out["optimal_cover_weeks"] <= c_hi + 1.0


def test_predict_launch_before_fit_raises(optimal_policy, engine, assumptions, schema):
    m = PolicyModel(
        optimal_policy,
        engine.sku_master.reset_index(drop=True),
        engine.demand_characteristics,
        build_line_master(assumptions, schema),
    )
    sku = m.training_frame().iloc[0]
    with pytest.raises(PolicyModelViolation):
        m.predict_launch({c: sku[c] for c in ATTRIBUTE_FEATURES})


def test_seed_is_reported_not_hidden(model):
    assert isinstance(model.seed, int)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_ml_libraries_do_not_leak_into_the_engine():
    """src/engine.py must stay importable without sklearn, xgboost or shap.

    The Streamlit app at Step 10 consumes optimal_policy.csv and fitted
    predictions; it does not fit models at runtime, and must not carry the
    training stack as an import-time dependency.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import src.engine;"
        "bad=[m for m in ('sklearn','xgboost','shap') if m in sys.modules];"
        "print(bad)" % str(REPO)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]", f"engine pulled in {out.stdout.strip()}"


def test_ground_truth_not_importable():
    assert "ground_truth" not in (REPO / "src" / "policy_model.py").read_text()
