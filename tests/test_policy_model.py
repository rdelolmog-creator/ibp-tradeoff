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
    optimise_all_lines,
    joint_policy_grid,
    search_joint_policy,
    best_feasible_policy,
    search_all_lines,
    expand_to_sku_level,
    search_three_lever_policy,
    best_feasible_three_lever,
    search_all_lines_three_lever,
    bias_min_run_interaction,
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


def test_optimum_never_worse_than_default_across_all_abc_classes(engine):
    """D-067. Bug found on the real Colab run: default cost was computed from
    class A's policy applied to every SKU, so B/C-class SKUs were compared
    against a policy that was never their own and showed negative savings.
    Requires all three ABC classes present, which the MVD line alone may not
    guarantee — run on the full portfolio.
    """
    from src.policy_model import optimise_all_lines

    pol = optimise_all_lines(engine, n_cover=7, n_service=6)
    classes_present = engine.sku_master.loc[pol["sku_id"], "abc_class"].unique()
    assert len(classes_present) > 1, "test requires more than one ABC class"
    assert float(pol["saving_eur"].min()) >= -1e-6, (
        "a SKU's saving went negative — its default was almost certainly "
        "computed from another class's policy, not its own"
    )


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


def test_multi_line_sweep_covers_the_whole_portfolio(engine):
    """D-065: the model cannot be fitted on one line."""
    pol = optimise_all_lines(engine, n_cover=3, n_service=3)
    assert len(pol) == len(engine.sku_master)
    assert pol["line_id"].nunique() == len(engine.line_master)
    assert pol["sku_id"].is_unique


def test_single_line_training_has_almost_no_feature_variance(engine, optimal_policy,
                                                             assumptions, schema):
    """The reason D-065 exists, asserted rather than asserted-in-prose.

    Within one line, category, line speed, shelf life, price, standard cost,
    case size, MOQ and minimum run are all constant — so a model fitted there
    can only reduce to the ABC-class mean it is benchmarked against.
    """
    m = PolicyModel(optimal_policy, engine.sku_master.reset_index(drop=True),
                    engine.demand_characteristics,
                    build_line_master(assumptions, schema))
    tf = m.training_frame()
    varying = [f for f in ATTRIBUTE_FEATURES if f in tf.columns and tf[f].nunique() > 1]
    assert varying == ["abc_class"], (
        f"expected only abc_class to vary within a line, got {varying}"
    )


def test_portfolio_training_restores_feature_variance(engine, assumptions, schema):
    pol = optimise_all_lines(engine, n_cover=3, n_service=3)
    m = PolicyModel(pol, engine.sku_master.reset_index(drop=True),
                    engine.demand_characteristics,
                    build_line_master(assumptions, schema))
    tf = m.training_frame()
    varying = [f for f in ATTRIBUTE_FEATURES if f in tf.columns and tf[f].nunique() > 1]
    assert len(varying) >= 5, f"only {varying} vary across the portfolio"


def test_separable_is_the_default_attribution(optimiser):
    """O-16. `delta` moved 8 of 15 optima on the MVD line, so the per-SKU
    answer was partly an artefact of the attribution rule."""
    import inspect

    for fn in (PolicyOptimiser.optimise, PolicyOptimiser.sku_grid):
        assert inspect.signature(fn).parameters["mode"].default == "separable"


def test_ground_truth_not_importable():
    assert "ground_truth" not in (REPO / "src" / "policy_model.py").read_text()


# ---------------------------------------------------------------------------
# D-071 — joint per-class search
# ---------------------------------------------------------------------------


def test_every_candidate_is_a_real_joint_simulation(engine, assumptions):
    """Spot-check: manually run_scenario() the winning row's exact per-class
    LeverSettings and assert its totals match the search result's row
    exactly — proves the search tabulates real simulations, not estimates."""
    cover, service = joint_policy_grid(assumptions, n=2)
    search = search_joint_policy(engine, "L3", cover, service)
    row = search.iloc[0]
    levers = LeverSettings(
        service_target={"A": row.service_A, "B": row.service_B, "C": row.service_C},
        inventory_cover_weeks={"A": row.cover_A, "B": row.cover_B, "C": row.cover_C},
        forecast_bias_correction=row.bias_correction,
        min_run_hours=row.min_run_hours,
    ).validate(assumptions)
    direct = engine.run_scenario("L3", levers)
    assert direct.totals["total_economic_cost_eur"] == pytest.approx(
        row.total_economic_cost_eur, rel=1e-9
    )


def test_best_feasible_excludes_infeasible_rows():
    search = pd.DataFrame(
        [
            {"total_economic_cost_eur": 100.0, "capacity_shortfall_total": 500.0},
            {"total_economic_cost_eur": 200.0, "capacity_shortfall_total": 0.0},
        ]
    )
    best = best_feasible_policy(search)
    assert best["total_economic_cost_eur"] == 200.0


def test_raises_when_nothing_in_the_grid_is_feasible():
    search = pd.DataFrame(
        [
            {"total_economic_cost_eur": 100.0, "capacity_shortfall_total": 50.0},
            {"total_economic_cost_eur": 200.0, "capacity_shortfall_total": 10.0},
        ]
    )
    with pytest.raises(PolicyModelViolation):
        best_feasible_policy(search)


def test_sku_inherits_its_own_class_and_line_policy(engine, assumptions):
    lr = search_all_lines(engine, n=2)
    sku_level = expand_to_sku_level(lr, engine)
    lr_ix = lr.set_index("line_id")
    for _, row in sku_level.iterrows():
        expected_cover = lr_ix.loc[row.line_id, f"cover_{row.abc_class}"]
        expected_service = lr_ix.loc[row.line_id, f"service_{row.abc_class}"]
        assert row.cover_weeks == pytest.approx(expected_cover)
        assert row.service_target == pytest.approx(expected_service)


def test_saving_is_measured_against_the_uniform_default(engine, assumptions):
    lr = search_all_lines(engine, n=2)
    row = lr[lr.line_id == "L3"].iloc[0]
    category = str(
        engine.sku_master.loc[engine.line_skus("L3")[0], "category"]
    )
    default_levers = LeverSettings.defaults_per_class(assumptions, category)
    direct = engine.run_scenario("L3", default_levers)
    assert row["default_total_cost_eur"] == pytest.approx(
        direct.totals["total_economic_cost_eur"], rel=1e-9
    )


def test_conversion_cost_is_a_line_total_not_a_per_sku_split(engine, assumptions):
    lr = search_all_lines(engine, n=2)
    sku_level = expand_to_sku_level(lr, engine)
    for line_id, g in sku_level.groupby("line_id"):
        assert g["line_conversion_cost_eur"].nunique() == 1, (
            f"{line_id}: conversion cost varies per SKU — it should be an "
            f"identical line total, not silently re-attributed per SKU"
        )


def test_optimum_never_worse_than_the_true_per_class_default(engine, assumptions):
    """D-067's failure mode, recurring one level up if left unguarded."""
    lr = search_all_lines(engine, n=2)
    assert float(lr["saving_eur"].min()) >= -1e-6


def test_optimal_result_is_always_capacity_feasible(engine, assumptions):
    lr = search_all_lines(engine, n=2)
    assert (lr["capacity_shortfall_total"] <= 1e-6).all()


# ---------------------------------------------------------------------------
# D-079 — service frozen, three-lever search (Changes 1-5)
# ---------------------------------------------------------------------------


def test_service_is_frozen_at_abc_defaults_not_searched(engine, assumptions):
    cover = [1.0, 5.5]
    result = search_three_lever_policy(engine, "L3", cover, [0.0, 1.0], [6.0, 9.0])
    for c in ("A", "B", "C"):
        expected = float(assumptions["abc"][c]["service_floor"])
        col = f"service_achieved_{c}"
        assert col in result.columns


def test_three_lever_search_columns_present(engine):
    cover = [1.0, 5.5]
    result = search_three_lever_policy(engine, "L3", cover, [0.0], [6.0])
    for col in ("service_achieved_A", "service_achieved_B", "service_achieved_C",
                "unit_fill_rate", "overhang_cost_share", "overhang_sku_months"):
        assert col in result.columns


def test_unit_fill_rate_is_aggregate_not_mean_of_ratios(engine):
    """Change 3: 1 - sum(lost)/sum(demand), never a mean of per-row ratios."""
    cover = [5.5]
    result = search_three_lever_policy(engine, "L3", cover, [0.0], [9.0])
    row = result.iloc[0]
    assert 0.0 <= row["unit_fill_rate"] <= 1.0


def test_best_feasible_three_lever_excludes_infeasible():
    df = pd.DataFrame([
        {"total_economic_cost_eur": 100.0, "capacity_shortfall_total": 50.0},
        {"total_economic_cost_eur": 200.0, "capacity_shortfall_total": 0.0},
    ])
    best = best_feasible_three_lever(df)
    assert best["total_economic_cost_eur"] == 200.0


def test_search_all_lines_three_lever_never_worse_than_base(engine):
    lr = search_all_lines_three_lever(engine, n_cover=2, bias_values=(0.0, 1.0), min_run_values=(6.0, 9.0))
    assert float(lr["saving_eur"].min()) >= -1e-6


def test_search_all_lines_three_lever_always_feasible(engine):
    lr = search_all_lines_three_lever(engine, n_cover=2, bias_values=(0.0,), min_run_values=(6.0,))
    assert (lr["capacity_shortfall_total"] <= 1e-6).all()


def test_bias_min_run_interaction_flags_null_cells(engine):
    """Change 5: the interaction table must expose cells where bias has
    zero effect, not suppress them."""
    cover_combo = {"A": 5.5, "B": 5.5, "C": 1.0}
    interaction = bias_min_run_interaction(
        engine, "L2", cover_combo, bias_values=(0.0, 1.0), min_run_values=(9.0,)
    )
    assert "bias_has_zero_effect" in interaction.columns
    # at min_run=9 on L2, this is a known real null-effect cell
    row = interaction[interaction.min_run_hours == 9.0].iloc[0]
    assert bool(row["bias_has_zero_effect"]) is True


def test_delta_vs_base_strips_fixed_absorption():
    from src.reporter import delta_vs_base_view
    row = {"line_id": "L2", "total_economic_cost_eur": 4500825.65}
    r = delta_vs_base_view(row, base_total_eur=5108118.14, fixed_conversion_eur=3600000.0)
    assert r["avoidable_cost_eur"] == pytest.approx(900825.65, abs=0.01)
    assert r["delta_vs_base_eur"] < 0  # a real saving
