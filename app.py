"""Step 10 — Streamlit app.

Thin client. Every number displayed here traces to engine.run_scenario() or a
Step 9 reporter function via src/app_logic.py — this file contains NO
economic arithmetic (asserted by test_app_py_contains_no_inline_economic_
arithmetic-style checks, and by inspection: it only calls app_logic and
renders what comes back).

Rebuilt for D-071/D-072 to match the dashboard mockup agreed with the user:
per-class sliders (all four levers, per ABC class), a multi-select line
picker, Base/Optimal preset buttons, and a cost bridge with both delta rows.
Capacity warnings are computed LIVE from the scenario being viewed, not
looked up in a static table.
"""

import os

import pandas as pd
import streamlit as st

from src.app_logic import (
    ABC_CLASSES,
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
from src.engine import LeverSettings

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="IBP Trade-Off Engine", layout="wide")


@st.cache_resource
def _load_engine():
    assumptions, schema = load_config(REPO_ROOT)
    clean_master, sku_master = build_clean_data(REPO_ROOT)
    app_data = load_app_data(REPO_ROOT)
    engine = build_engine(assumptions, schema, clean_master, sku_master, app_data)
    return engine, assumptions, schema, app_data


engine, assumptions, schema, app_data = _load_engine()
fp_status = fingerprint_status(engine, app_data)
LEVER_CFG = assumptions["levers"]

st.title("IBP Trade-Off Engine")
st.caption(
    "Four levers, set per ABC class, in — four costs out. Deterministic "
    "simulation, capacity checked live on every change. Step 7's ML explains "
    "and generalises the policy; it does not produce this scenario's numbers."
)

# --------------------------------------------------------------------------
# Line selection — multi-select, "All lines" toggle
# --------------------------------------------------------------------------

all_line_ids = sorted(engine.line_master["line_id"].tolist())
line_category = {
    lid: str(engine.sku_master.loc[engine.line_skus(lid)[0], "category"])
    for lid in all_line_ids
}

st.subheader("Production line(s)")
cols = st.columns(len(all_line_ids) + 1)
if "selected_lines" not in st.session_state:
    st.session_state.selected_lines = [all_line_ids[0]]

with cols[0]:
    if st.button("All lines", width="stretch"):
        st.session_state.selected_lines = list(all_line_ids)
for i, lid in enumerate(all_line_ids):
    with cols[i + 1]:
        active = lid in st.session_state.selected_lines
        label = f"● {lid}" if active else lid
        if st.button(f"{label} — {line_category[lid]}", width="stretch"):
            sel = set(st.session_state.selected_lines)
            if lid in sel:
                if len(sel) > 1:
                    sel.remove(lid)
            else:
                sel.add(lid)
            st.session_state.selected_lines = sorted(sel)

selected_lines = st.session_state.selected_lines
n_lines = len(selected_lines)
st.caption(
    f"{'Consolidated — ' if n_lines > 1 else ''}"
    f"{' + '.join(selected_lines)} ({n_lines} line{'s' if n_lines != 1 else ''})"
)

# All selected lines must share categories for one set of class sliders to
# mean the same thing physically (min_run_hours default, line speed, etc
# vary by category) — the app still allows a mixed selection (levers are
# class semantics and apply the same way regardless of category) but flags
# it so the person knows what they are looking at.
categories_in_selection = {line_category[l] for l in selected_lines}
if len(categories_in_selection) > 1:
    st.caption(
        f"Note: selected lines span categories {sorted(categories_in_selection)} "
        f"— the same per-class levers are applied to each line's own economics."
    )

# --------------------------------------------------------------------------
# Base / Optimal presets
# --------------------------------------------------------------------------

primary_line = selected_lines[0]
base_levers = base_levers_for_line(engine, assumptions, primary_line)

if "current_levers" not in st.session_state:
    st.session_state.current_levers = base_levers.as_dict()

preset_col1, preset_col2 = st.columns(2)
with preset_col1:
    if st.button("Jump to base"):
        st.session_state.current_levers = base_levers.as_dict()
with preset_col2:
    optimal_disabled = n_lines > 1
    if st.button(
        "Jump to optimal",
        disabled=optimal_disabled,
        help="Only available for a single selected line — different lines "
        "have independently-found optima." if optimal_disabled else None,
    ):
        opt = optimal_levers_for_line(app_data["line_results"], primary_line)
        st.session_state.current_levers = opt.as_dict()

# --------------------------------------------------------------------------
# Sliders — per ABC class, all four levers
# --------------------------------------------------------------------------

st.subheader("Levers, per ABC class")
cur = st.session_state.current_levers
class_cols = st.columns(3)
new_levers: dict = {
    "service_target": {}, "inventory_cover_weeks": {},
    "forecast_bias_correction": {}, "min_run_hours": {},
}

st_lo, st_hi = (float(v) for v in LEVER_CFG["service_target_by_abc"]["range"])
cov_lo, cov_hi = (float(v) for v in LEVER_CFG["inventory_cover_weeks"]["range"])
bias_lo, bias_hi = (float(v) for v in LEVER_CFG["forecast_bias_correction"]["range"])
mrh_lo, mrh_hi = (float(v) for v in LEVER_CFG["min_run_hours"]["range"])

for i, cls in enumerate(ABC_CLASSES):
    with class_cols[i]:
        n_skus = sum(
            1 for lid in selected_lines for s in engine.line_skus(lid)
            if engine.sku_master.loc[s, "abc_class"] == cls
        )
        st.markdown(f"**Class {cls}** — {n_skus} SKUs")

        def _cur(field, default):
            v = cur.get(field, default)
            return float(v[cls]) if isinstance(v, dict) else float(v)

        new_levers["service_target"][cls] = st.slider(
            "Service target", st_lo, st_hi,
            _cur("service_target", 0.95), 0.005, key=f"svc_{cls}",
        )
        new_levers["inventory_cover_weeks"][cls] = st.slider(
            "Cover (weeks)", cov_lo, cov_hi,
            _cur("inventory_cover_weeks", 4.0), 0.5, key=f"cov_{cls}",
        )
        new_levers["forecast_bias_correction"][cls] = st.slider(
            "Bias correction", bias_lo, bias_hi,
            _cur("forecast_bias_correction", 0.0), 0.05, key=f"bias_{cls}",
        )
        new_levers["min_run_hours"][cls] = st.slider(
            "Min run (hours)", mrh_lo, mrh_hi,
            _cur("min_run_hours", 7.0), 0.5, key=f"mrh_{cls}",
        )

levers = LeverSettings(**new_levers).validate(assumptions)
st.session_state.current_levers = levers.as_dict()

# --------------------------------------------------------------------------
# Assumptions panel
# --------------------------------------------------------------------------

st.subheader("Assumptions in effect")
econ = assumptions["plant_economics"]
a1, a2, a3, a4 = st.columns(4)
a1.metric("Categories in view", ", ".join(sorted(categories_in_selection)))
a2.metric("Scheduled hrs / line-month", f"{econ['scheduled_hours_per_line_month']:g}")
a3.metric("Max overtime hrs / month", f"{econ['max_overtime_hours_month']:g}")
a4.metric("Fixed absorption / line-month", f"€{econ['fixed_absorption_eur_line_month']:,.0f}")

if not fp_status["match"]:
    st.warning(
        f"Reference tables (Step 7 output) were generated under a different "
        f"assumption set than the one currently loaded. "
        f"Live: {fp_status['live']} · Static: {fp_status['static']}. "
        f"'Jump to optimal' and the Optimal row below may be stale — "
        f"regenerate app_data/ from Step 7 if assumptions.yaml has changed."
    )

# --------------------------------------------------------------------------
# Live scenario
# --------------------------------------------------------------------------

scenario = run_scenario_for_lines(engine, selected_lines, levers, assumptions, schema)
base_scenario = run_scenario_for_lines(
    engine, selected_lines, base_levers, assumptions, schema
)

warnings = live_capacity_warnings(scenario["capacity"])
for w in warnings:
    st.warning(w)

# --------------------------------------------------------------------------
# Cost bridge — Base, Scenario, Δ vs base, Δ vs optimal, Optimal
# --------------------------------------------------------------------------

st.subheader("Cost bridge")

OUTPUTS = [
    ("lost_sales_eur", "Lost sales"),
    ("excess_obsolescence_eur", "Excess / slow-moving"),
    ("working_capital_cost_eur", "Working capital"),
    ("conversion_cost_avoidable_eur", "Conversion (avoidable)"),
]

bridge_rows = []
bv = base_scenario["avoidable_view"]
sv = scenario["avoidable_view"]
bridge_rows.append(
    {"": "Base", **{lbl: bv[k] for k, lbl in OUTPUTS},
     "Total": sum(bv[k] for k, _ in OUTPUTS)}
)
bridge_rows.append(
    {"": "Scenario", **{lbl: sv[k] for k, lbl in OUTPUTS},
     "Total": sum(sv[k] for k, _ in OUTPUTS)}
)
bridge_rows.append(
    {"": "Δ vs base",
     **{lbl: sv[k] - bv[k] for k, lbl in OUTPUTS},
     "Total": sum(sv[k] - bv[k] for k, _ in OUTPUTS)}
)

if n_lines == 1:
    opt_levers = optimal_levers_for_line(app_data["line_results"], primary_line)
    opt_scenario = run_scenario_for_lines(
        engine, selected_lines, opt_levers, assumptions, schema
    )
    ov = opt_scenario["avoidable_view"]
    bridge_rows.append(
        {"": "Δ vs optimal",
         **{lbl: sv[k] - ov[k] for k, lbl in OUTPUTS},
         "Total": sum(sv[k] - ov[k] for k, _ in OUTPUTS)}
    )
    bridge_rows.append(
        {"": "Optimal", **{lbl: ov[k] for k, lbl in OUTPUTS},
         "Total": sum(ov[k] for k, _ in OUTPUTS)}
    )
else:
    st.caption(
        "Optimal row omitted for a multi-line selection — each line's "
        "optimum was found independently; view one line at a time to see it."
    )

bridge_df = pd.DataFrame(bridge_rows).set_index("")
st.dataframe(bridge_df.style.format("€{:,.0f}"), width="stretch")

st.markdown(
    f"**Conversion — fixed (no lever moves this):** "
    f"€{sv['conversion_cost_fixed_eur']:,.0f}  ·  "
    f"**Total reported (full P&L) cost:** €{sv['total_reported_cost_eur']:,.0f}"
)

# --------------------------------------------------------------------------
# By-class breakdown
# --------------------------------------------------------------------------

st.subheader("By product class — current scenario")
cb = by_class_breakdown(scenario["sku_month"], levers)
st.dataframe(cb, width="stretch", hide_index=True)

st.caption("Cost accountability, by role (not named individuals)")
st.dataframe(
    owner_table(scenario["avoidable_view"], assumptions),
    width="stretch", hide_index=True,
)

# --------------------------------------------------------------------------
# SKU detail
# --------------------------------------------------------------------------

st.subheader("SKU detail — resulting parameters, this scenario")
sku_totals = (
    scenario["sku_month"]
    .groupby(["sku_id", "line_id", "abc_class"])
    .agg(
        lost_sales_eur=("lost_sales_eur", "sum"),
        excess_obsolescence_eur=("excess_obsolescence_eur", "sum"),
        working_capital_cost_eur=("working_capital_cost_eur", "sum"),
    )
    .reset_index()
)
sku_totals["service_target"] = sku_totals["abc_class"].map(
    lambda c: levers.resolve("service_target", c)
)
sku_totals["cover_weeks"] = sku_totals["abc_class"].map(
    lambda c: levers.resolve("inventory_cover_weeks", c)
)
sku_totals["bias_correction"] = sku_totals["abc_class"].map(
    lambda c: levers.resolve("forecast_bias_correction", c)
)
sku_totals["min_run_hours"] = sku_totals["abc_class"].map(
    lambda c: levers.resolve("min_run_hours", c)
)
sku_totals["est_cost_eur"] = (
    sku_totals["lost_sales_eur"]
    + sku_totals["excess_obsolescence_eur"]
    + sku_totals["working_capital_cost_eur"]
)
st.dataframe(
    sku_totals[
        ["sku_id", "line_id", "abc_class", "service_target", "cover_weeks",
         "bias_correction", "min_run_hours", "est_cost_eur"]
    ],
    width="stretch", hide_index=True,
)

# --------------------------------------------------------------------------
# Portfolio evidence — static reference, clearly dated
# --------------------------------------------------------------------------

st.subheader("Portfolio evidence (reference, from Steps 7-8)")
st.caption(
    f"Generated under assumption fingerprint {fp_status['static']}. "
    f"Not recomputed live — see decision D-072."
)

tab1, tab2 = st.tabs(["Optimal policy by class", "Lever consistency across lines"])
with tab1:
    st.dataframe(
        line_policy_table(app_data["line_results"]),
        width="stretch", hide_index=True,
    )
with tab2:
    st.dataframe(
        app_data["lever_consistency"], width="stretch", hide_index=True
    )

st.caption(
    "Magnitudes are not findings (architecture §10) — this is synthetic "
    "data. The method is the deliverable."
)
