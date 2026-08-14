"""Step 10 — Streamlit app.

Thin client. Every number displayed here traces to engine.run_scenario() or a
Step 9 reporter function via src/app_logic.py — this file contains NO economic
arithmetic (asserted by test_app_py_contains_no_inline_economic_arithmetic).

If you are reading this file to understand the four economic outputs, you are
in the wrong place: see src/engine.py's TradeOffEngine.economics-producing
code, or src/reporter.py's avoidable_cost_view. This file is layout and
sliders.
"""

import os

import streamlit as st

from src.app_logic import (
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

st.title("IBP Trade-Off Engine")
st.caption(
    "Four levers in, four costs out — deterministic simulation. "
    "Machine learning (Step 7) explains and generalises the policy; "
    "it does not produce this scenario's numbers."
)

# --------------------------------------------------------------------------
# SIDEBAR — line selector + four lever sliders, bound to assumptions ranges
# --------------------------------------------------------------------------

line_ids = sorted(engine.line_master["line_id"].tolist())
with st.sidebar:
    st.header("Scenario")
    line_id = st.selectbox("Production line", line_ids)

    category = str(
        engine.sku_master.loc[engine.line_skus(line_id)[0], "category"]
    )
    default_levers = LeverSettings.defaults(assumptions, category, "A")

    levers_cfg = assumptions["levers"]
    st_lo, st_hi = (float(v) for v in levers_cfg["service_target_by_abc"]["range"])
    cov_lo, cov_hi = (float(v) for v in levers_cfg["inventory_cover_weeks"]["range"])
    bias_lo, bias_hi = (
        float(v) for v in levers_cfg["forecast_bias_correction"]["range"]
    )
    mrh_lo, mrh_hi = (float(v) for v in levers_cfg["min_run_hours"]["range"])

    service_target = st.slider(
        "Service target", st_lo, st_hi, float(default_levers.service_target), 0.005
    )
    inventory_cover_weeks = st.slider(
        "Inventory cover (weeks)",
        cov_lo,
        cov_hi,
        float(default_levers.inventory_cover_weeks),
        0.5,
    )
    forecast_bias_correction = st.slider(
        "Forecast bias correction",
        bias_lo,
        bias_hi,
        float(default_levers.forecast_bias_correction),
        0.05,
    )
    min_run_hours = st.slider(
        "Minimum run length (hours)", mrh_lo, mrh_hi, float(default_levers.min_run_hours), 0.5
    )

levers = LeverSettings(
    service_target=service_target,
    inventory_cover_weeks=inventory_cover_weeks,
    forecast_bias_correction=forecast_bias_correction,
    min_run_hours=min_run_hours,
).validate(assumptions)

# --------------------------------------------------------------------------
# ASSUMPTIONS PANEL — main screen, not buried (architecture §9)
# --------------------------------------------------------------------------

st.subheader("Assumptions in effect")
econ = assumptions["plant_economics"]
cat_cfg = assumptions["categories"][category]
a1, a2, a3, a4 = st.columns(4)
a1.metric("Category", category)
a2.metric("Line speed (units/hr)", f"{cat_cfg['line_speed_units_hr']:,}")
a3.metric("Scheduled hrs / line-month", f"{econ['scheduled_hours_per_line_month']:g}")
a4.metric("Fixed absorption / line-month", f"€{econ['fixed_absorption_eur_line_month']:,.0f}")

if not fp_status["match"]:
    st.warning(
        f"Reference tables (Step 7/8 outputs) were generated under a "
        f"different assumption set than the one currently loaded. "
        f"Live: {fp_status['live']} · Static: {fp_status['static']}. "
        f"Portfolio evidence below may be stale — regenerate app_data/ from "
        f"Steps 7 and 8 if assumptions.yaml has changed."
    )
else:
    st.caption(f"Assumption fingerprint: {fp_status['live']} (reference tables match)")

# --------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------

st.subheader(f"Scenario result — {line_id}")
view = scenario_view(engine, line_id, levers, assumptions, schema)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lost sales", f"€{view['lost_sales_eur']:,.0f}")
c2.metric("Excess / slow-moving", f"€{view['excess_obsolescence_eur']:,.0f}")
c3.metric("Working capital", f"€{view['working_capital_cost_eur']:,.0f}")
c4.metric("Conversion — avoidable", f"€{view['conversion_cost_avoidable_eur']:,.0f}")

st.metric(
    "Conversion — FIXED (no lever moves this)",
    f"€{view['conversion_cost_fixed_eur']:,.0f}",
    help="Fixed plant absorption. Included in the full P&L total below but "
    "excluded from the optimisation objective — see decision D-062.",
)

st.markdown(
    f"**Total avoidable cost:** €{view['total_avoidable_cost_eur']:,.0f}  ·  "
    f"**Total reported (full P&L) cost:** €{view['total_reported_cost_eur']:,.0f}"
)

warning = capacity_check(line_id, inventory_cover_weeks, app_data["portfolio_summary"])
if warning:
    st.warning(warning)

st.caption("Cost accountability, by role (not named individuals)")
st.dataframe(owner_table(view, assumptions), use_container_width=True, hide_index=True)

st.bar_chart(
    {
        "Lost sales": view["lost_sales_eur"],
        "Excess": view["excess_obsolescence_eur"],
        "Working capital": view["working_capital_cost_eur"],
        "Conversion (avoidable)": view["conversion_cost_avoidable_eur"],
    }
)

# --------------------------------------------------------------------------
# PORTFOLIO EVIDENCE — static reference, clearly dated
# --------------------------------------------------------------------------

st.subheader("Portfolio evidence (reference, from Steps 7-8)")
st.caption(
    f"Generated under assumption fingerprint {fp_status['static']}. "
    f"Not recomputed live — see decision D-072."
)

tab1, tab2 = st.tabs(["Top policy moves", "Lever consistency across lines"])
with tab1:
    moves = top_policy_moves(
        app_data["optimal_policy"], engine.sku_master, assumptions
    )
    st.dataframe(moves, use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(
        app_data["lever_consistency"], use_container_width=True, hide_index=True
    )

st.caption(
    "Magnitudes are not findings (architecture §10) — this is synthetic "
    "data. The method is the deliverable."
)
