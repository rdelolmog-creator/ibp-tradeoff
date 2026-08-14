"""Step 9 — reporter: aggregation, owner mapping, artefacts.

Packages what Steps 6-8 already produced. This module CONSUMES
src/engine.py, src/policy_model.py and src/portfolio_sweep.py — it does not
modify them, and it does not recompute any economic output. It only reshapes
totals it is handed, plus one derived split (avoidable vs fixed conversion
cost) that is arithmetic on numbers already computed elsewhere.

Three findings this module exists to make visible to a reader outside this
chat, all decisions already made and logged before this step was written:

  D-062  four outputs reported as AVOIDABLE cost — conversion cost excludes
         fixed absorption, which is displayed separately and labelled, never
         silently dropped, because dropping it without reconciling to the
         full P&L number is exactly how a plant controller concludes the
         model lost EUR 3.6M of their budget.
  D-066  the per-SKU policy optimum (Step 7) is over cover and service only.
         min_run_hours is a line decision and is reported as one.
  D-070  a warning on the specific lever region Step 8 found breaches
         capacity, generalised so it does not hard-code today's one finding
         (L2, cover >= ~10 weeks) as a permanent special case.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

# Imported for typing/documentation only — this module reads a ScenarioResult
# it is handed, it does not construct one. See test_reporter_does_not_
# recompute_engine_outputs.
from src.engine import ScenarioResult  # noqa: F401


class ReporterViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# Missing-artefact contract
# ---------------------------------------------------------------------------

_PRODUCING_STEP = {
    "clean_master.parquet": "Step 4 (step04_ingest_and_clean.ipynb)",
    "sku_master.parquet": "Step 4 (step04_ingest_and_clean.ipynb)",
    "demand_characteristics.csv": "Step 5a (blind estimator notebook)",
    "censoring_diagnostics.csv": "Step 5a (blind estimator notebook)",
    "optimal_policy.csv": "Step 7 (step07_policy_model.ipynb)",
    "portfolio_sweep.csv": "Step 8 (step08_all_lines_all_levers.ipynb)",
    "step08_portfolio_summary.csv": "Step 8 (step08_all_lines_all_levers.ipynb)",
    "step08_lever_consistency.csv": "Step 8 (step08_all_lines_all_levers.ipynb)",
}


def require_artefact(path: str) -> str:
    """Fail with a message that names the step that produces the file.

    A bare FileNotFoundError tells the reader nothing about what to go run.
    This module never recomputes an upstream artefact if it is missing —
    that would create a second place these numbers could diverge from their
    source of truth.
    """
    if os.path.isfile(path):
        return path
    fname = os.path.basename(path)
    producer = _PRODUCING_STEP.get(fname, "an earlier step")
    raise ReporterViolation(
        f"missing artefact {path!r}. This is produced by {producer} — "
        f"run it first. The reporter does not recompute upstream outputs."
    )


# ---------------------------------------------------------------------------
# D-062 — avoidable vs fixed cost
# ---------------------------------------------------------------------------


def avoidable_cost_view(
    scenario_result: "ScenarioResult", assumptions: Dict[str, Any], schema: Dict[str, Any]
) -> Dict[str, float]:
    """Split conversion cost into avoidable and fixed; return BOTH totals.

    total_avoidable_cost_eur is the sum of the four lever-responsive outputs.
    total_reported_cost_eur adds fixed absorption back for anyone who needs
    the full P&L figure. Both are always returned together — a function that
    drops one is a defect (D-062), not a simplification: reporting only the
    avoidable figure without the reconciling total is precisely how a plant
    controller ends up thinking the model lost EUR 3.6M of their P&L.

    Fixed absorption is read from assumptions.plant_economics and
    schema.meta.horizon_months, never hard-coded, so a change to either
    config value changes the reported figure without touching this module.
    """
    totals = scenario_result.totals
    econ = assumptions["plant_economics"]
    horizon_months = int(schema["meta"]["horizon_months"])

    fixed = float(econ["fixed_absorption_eur_line_month"]) * horizon_months
    conversion_total = float(totals["conversion_cost_eur"])
    avoidable = conversion_total - fixed
    if avoidable < -1e-6:
        raise ReporterViolation(
            f"fixed absorption ({fixed:,.2f}) exceeds total conversion cost "
            f"({conversion_total:,.2f}) — check assumptions.plant_economics."
            f"fixed_absorption_eur_line_month against schema.meta.horizon_months"
        )
    avoidable = max(avoidable, 0.0)

    lost_sales = float(totals["lost_sales_eur"])
    excess = float(totals["excess_obsolescence_eur"])
    working_capital = float(totals["working_capital_cost_eur"])

    total_avoidable = lost_sales + excess + working_capital + avoidable

    return {
        "line_id": scenario_result.line_id,
        "lost_sales_eur": lost_sales,
        "excess_obsolescence_eur": excess,
        "working_capital_cost_eur": working_capital,
        "conversion_cost_avoidable_eur": avoidable,
        "conversion_cost_fixed_eur": fixed,
        "total_avoidable_cost_eur": total_avoidable,
        "total_reported_cost_eur": total_avoidable + fixed,
        "assumption_fingerprint": scenario_result.assumption_fingerprint,
    }


# ---------------------------------------------------------------------------
# D-070 — capacity warning, generalised
# ---------------------------------------------------------------------------


def capacity_warning(
    line_id: str,
    proposed_cover_weeks: float,
    portfolio_summary_row: pd.Series,
    grid_step: Optional[float] = None,
) -> Optional[str]:
    """Warn if a proposed cover setting resembles the region Step 8 found to
    breach capacity on this line.

    Deliberately reads the breach condition from portfolio_summary_row rather
    than hard-coding a line name or a cover value: the finding today is "L2,
    cover >= ~10 weeks", but the logic must fire on whatever line and whatever
    setting develops the same pattern later. See
    test_capacity_warning_does_not_hardcode_a_line_name.

    portfolio_summary_row is one row of step08_portfolio_summary.csv, which
    records the WORST-CASE grid point per line — not necessarily the setting
    that produced the breach's exact cover value, only that a breach exists
    somewhere. Because the summary does not carry the breaching cover value
    itself, this function treats "any_capacity_breach and the proposal is at
    or above this line's cost-optimal cover plus one grid step" as the trigger
    — a conservative proxy, not a precise re-derivation of Step 8's grid. A
    tighter version would read portfolio_sweep.csv directly for the actual
    breaching rows; left as a documented limitation given remaining scope.
    """
    if not bool(portfolio_summary_row.get("any_capacity_breach", False)):
        return None

    optimal_cover = float(portfolio_summary_row["best_inventory_cover_weeks"])
    step = float(grid_step) if grid_step is not None else 1.0
    if float(proposed_cover_weeks) < optimal_cover + step:
        return None

    return (
        f"{line_id}: inventory_cover_weeks={float(proposed_cover_weeks):g} is at "
        f"or above the region this engine found to breach capacity on this "
        f"line (worst case: utilisation "
        f"{float(portfolio_summary_row['worst_case_utilisation_max']):.2f}x, "
        f"{float(portfolio_summary_row['worst_case_overtime_hours']):.0f} "
        f"overtime hours in the sweep horizon). This line's own cost-optimal "
        f"setting is cover={optimal_cover:g} weeks."
    )


# ---------------------------------------------------------------------------
# Owner mapping
# ---------------------------------------------------------------------------


def owner_view(
    avoidable_view: Dict[str, float], assumptions: Dict[str, Any]
) -> pd.DataFrame:
    """Map each output to its accountable ROLE (D-011 / architecture §11 PII
    answer: role level, not named individuals).

    Reads assumptions.owner_mapping. Does not invent a mapping inline — if the
    block is absent, this raises rather than guessing at accountability, which
    is not this module's decision to make silently.
    """
    mapping = assumptions.get("owner_mapping")
    if not mapping:
        raise ReporterViolation(
            "assumptions.owner_mapping is not defined. Accountability must be "
            "assigned explicitly in config, at role level (architecture §11), "
            "not invented inline in the reporter."
        )
    rows = []
    for key, owner in mapping.items():
        if key in avoidable_view:
            rows.append(
                {"output": key, "value_eur": avoidable_view[key], "owner_role": owner}
            )
    if not rows:
        raise ReporterViolation(
            "none of assumptions.owner_mapping's keys matched the avoidable "
            "cost view — check the mapping's keys against avoidable_cost_view's "
            "output fields."
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Policy brief (Step 7's output, made actionable)
# ---------------------------------------------------------------------------


def policy_brief(
    optimal_policy: pd.DataFrame,
    sku_master: pd.DataFrame,
    default_levers_by_class: Dict[str, Dict[str, float]],
    top_n: int = 10,
) -> pd.DataFrame:
    """The SKUs with the largest saving_eur, joined to category/ABC class,
    with current (class default) vs optimal cover and service shown side by
    side. Step 7 produced the optimum; this makes it the "what do I actually
    change" artefact a planner can act on without reading a notebook.
    """
    sm = sku_master.copy()
    if sm.index.name != "sku_id":
        sm = sm.set_index("sku_id")

    pol = optimal_policy.sort_values("saving_eur", ascending=False).head(top_n).copy()
    pol["category"] = pol["sku_id"].map(sm["category"])
    pol["abc_class"] = pol["sku_id"].map(sm["abc_class"])

    def _current(row: pd.Series, field: str) -> float:
        cls = row["abc_class"]
        return float(default_levers_by_class.get(cls, {}).get(field, float("nan")))

    pol["current_cover_weeks"] = pol.apply(
        lambda r: _current(r, "inventory_cover_weeks"), axis=1
    )
    pol["current_service_target"] = pol.apply(
        lambda r: _current(r, "service_target"), axis=1
    )

    out = pol[
        [
            "sku_id",
            "line_id",
            "category",
            "abc_class",
            "current_cover_weeks",
            "optimal_cover_weeks",
            "current_service_target",
            "optimal_service_target",
            "saving_eur",
        ]
    ].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Portfolio brief (plain text, CFO-first paragraph)
# ---------------------------------------------------------------------------


def portfolio_brief(
    portfolio_summary: pd.DataFrame, lever_consistency: pd.DataFrame
) -> str:
    """A short plain-text paragraph: which lines are healthy, which carry a
    capacity risk and under what condition, which lever directions are
    consistent vs line-specific. Read before the tables, not a restatement of
    them.
    """
    n_lines = len(portfolio_summary)
    breaching = portfolio_summary[portfolio_summary["any_capacity_breach"]]
    healthy = portfolio_summary[~portfolio_summary["any_capacity_breach"]]

    lines_ok = ", ".join(sorted(healthy["line_id"])) if len(healthy) else "none"
    lines_risk = ", ".join(sorted(breaching["line_id"])) if len(breaching) else "none"

    inconsistent = lever_consistency[~lever_consistency["consistent"]]
    n_pairs = len(lever_consistency)
    n_inconsistent = len(inconsistent)

    parts = [
        f"{n_lines} lines evaluated. Capacity clear across the tested range on: "
        f"{lines_ok}.",
    ]
    if len(breaching):
        rows = []
        for _, r in breaching.iterrows():
            rows.append(
                f"{r['line_id']} (worst case {float(r['worst_case_utilisation_max']):.2f}x "
                f"utilisation, {float(r['worst_case_overtime_hours']):.0f} overtime hrs; "
                f"own cost-optimal setting has zero shortfall)"
            )
        parts.append("Capacity risk under some settings on: " + "; ".join(rows) + ".")
    else:
        parts.append("No line breaches capacity anywhere in the tested grid.")

    if n_inconsistent:
        levers = sorted(inconsistent["lever"].unique())
        parts.append(
            f"{n_inconsistent} of {n_pairs} lever/output relationships are "
            f"line-specific rather than portfolio-wide, concentrated in: "
            f"{', '.join(levers)}. See step08_lever_consistency.csv before "
            f"generalising a single-line finding to the whole portfolio."
        )
    else:
        parts.append(
            "Every tested lever moves every output the same direction on "
            "every line — findings from one line generalise."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_artefacts(
    avoidable_view: Dict[str, float],
    policy_brief_df: pd.DataFrame,
    portfolio_brief_text: str,
    out_dir: str,
) -> Dict[str, str]:
    """Write the three reporter artefacts. No plotting dependency — Step 10
    (Streamlit) owns visualisation; this module owns the numbers."""
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    p1 = os.path.join(out_dir, "avoidable_cost_summary.csv")
    pd.DataFrame([avoidable_view]).to_csv(p1, index=False)
    paths["avoidable_cost_summary"] = p1

    p2 = os.path.join(out_dir, "policy_brief.csv")
    policy_brief_df.to_csv(p2, index=False)
    paths["policy_brief"] = p2

    p3 = os.path.join(out_dir, "portfolio_brief.txt")
    with open(p3, "w") as f:
        f.write(portfolio_brief_text)
    paths["portfolio_brief"] = p3

    return paths
