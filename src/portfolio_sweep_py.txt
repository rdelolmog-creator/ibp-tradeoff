"""Step 8 — all lines, all levers, together.

Broadens src/engine.py's use, does not modify it (D-046, architecture A-03).
Every result so far moved ONE lever at a time on one line (Step 6 gate) or
swept two levers per SKU on one line (Step 7). This module runs the full
four-lever factorial on every line, which is the first point interactions
become visible — a tight cover setting plus a large min_run_hours can push a
line toward overtime in a way no single-lever sweep would show.

This run is also the evidence base for the O-10 review (D-063): does any line
breach capacity under realistic settings, and do lever directions found on the
MVD line (L3) generalise to the rest of the portfolio? This module reports
that evidence. It does not draw O-10's conclusion — that is the review's job,
using this module's output.

No new parameters, no new mechanisms. Grid, run, report.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.engine import LeverSettings, TradeOffEngine

LEVER_NAMES = [
    "service_target",
    "inventory_cover_weeks",
    "forecast_bias_correction",
    "min_run_hours",
]

LEVER_RANGE_KEYS = {
    "service_target": "service_target_by_abc",
    "inventory_cover_weeks": "inventory_cover_weeks",
    "forecast_bias_correction": "forecast_bias_correction",
    "min_run_hours": "min_run_hours",
}


class PortfolioSweepViolation(Exception):
    pass


def lever_grid(assumptions: Dict[str, Any], n: int = 4) -> Dict[str, List[float]]:
    """n values per lever, evenly spaced across assumptions.levers.*.range.

    n=4 chosen from a measured runtime check (run_scenario ~63ms): a 4-value
    full factorial across 4 lines is 1,024 calls, ~65 seconds. n<3 cannot show
    an interior optimum; do not reduce below 3. Do not raise without
    re-checking runtime — it scales as n^4.
    """
    if n < 3:
        raise PortfolioSweepViolation(
            f"n={n} cannot show an interior optimum; use n >= 3"
        )
    levers = assumptions["levers"]
    grid: Dict[str, List[float]] = {}
    for name, key in LEVER_RANGE_KEYS.items():
        lo, hi = (float(v) for v in levers[key]["range"])
        grid[name] = [round(v, 5) for v in np.linspace(lo, hi, n)]
    return grid


def sweep_all_lines_all_levers(
    engine: TradeOffEngine, grid: Dict[str, List[float]]
) -> pd.DataFrame:
    """Full factorial over the four levers, run on every line.

    Records per-month max utilisation and total overtime hours, not only the
    mean — a mean can hide a peak month that actually breaches capacity, which
    is exactly the evidence O-10 needs.
    """
    rows: List[Dict[str, Any]] = []
    combos = list(itertools.product(*(grid[name] for name in LEVER_NAMES)))

    for line_id in engine.line_master["line_id"]:
        skus = engine.line_skus(line_id)
        if not skus:
            continue
        category = str(engine.sku_master.loc[skus[0], "category"])
        base = LeverSettings.defaults(engine.assumptions, category, "A")

        for combo in combos:
            levers = base.replace(**dict(zip(LEVER_NAMES, combo)))
            res = engine.run_scenario(line_id, levers)
            lm = res.line_month
            row = {"line_id": line_id, **dict(zip(LEVER_NAMES, combo))}
            row.update(res.totals)
            row["utilisation_max"] = float(lm["utilisation"].max())
            row["overtime_hours_total"] = float(lm["overtime_hours"].sum())
            row["capacity_shortfall_total"] = float(
                lm["capacity_shortfall_units"].sum()
            )
            rows.append(row)

    return pd.DataFrame(rows)


def portfolio_summary(sweep: pd.DataFrame, assumptions: Dict[str, Any]) -> pd.DataFrame:
    """One row per line: the cost-optimal setting, and worst-case capacity
    risk across the WHOLE grid — deliberately not only at the optimum.

    O-10 needs to know whether ANY realistic setting breaches capacity, not
    only whether the cheapest one does. A line that is fine at its optimum but
    breaches under a plausible alternative is still a live risk.
    """
    econ = assumptions["plant_economics"]
    cap = float(econ["scheduled_hours_per_line_month"]) + float(
        econ["max_overtime_hours_month"]
    )
    # `cap` is the physical ceiling the engine itself enforces (see
    # tests.test_engine.test_capacity_never_breaches_available_plus_max_overtime).
    # any_capacity_breach below reads capacity_shortfall_total instead of
    # recomputing the ceiling here, because capacity_shortfall_total is the
    # engine's own record of demand it could not schedule — the more direct
    # signal. cap is kept for the sanity check in the accompanying test.

    rows: List[Dict[str, Any]] = []
    for line_id, g in sweep.groupby("line_id"):
        best = g.loc[g["total_economic_cost_eur"].idxmin()]
        worst_util_row = g.loc[g["utilisation_max"].idxmax()]

        interior: Dict[str, bool] = {}
        for name in LEVER_NAMES:
            values = sorted(g[name].unique())
            interior[name] = bool(
                len(values) > 2 and values[0] < best[name] < values[-1]
            )

        rows.append(
            {
                "line_id": line_id,
                "best_total_cost_eur": float(best["total_economic_cost_eur"]),
                "best_service_target": float(best["service_target"]),
                "best_inventory_cover_weeks": float(best["inventory_cover_weeks"]),
                "best_forecast_bias_correction": float(
                    best["forecast_bias_correction"]
                ),
                "best_min_run_hours": float(best["min_run_hours"]),
                "worst_case_utilisation_max": float(g["utilisation_max"].max()),
                "worst_case_overtime_hours": float(g["overtime_hours_total"].max()),
                "any_capacity_breach": bool(
                    (g["capacity_shortfall_total"] > 1e-6).any()
                ),
                "interior_optimum_service_target": interior["service_target"],
                "interior_optimum_inventory_cover_weeks": interior[
                    "inventory_cover_weeks"
                ],
                "interior_optimum_forecast_bias_correction": interior[
                    "forecast_bias_correction"
                ],
                "interior_optimum_min_run_hours": interior["min_run_hours"],
            }
        )
    return pd.DataFrame(rows)


def lever_consistency(
    sweep: pd.DataFrame, grid: Dict[str, List[float]]
) -> pd.DataFrame:
    """For each lever, does moving it low->high change each output the SAME
    direction on every line, or does the direction flip?

    Tests directly whether single-line findings generalise — D-059's "don't
    correct bias on L3" result, and Step 7's service-target boundary-hugging,
    are the two live candidates for a line-specific result this function
    should catch.
    """
    outputs = [
        "lost_sales_eur",
        "excess_obsolescence_eur",
        "conversion_cost_eur",
        "working_capital_cost_eur",
        "total_economic_cost_eur",
    ]
    rows: List[Dict[str, Any]] = []

    for lever in LEVER_NAMES:
        lo_val, hi_val = min(grid[lever]), max(grid[lever])
        for output in outputs:
            directions: Dict[str, str] = {}
            for line_id, g in sweep.groupby("line_id"):
                others = [n for n in LEVER_NAMES if n != lever]
                # hold the other three at THIS line's most common (default-ish)
                # setting, i.e. the mode of each, so the comparison isolates
                # the lever under test
                mask = pd.Series(True, index=g.index)
                for other in others:
                    mask &= g[other] == g[other].mode().iloc[0]
                sub = g[mask]
                lo_rows = sub[sub[lever] == lo_val]
                hi_rows = sub[sub[lever] == hi_val]
                if lo_rows.empty or hi_rows.empty:
                    directions[line_id] = "unavailable"
                    continue
                delta = float(hi_rows[output].mean() - lo_rows[output].mean())
                if abs(delta) < 1e-6:
                    directions[line_id] = "flat"
                else:
                    directions[line_id] = "up" if delta > 0 else "down"

            observed = {d for d in directions.values() if d not in ("unavailable",)}
            consistent = len(observed) <= 1
            row = {"lever": lever, "output": output, "consistent": consistent}
            row.update({f"{k}_direction": v for k, v in directions.items()})
            rows.append(row)

    return pd.DataFrame(rows)
