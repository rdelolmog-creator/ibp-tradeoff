"""Step 7 — optimal-policy model.

Two parts, and the order matters:

  PART 1  PolicyOptimiser sweeps the engine per SKU for the cost-minimising
          inventory cover and service target. Deterministic, no ML. This is the
          operational answer.

  PART 2  PolicyModel fits SKU ATTRIBUTES to those optimal parameters, so a
          policy can be set for a SKU the engine cannot sweep. The principal
          case is a new launch: no demand history, so Step 5a cannot
          characterise it and the engine cannot simulate it, but every
          attribute this model needs is known before first shipment.

The optimisation is deterministic. Machine learning explains and generalises
the policy; it does not produce it (D-064).

Why not learn cost directly: the engine already computes conversion cost
causally from run hours, changeovers, overtime and absorption. A model fitted
to engine output would be learning the engine — circular, and §9 territory.

sklearn / xgboost / shap are imported lazily inside this module only. They must
never become import-time dependencies of src/engine.py or of app.py's critical
path: the Streamlit app consumes optimal_policy.csv and fitted predictions, it
does not fit models at runtime.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.engine import LeverSettings, TradeOffEngine

SEED = 20260807  # same seed discipline as the generator; reported, never hidden

# Features derived from demand history. A new launch has neither, which is the
# whole reason the LAUNCH feature set exists.
HISTORY_DERIVED_FEATURES = ["irreducible_volatility_cv", "chronic_bias_l1"]

ATTRIBUTE_FEATURES = [
    "category",
    "abc_class",
    "gross_margin_eur",
    "price_eur",
    "std_cost_eur",
    "shelf_life_days",
    "case_size",
    "moq_units",
    "min_run_units",
    "line_speed_units_hr",
]

TARGETS = ["optimal_cover_weeks", "optimal_service_target"]


class PolicyModelViolation(Exception):
    """Raised when a contract, feature-set or completeness rule is breached."""


# ---------------------------------------------------------------------------
# PART 1 — deterministic sweep
# ---------------------------------------------------------------------------


class PolicyOptimiser:
    """Find each SKU's cost-minimising cover and service target.

    THE ATTRIBUTION PROBLEM. Conversion cost is a LINE property: changeovers
    and capacity are shared across the SKUs on a line, so one SKU's cost is not
    separable from its neighbours'. Two admissible treatments:

      'separable' exclude conversion cost from the per-SKU objective and
                  optimise only the three separable outputs. THE DEFAULT.
      'delta'     attribute the change in line conversion cost across SKUs
                  pro-rata by run hours. Retained for comparison only.

    'separable' is the default per O-16. Measured on the MVD line, 8 of 15 SKUs
    chose a DIFFERENT optimum under the two modes — so the per-SKU answer was
    partly an artefact of an arbitrary attribution rule, and pro-rata by run
    hours is only one defensible split among several. A result that depends on
    that choice is not a result.

    This also matches the architecture's own division of levers: min_run_hours
    is the LINE lever, cover and service are the SKU levers. Making a per-SKU
    optimum depend on a line-level joint cost contradicts that split.

    The cost of the choice, which belongs on the limitations slide rather than
    being hidden behind a fake attribution: a SKU that wants small frequent
    batches genuinely does impose changeover cost on its neighbours, and
    'separable' does not see that interaction.
    """

    def __init__(self, engine: TradeOffEngine, line_id: str) -> None:
        self.engine = engine
        self.line_id = line_id
        self.assumptions = engine.assumptions

        skus = engine.line_skus(line_id)
        if not skus:
            raise PolicyModelViolation(f"no SKUs allocated to line {line_id!r}")
        self.skus = skus

        category = str(engine.sku_master.loc[skus[0], "category"])
        self.base_levers = LeverSettings.defaults(self.assumptions, category, "A")
        self._baseline = engine.run_scenario(line_id, self.base_levers)

    # -- grids ----------------------------------------------------------

    def default_grid(self, n_cover: int = 7, n_service: int = 6) -> Tuple[List[float], List[float]]:
        """Grid endpoints come from assumptions.levers.*.range, not literals.

        A grid rather than a continuous optimiser: the cost surface is not
        smooth. Min-run rounding and the excess-cover threshold both create
        steps, so a gradient method would chase artefacts of the discretisation.
        """
        lev = self.assumptions["levers"]
        c_lo, c_hi = (float(v) for v in lev["inventory_cover_weeks"]["range"])
        s_lo, s_hi = (float(v) for v in lev["service_target_by_abc"]["range"])
        cover = [round(v, 4) for v in np.linspace(c_lo, c_hi, n_cover)]
        service = [round(v, 4) for v in np.linspace(s_lo, s_hi, n_service)]
        return cover, service

    # -- per-SKU cost surface -------------------------------------------

    def _sku_cost(
        self, sku_id: str, levers: LeverSettings, mode: str
    ) -> Dict[str, float]:
        """Cost attributable to one SKU under a lever setting.

        The three separable outputs are read directly from that SKU's rows.
        Conversion cost is a line total, so under 'delta' it enters as the
        change against the baseline line cost.
        """
        res = self.engine.run_scenario(self.line_id, levers)
        s = res.sku_month
        row = s[s["sku_id"] == sku_id]

        separable = float(
            row["lost_sales_eur"].sum()
            + row["excess_obsolescence_eur"].sum()
            + row["working_capital_cost_eur"].sum()
        )
        line_conv = float(res.line_month["conversion_cost_eur"].sum())
        conv_delta = line_conv - float(
            self._baseline.line_month["conversion_cost_eur"].sum()
        )

        if mode == "separable":
            total = separable
        elif mode == "delta":
            total = separable + conv_delta
        else:
            raise PolicyModelViolation(f"unknown attribution mode {mode!r}")

        return {
            "lost_sales_eur": float(row["lost_sales_eur"].sum()),
            "excess_obsolescence_eur": float(row["excess_obsolescence_eur"].sum()),
            "working_capital_cost_eur": float(row["working_capital_cost_eur"].sum()),
            "conversion_delta_eur": conv_delta,
            "total_eur": total,
        }

    def sku_grid(
        self,
        sku_id: str,
        cover_values: Sequence[float],
        service_values: Sequence[float],
        mode: str = "separable",
    ) -> pd.DataFrame:
        """Cost surface for one SKU. min_run_hours held at the line default.

        min_run_hours is deliberately not optimised per SKU: it is a LINE
        decision, and changeovers are a property of the sequence on a line, not
        of any one SKU (D-064).
        """
        if sku_id not in self.skus:
            raise PolicyModelViolation(
                f"{sku_id!r} is not allocated to line {self.line_id!r}"
            )
        rows: List[Dict[str, Any]] = []
        for cover, service in itertools.product(cover_values, service_values):
            levers = self.base_levers.replace(
                inventory_cover_weeks=float(cover), service_target=float(service)
            )
            costs = self._sku_cost(sku_id, levers, mode)
            rows.append(
                {
                    "sku_id": sku_id,
                    "inventory_cover_weeks": float(cover),
                    "service_target": float(service),
                    "attribution_mode": mode,
                    **costs,
                }
            )
        return pd.DataFrame(rows)

    # -- optimise -------------------------------------------------------

    def optimise(
        self,
        cover_values: Optional[Sequence[float]] = None,
        service_values: Optional[Sequence[float]] = None,
        mode: str = "separable",
    ) -> pd.DataFrame:
        """One row per SKU: its cost-minimising cover and service target.

        Vectorised over the grid rather than per SKU, because run_scenario is
        line-level: one scenario evaluation yields every SKU's separable costs
        at once. Evaluating per SKU per grid point would repeat the same
        simulation len(skus) times for no additional information.
        """
        if cover_values is None or service_values is None:
            dc, ds = self.default_grid()
            cover_values = cover_values or dc
            service_values = service_values or ds

        base_conv = float(self._baseline.line_month["conversion_cost_eur"].sum())
        frames: List[pd.DataFrame] = []

        # D-067. Each SKU's default depends on its OWN abc_class (service
        # floor and target cover both vary by class — see assumptions.abc),
        # but the grid is built once from the overall lever range. A class's
        # default therefore often does not land on the shared grid, and
        # because the cost surface is NOT smooth (min-run rounding, the
        # excess-cover threshold both create steps), a coarse grid can score
        # worse everywhere it samples than a default point it never tested.
        # The optimum must never lose to the default, so every distinct
        # class default actually present on this line is added as an
        # explicit candidate point, not just used as the comparison baseline.
        class_defaults = set()
        for s_id in self.skus:
            cat = str(self.engine.sku_master.loc[s_id, "category"])
            abc = str(self.engine.sku_master.loc[s_id, "abc_class"])
            dl = LeverSettings.defaults(self.assumptions, cat, abc)
            class_defaults.add(
                (dl.inventory_cover_weeks, dl.service_target)
            )
        grid_points = set(itertools.product(cover_values, service_values))
        eval_points = grid_points | class_defaults

        for cover, service in eval_points:
            levers = self.base_levers.replace(
                inventory_cover_weeks=float(cover), service_target=float(service)
            )
            res = self.engine.run_scenario(self.line_id, levers)
            conv_delta = (
                float(res.line_month["conversion_cost_eur"].sum()) - base_conv
            )
            g = (
                res.sku_month.groupby("sku_id")[
                    [
                        "lost_sales_eur",
                        "excess_obsolescence_eur",
                        "working_capital_cost_eur",
                    ]
                ]
                .sum()
                .reset_index()
            )
            g["separable_eur"] = (
                g["lost_sales_eur"]
                + g["excess_obsolescence_eur"]
                + g["working_capital_cost_eur"]
            )
            # the line-level conversion delta is shared, so it is spread across
            # the line's SKUs in proportion to their run hours — the only
            # defensible split when the cost is genuinely joint
            hours = res.sku_month.groupby("sku_id")["run_hours"].sum()
            share = hours / hours.sum() if float(hours.sum()) > 0 else 0.0
            g["conversion_delta_eur"] = g["sku_id"].map(share).fillna(0.0) * conv_delta
            g["total_eur"] = g["separable_eur"] + (
                g["conversion_delta_eur"] if mode == "delta" else 0.0
            )
            g["inventory_cover_weeks"] = float(cover)
            g["service_target"] = float(service)
            frames.append(g)

        surface = pd.concat(frames, ignore_index=True)

        best_idx = surface.groupby("sku_id")["total_eur"].idxmin()
        best = surface.loc[best_idx].set_index("sku_id")

        # D-067 (continued) — the real root cause. total_cost_at_default_eur
        # must be each SKU's OWN class default, not self.base_levers applied
        # uniformly, which is always class A's policy. A class-B or class-C
        # SKU compared against class A's (tighter) service floor and shorter
        # cover was being scored against a policy that was never its own —
        # every one of the observed "negative savings" was a class-B SKU.
        default_lookup: Dict[str, Tuple[float, float]] = {}
        for s_id in self.skus:
            dl = LeverSettings.defaults(
                self.assumptions,
                str(self.engine.sku_master.loc[s_id, "category"]),
                str(self.engine.sku_master.loc[s_id, "abc_class"]),
            )
            default_lookup[s_id] = (dl.inventory_cover_weeks, dl.service_target)

        default_by_sku: Dict[str, float] = {}
        surface_idx = surface.set_index(
            ["sku_id", "inventory_cover_weeks", "service_target"]
        )["total_eur"]
        for s_id in self.skus:
            key = (s_id,) + default_lookup[s_id]
            if key in surface_idx.index:
                default_by_sku[s_id] = float(surface_idx.loc[key])
        missing = set(self.skus) - set(default_by_sku)
        if missing:
            # a class default was not among eval_points for some reason —
            # evaluate it directly rather than silently falling back to A
            for s_id in missing:
                cov, srv = default_lookup[s_id]
                dl = self.base_levers.replace(
                    inventory_cover_weeks=cov, service_target=srv
                )
                res = self.engine.run_scenario(self.line_id, dl)
                row = res.sku_month[res.sku_month["sku_id"] == s_id]
                default_by_sku[s_id] = float(
                    row["lost_sales_eur"].sum()
                    + row["excess_obsolescence_eur"].sum()
                    + row["working_capital_cost_eur"].sum()
                )
        default_cost = pd.Series(default_by_sku)

        c_min, c_max = min(cover_values), max(cover_values)
        s_min, s_max = min(service_values), max(service_values)

        out = pd.DataFrame(
            {
                "sku_id": best.index,
                "line_id": self.line_id,
                "optimal_cover_weeks": best["inventory_cover_weeks"].to_numpy(),
                "optimal_service_target": best["service_target"].to_numpy(),
                "total_cost_at_optimum_eur": best["total_eur"].to_numpy(),
                "total_cost_at_default_eur": default_cost.reindex(
                    best.index
                ).to_numpy(),
                "attribution_mode": mode,
            }
        )
        out["saving_eur"] = (
            out["total_cost_at_default_eur"] - out["total_cost_at_optimum_eur"]
        )
        # A boundary optimum. NOTE this is NOT "the optimum lies outside the
        # searched range": the grid spans the full admissible lever range from
        # assumptions.levers, which is a POLICY CONSTRAINT, not a search
        # window. A SKU whose optimum sits at service_target = 0.995 is telling
        # you the constraint binds — the correct operational answer is the
        # boundary. So the flag is informative, but these rows are NOT dropped
        # from training by default (see PolicyModel.drop_edge_optima).
        # On the MVD line most SKUs are boundary-optimal on service, which is
        # the same fact D-059 recorded: on a 52% margin, long-shelf-life line,
        # service is a dominated choice rather than a trade-off.
        out["edge_optimum_flag"] = (
            out["optimal_cover_weeks"].isin([c_min, c_max])
            | out["optimal_service_target"].isin([s_min, s_max])
        )
        out["assumption_fingerprint"] = self.engine.assumption_fingerprint
        return out.reset_index(drop=True)


def optimise_all_lines(
    engine: TradeOffEngine,
    n_cover: int = 7,
    n_service: int = 6,
    mode: str = "separable",
) -> pd.DataFrame:
    """Sweep every line, returning one policy row per SKU across the portfolio.

    Required, not optional (D-065). Within a single line, category, line speed,
    shelf life, price, standard cost, case size, MOQ and minimum run are all
    CONSTANT — nine of the ten attribute features have zero variance, which
    forces the policy model to reduce to the ABC-class mean it is benchmarked
    against. Fitting on one line cannot discriminate; it is not a weak result
    but an uninterpretable one.

    The optimiser is already line-agnostic, so this is a loop, not new logic.
    """
    frames: List[pd.DataFrame] = []
    for line_id in engine.line_master["line_id"]:
        if not engine.line_skus(line_id):
            continue
        opt = PolicyOptimiser(engine, line_id)
        cover, service = opt.default_grid(n_cover=n_cover, n_service=n_service)
        frames.append(opt.optimise(cover, service, mode=mode))
    if not frames:
        raise PolicyModelViolation("no line produced a policy")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# PART 1b — D-071. Joint per-class search, superseding optimise_all_lines()
#
# The functions above search UNIFORM line-wide (cover, service) points and
# then, for each SKU, cherry-pick whichever uniform run happened to be
# cheapest for it. That combination — "SKU A at its own optimum, SKU B at a
# DIFFERENT optimum, simultaneously" — was never actually simulated, and
# capacity was verified only for each individual uniform run, never for the
# real combined policy. D-071 retracts that result.
#
# This section fixes it: cover and service are searched PER CLASS, and every
# candidate combination is ONE REAL run_scenario() call with a full per-class
# LeverSettings. Whatever is returned as "optimal" is a plan that was
# actually, jointly simulated — capacity infeasible candidates are excluded
# before cost is even compared, not discovered after the fact.
#
# Scope decision (stated, not hidden): forecast_bias_correction and
# min_run_hours are held at fixed scalars DURING THIS SEARCH. Not an engine
# limitation — D-071 made both per-class-capable — but a full 3-class x
# 4-lever grid is 3^12 combinations, computationally infeasible in the
# available time. Cover is what drives batch size and therefore aggregate
# workload, which is where the capacity-feasibility defect actually lived;
# widening the search to all four levers per class is a legitimate future
# extension, not attempted here.
# ---------------------------------------------------------------------------

ABC_CLASSES = ("A", "B", "C")


def joint_policy_grid(
    assumptions: Dict[str, Any], n: int = 3
) -> Tuple[List[float], List[float]]:
    """n values per lever from assumptions.levers.*.range — same principle as
    every prior grid, no literals."""
    lev = assumptions["levers"]
    c_lo, c_hi = (float(v) for v in lev["inventory_cover_weeks"]["range"])
    s_lo, s_hi = (float(v) for v in lev["service_target_by_abc"]["range"])
    cover = [round(v, 4) for v in np.linspace(c_lo, c_hi, n)]
    service = [round(v, 4) for v in np.linspace(s_lo, s_hi, n)]
    return cover, service


def search_three_lever_policy(
    engine: TradeOffEngine,
    line_id: str,
    cover_values: Sequence[float],
    bias_values: Sequence[float],
    min_run_values: Sequence[float],
    extra_combos: Optional[Sequence[Tuple[float, float, float, float, float]]] = None,
) -> pd.DataFrame:
    """D-079. Full factorial over (cover_A, cover_B, cover_C, bias, min_run) —
    service_target is FROZEN at each SKU's ABC-class default (assumptions.abc),
    not searched. Bias and min_run are uniform per line (not per class),
    matching how Change 1's confirmatory sweep treated them.

    This replaces search_joint_policy() as the live Step 7 search. That
    function is UNCHANGED and left in the file — the 2-lever (cover, service)
    result it produces is retracted (D-079), not deleted, since Step 6's own
    tests still exercise LeverSettings with a scalar service_target and
    nothing here should touch that.

    Grid size at the default n=3/3/4: 3^3 x 3 x 4 = 324 combinations per
    line — small enough to run in seconds, unlike the earlier 2-lever
    729-combination grid this replaces.
    """
    skus = engine.line_skus(line_id)
    if not skus:
        raise PolicyModelViolation(f"no SKUs allocated to line {line_id!r}")
    category = str(engine.sku_master.loc[skus[0], "category"])
    frozen_service = {
        c: float(engine.assumptions["abc"][c]["service_floor"])
        for c in ("A", "B", "C")
    }

    rows: List[Dict[str, Any]] = []
    combos = set(
        itertools.product(cover_values, cover_values, cover_values,
                           bias_values, min_run_values)
    )
    for extra in extra_combos or []:
        combos.add(tuple(round(float(v), 6) for v in extra))

    for cov_a, cov_b, cov_c, bias, mrh in combos:
        levers = LeverSettings(
            service_target=dict(frozen_service),
            inventory_cover_weeks={"A": cov_a, "B": cov_b, "C": cov_c},
            forecast_bias_correction=float(bias),
            min_run_hours=float(mrh),
        ).validate(engine.assumptions)
        res = engine.run_scenario(line_id, levers)
        s = res.sku_month.merge(
            engine.sku_master[["abc_class"]], left_on="sku_id", right_index=True
        )
        achieved_by_class = {
            c: float(g["shipped_units"].sum() / g["demand_units"].sum())
            if g["demand_units"].sum() > 0 else float("nan")
            for c, g in s.groupby("abc_class")
        }
        fill_rate = 1.0 - (
            s["lost_units"].sum() / s["demand_units"].sum()
            if s["demand_units"].sum() > 0 else float("nan")
        )
        overhang_mask = (s["production_units"] == 0) & (
            s["stock_open_units"] > s["target_stock_units"]
        )
        overhang_months = int(overhang_mask.sum())
        overhang_cost_share = float(
            (s.loc[overhang_mask, "lost_sales_eur"].sum()
             + s.loc[overhang_mask, "excess_obsolescence_eur"].sum()
             + s.loc[overhang_mask, "working_capital_cost_eur"].sum())
            / max(1e-9, s["lost_sales_eur"].sum() + s["excess_obsolescence_eur"].sum()
                  + s["working_capital_cost_eur"].sum())
        )
        rows.append(
            {
                "line_id": line_id,
                "cover_A": cov_a, "cover_B": cov_b, "cover_C": cov_c,
                "bias_correction": float(bias), "min_run_hours": float(mrh),
                **res.totals,
                "service_achieved_A": achieved_by_class.get("A", float("nan")),
                "service_achieved_B": achieved_by_class.get("B", float("nan")),
                "service_achieved_C": achieved_by_class.get("C", float("nan")),
                "unit_fill_rate": fill_rate,
                "overhang_sku_months": overhang_months,
                "overhang_sku_months_total": int(len(s)),
                "overhang_cost_share": overhang_cost_share,
                "capacity_shortfall_total": float(
                    res.line_month["capacity_shortfall_units"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def best_feasible_three_lever(search_result: pd.DataFrame) -> pd.Series:
    """Same feasibility-first selection rule as best_feasible_policy()."""
    feasible = search_result[search_result["capacity_shortfall_total"] <= 1e-6]
    if feasible.empty:
        raise PolicyModelViolation(
            "no combination in the three-lever grid is capacity-feasible"
        )
    return feasible.loc[feasible["total_economic_cost_eur"].idxmin()]


def search_all_lines_three_lever(
    engine: TradeOffEngine, n_cover: int = 3,
    bias_values: Sequence[float] = (0.0, 0.5, 1.0),
    min_run_values: Sequence[float] = (4.0, 6.0, 9.0, 12.0),
) -> pd.DataFrame:
    """D-079 replacement for search_all_lines(). service_target is frozen at
    ABC defaults on every row — it is reported (service_achieved_*) never
    searched. Base comparator is the true simultaneous per-class-default
    scenario at bias=0, min_run=each category's own default — matching what
    was already computed as 'true_base' during the D-071/D-072 dashboard
    rebuild.
    """
    lev = engine.assumptions["levers"]
    c_lo, c_hi = (float(v) for v in lev["inventory_cover_weeks"]["range"])
    cover_values = [round(c_lo + i * (c_hi - c_lo) / (n_cover - 1), 4) for i in range(n_cover)]

    rows: List[Dict[str, Any]] = []
    for line_id in engine.line_master["line_id"]:
        skus = engine.line_skus(line_id)
        if not skus:
            continue
        category = str(engine.sku_master.loc[skus[0], "category"])
        default_cover = {
            c: float(engine.assumptions["abc"][c]["target_cover_weeks"])
            for c in ("A", "B", "C")
        }
        default_min_run = float(
            engine.assumptions["categories"][category]["min_run_hours"]
        )
        default_combo = (
            default_cover["A"], default_cover["B"], default_cover["C"],
            0.0, default_min_run,
        )
        search = search_three_lever_policy(
            engine, line_id, cover_values, bias_values, min_run_values,
            extra_combos=[default_combo],
        )
        best = best_feasible_three_lever(search)

        base_levers = LeverSettings.defaults_per_class(engine.assumptions, category)
        base_res = engine.run_scenario(line_id, base_levers)
        base_total = base_res.totals["total_economic_cost_eur"]

        row = best.to_dict()
        row["default_total_cost_eur"] = base_total
        row["saving_eur"] = base_total - best["total_economic_cost_eur"]
        row["assumption_fingerprint"] = engine.assumption_fingerprint
        rows.append(row)

    if not rows:
        raise PolicyModelViolation("no line produced a feasible three-lever policy")
    return pd.DataFrame(rows)


def bias_min_run_interaction(
    engine: TradeOffEngine, line_id: str,
    cover_combo: Dict[str, float],
    bias_values: Sequence[float] = (0.0, 0.5, 1.0),
    min_run_values: Sequence[float] = (4.0, 6.0, 9.0, 12.0),
) -> pd.DataFrame:
    """D-079 Change 5 reporting requirement: the bias x min_run interaction,
    at a FIXED cover combination, shown as a full grid INCLUDING cells where
    bias has zero effect (a real mechanism — the batch floor exceeds the
    plan adjustment — not suppressed or smoothed)."""
    skus = engine.line_skus(line_id)
    category = str(engine.sku_master.loc[skus[0], "category"])
    frozen_service = {
        c: float(engine.assumptions["abc"][c]["service_floor"]) for c in ("A", "B", "C")
    }
    rows = []
    for bias in bias_values:
        for mrh in min_run_values:
            levers = LeverSettings(
                service_target=dict(frozen_service),
                inventory_cover_weeks=dict(cover_combo),
                forecast_bias_correction=float(bias), min_run_hours=float(mrh),
            ).validate(engine.assumptions)
            res = engine.run_scenario(line_id, levers)
            rows.append({
                "line_id": line_id, "bias_correction": bias, "min_run_hours": mrh,
                "total_economic_cost_eur": res.totals["total_economic_cost_eur"],
            })
    df = pd.DataFrame(rows)
    pivoted = df.pivot(index="min_run_hours", columns="bias_correction", values="total_economic_cost_eur")
    pivoted["bias_has_zero_effect"] = pivoted.nunique(axis=1) == 1
    return pivoted.reset_index()


def search_joint_policy(
    engine: TradeOffEngine,
    line_id: str,
    cover_values: Sequence[float],
    service_values: Sequence[float],
    bias_correction: float = 0.0,
    min_run_hours: Optional[float] = None,
    extra_combos: Optional[Sequence[Tuple[float, float, float, float, float, float]]] = None,
) -> pd.DataFrame:
    """Full factorial over (cover_A, cover_B, cover_C, service_A, service_B,
    service_C) — every combination is ONE REAL joint simulation.

    This is the entire fix: each row's capacity_shortfall_total is the TRUE
    aggregate answer for that exact combination, because that exact
    combination is what was simulated — not an inference stitched from
    separate single-class runs.

    `extra_combos` guarantees specific points (typically each class's actual
    default policy) are evaluated even if the grid is too coarse to land on
    them exactly. Without this, D-067's failure mode recurs one level up: a
    coarse grid can score worse everywhere it samples than an untested
    default point, making "optimal" appear worse than "default" — which is
    structurally impossible when default is itself an admissible candidate.
    """
    if min_run_hours is None:
        skus = engine.line_skus(line_id)
        if not skus:
            raise PolicyModelViolation(f"no SKUs allocated to line {line_id!r}")
        category = str(engine.sku_master.loc[skus[0], "category"])
        min_run_hours = float(
            engine.assumptions["categories"][category]["min_run_hours"]
        )

    rows: List[Dict[str, Any]] = []
    combos = set(
        itertools.product(cover_values, cover_values, cover_values,
                           service_values, service_values, service_values)
    )
    for extra in extra_combos or []:
        combos.add(tuple(round(float(v), 6) for v in extra))

    for cov_a, cov_b, cov_c, svc_a, svc_b, svc_c in combos:
        levers = LeverSettings(
            service_target={"A": svc_a, "B": svc_b, "C": svc_c},
            inventory_cover_weeks={"A": cov_a, "B": cov_b, "C": cov_c},
            forecast_bias_correction=float(bias_correction),
            min_run_hours=float(min_run_hours),
        ).validate(engine.assumptions)
        res = engine.run_scenario(line_id, levers)
        rows.append(
            {
                "line_id": line_id,
                "cover_A": cov_a, "cover_B": cov_b, "cover_C": cov_c,
                "service_A": svc_a, "service_B": svc_b, "service_C": svc_c,
                "bias_correction": float(bias_correction),
                "min_run_hours": float(min_run_hours),
                **res.totals,
                "capacity_shortfall_total": float(
                    res.line_month["capacity_shortfall_units"].sum()
                ),
                "utilisation_max": float(res.line_month["utilisation"].max()),
            }
        )
    return pd.DataFrame(rows)


def best_feasible_policy(search_result: pd.DataFrame) -> pd.Series:
    """Filter to FEASIBLE candidates first, then pick the cheapest.

    This ordering is the whole point: an infeasible-but-cheap combination is
    never eligible, so whatever this returns is a plan that could actually
    run. Raises if nothing in the grid is feasible — that is itself a finding
    (the grid's range cannot satisfy capacity anywhere) and must not be
    silently papered over by returning the least-bad infeasible point.
    """
    feasible = search_result[search_result["capacity_shortfall_total"] <= 1e-6]
    if feasible.empty:
        raise PolicyModelViolation(
            "no combination in the search grid is capacity-feasible — every "
            "candidate breached capacity. Widen the grid or lower demand "
            "before trusting any 'optimal' figure from this search."
        )
    return feasible.loc[feasible["total_economic_cost_eur"].idxmin()]


def search_all_lines(
    engine: TradeOffEngine, n: int = 3, bias_correction: float = 0.0
) -> pd.DataFrame:
    """The feasible optimum for every line: one row each.

    default_total_cost_eur is each line's TRUE per-class default policy —
    each class at ITS OWN class's default cover/service (LeverSettings.
    defaults_per_class()), the same "Base" concept the dashboard mockup
    uses — not the old convention of applying class A's policy to every SKU
    on the line. That default combination is guaranteed to be an evaluated
    candidate (extra_combos), so the optimum can never come out worse than
    the baseline it is measured against — the exact D-067 failure mode,
    recurring one level up if left unguarded.
    """
    cover_values, service_values = joint_policy_grid(engine.assumptions, n=n)
    rows: List[Dict[str, Any]] = []

    for line_id in engine.line_master["line_id"]:
        skus = engine.line_skus(line_id)
        if not skus:
            continue
        category = str(engine.sku_master.loc[skus[0], "category"])
        default_levers = LeverSettings.defaults_per_class(
            engine.assumptions, category
        )
        default_combo = (
            default_levers.inventory_cover_weeks["A"],
            default_levers.inventory_cover_weeks["B"],
            default_levers.inventory_cover_weeks["C"],
            default_levers.service_target["A"],
            default_levers.service_target["B"],
            default_levers.service_target["C"],
        )

        search = search_joint_policy(
            engine, line_id, cover_values, service_values,
            bias_correction=bias_correction,
            extra_combos=[default_combo],
        )
        best = best_feasible_policy(search)

        default_res = engine.run_scenario(line_id, default_levers)
        default_total = default_res.totals["total_economic_cost_eur"]

        row = best.to_dict()
        row["default_total_cost_eur"] = default_total
        row["saving_eur"] = default_total - best["total_economic_cost_eur"]
        row["assumption_fingerprint"] = engine.assumption_fingerprint
        rows.append(row)

    if not rows:
        raise PolicyModelViolation("no line produced a feasible policy")
    return pd.DataFrame(rows)


def expand_to_sku_level(
    line_results: pd.DataFrame, engine: TradeOffEngine
) -> pd.DataFrame:
    """One row per SKU, inheriting its own class's chosen policy from its
    line's winning combination.

    Re-reads the ALREADY-COMPUTED winning scenario's sku_month rather than
    re-simulating — the per-SKU lost_sales_eur / excess_obsolescence_eur /
    working_capital_cost_eur come directly from the one joint simulation that
    produced the winning row. Conversion cost is NOT split per SKU: it stays
    a line total, repeated and explicitly labelled as such (D-066's finding
    that it isn't separable per SKU still holds — the difference from the
    retracted version is this file no longer pretends a per-SKU split is a
    policy result).
    """
    rows: List[Dict[str, Any]] = []
    for _, line_row in line_results.iterrows():
        line_id = line_row["line_id"]
        levers = LeverSettings(
            service_target={
                c: float(line_row[f"service_{c}"]) for c in ABC_CLASSES
            },
            inventory_cover_weeks={
                c: float(line_row[f"cover_{c}"]) for c in ABC_CLASSES
            },
            forecast_bias_correction=float(line_row["bias_correction"]),
            min_run_hours=float(line_row["min_run_hours"]),
        ).validate(engine.assumptions)
        res = engine.run_scenario(line_id, levers)
        s = res.sku_month
        line_conv = float(line_row["conversion_cost_eur"])

        skus = engine.line_skus(line_id)
        for sku_id in skus:
            abc_class = str(engine.sku_master.loc[sku_id, "abc_class"])
            sku_rows = s[s["sku_id"] == sku_id]
            rows.append(
                {
                    "sku_id": sku_id,
                    "line_id": line_id,
                    "abc_class": abc_class,
                    "cover_weeks": float(line_row[f"cover_{abc_class}"]),
                    "service_target": float(line_row[f"service_{abc_class}"]),
                    "lost_sales_eur": float(sku_rows["lost_sales_eur"].sum()),
                    "excess_obsolescence_eur": float(
                        sku_rows["excess_obsolescence_eur"].sum()
                    ),
                    "working_capital_cost_eur": float(
                        sku_rows["working_capital_cost_eur"].sum()
                    ),
                    "line_conversion_cost_eur": line_conv,
                    "assumption_fingerprint": res.assumption_fingerprint,
                }
            )
    return pd.DataFrame(rows)


def sku_level_for_policy_model(sku_level: pd.DataFrame) -> pd.DataFrame:
    """Reshapes expand_to_sku_level()'s output into what PolicyModel expects.

    PolicyModel (Part 2, below) was built against the RETIRED per-SKU search
    and still reads column names optimal_cover_weeks / optimal_service_target
    — kept as-is rather than renamed throughout, since PolicyModel's own
    logic (feature assembly, LOO evaluation, SHAP) is unaffected by D-071 and
    does not need to change, only the column names feeding it do.

    edge_optimum_flag is set False for every row: the OLD flag existed
    because the retired per-SKU search commonly landed on grid boundaries as
    an ARTEFACT of attributing a uniform-line run to one SKU. These rows are
    genuinely joint-optimal, feasibility-filtered points — a class value
    sitting at the edge of the tested range here means the constraint
    binds, not that the search failed, so there is no reason to exclude
    these rows from training the way the old flag was designed to.
    """
    out = sku_level.rename(
        columns={
            "cover_weeks": "optimal_cover_weeks",
            "service_target": "optimal_service_target",
        }
    ).copy()
    out["edge_optimum_flag"] = False
    return out


# ---------------------------------------------------------------------------
# PART 2 — the policy model
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    target: str
    feature_set: str
    estimator: str
    mae: float
    r2: float
    naive_mae: float
    beats_naive: bool
    n_train: int


class PolicyModel:
    """Fit SKU attributes to optimal parameters.

    Two feature sets, fitted and reported SEPARATELY:

      FULL    every attribute, including the two derived from demand history
      LAUNCH  excludes irreducible_volatility_cv and chronic_bias_l1

    LAUNCH is the set that will actually be used, because a new launch has no
    demand history. Reporting only FULL would overstate real-world performance,
    since the most predictive features are precisely the ones a launch lacks.
    """

    def __init__(
        self,
        optimal_policy: pd.DataFrame,
        sku_master: pd.DataFrame,
        demand_characteristics: pd.DataFrame,
        line_master: pd.DataFrame,
        drop_edge_optima: bool = False,
    ) -> None:
        self.seed = SEED
        self.drop_edge_optima = drop_edge_optima
        self.frame = self._assemble(
            optimal_policy, sku_master, demand_characteristics, line_master
        )
        self.fitted: Dict[Tuple[str, str, str], Any] = {}
        self.encoders: Dict[str, Any] = {}

    # -- data -----------------------------------------------------------

    @staticmethod
    def _assemble(
        optimal_policy: pd.DataFrame,
        sku_master: pd.DataFrame,
        demand_characteristics: pd.DataFrame,
        line_master: pd.DataFrame,
    ) -> pd.DataFrame:
        sm = sku_master.copy()
        if sm.index.name != "sku_id":
            sm = sm.set_index("sku_id")
        dc = demand_characteristics.copy()
        if dc.index.name != "sku_id":
            dc = dc.set_index("sku_id")
        lm = line_master.copy()
        if lm.index.name != "line_id":
            lm = lm.set_index("line_id")

        f = optimal_policy.set_index("sku_id").copy()
        for col in ATTRIBUTE_FEATURES:
            if col in sm.columns:
                f[col] = sm[col].reindex(f.index)
        for col in HISTORY_DERIVED_FEATURES:
            if col in dc.columns:
                f[col] = dc[col].reindex(f.index)
        f["line_speed_units_hr"] = (
            f["line_id"].map(lm["units_per_hour"]).astype(float)
        )
        return f.reset_index()

    def feature_columns(self, feature_set: str) -> List[str]:
        if feature_set == "FULL":
            return ATTRIBUTE_FEATURES + HISTORY_DERIVED_FEATURES
        if feature_set == "LAUNCH":
            return list(ATTRIBUTE_FEATURES)
        raise PolicyModelViolation(f"unknown feature_set {feature_set!r}")

    def training_frame(self) -> pd.DataFrame:
        """Rows used for fitting.

        Boundary optima are RETAINED by default. The grid spans the admissible
        lever range, so a boundary result is the correct policy under the
        constraint, not a failed search. Dropping them would discard most of
        the MVD line and bias training toward the minority of SKUs whose
        optimum happens to be interior.
        """
        f = self.frame
        if self.drop_edge_optima and "edge_optimum_flag" in f.columns:
            f = f[~f["edge_optimum_flag"].astype(bool)]
        return f

    def _design_matrix(self, frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        cols = [c for c in self.feature_columns(feature_set) if c in frame.columns]
        X = frame[cols].copy()
        return pd.get_dummies(X, columns=[c for c in ("category", "abc_class") if c in X.columns])

    # -- fit and evaluate ------------------------------------------------

    def evaluate(self) -> pd.DataFrame:
        """Leave-one-out cross-validation, NOT a train/test split.

        60 SKUs — and fewer after edge optima are dropped — will not support a
        holdout: a single unlucky split would dominate the reported score.

        A naive baseline (predict the ABC-class mean) is reported alongside
        every model. If neither estimator beats it, that is the honest result
        and it is more informative than a tuned number. Do not remove the
        baseline to make the comparison look better.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import LeaveOneOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        frame = self.training_frame()
        results: List[FitResult] = []

        for target in TARGETS:
            y_all = frame[target].to_numpy(dtype=float)
            grp = frame["abc_class"] if "abc_class" in frame.columns else None

            for feature_set in ("FULL", "LAUNCH"):
                X_all = self._design_matrix(frame, feature_set)
                X_np = X_all.to_numpy(dtype=float)

                for name in ("xgboost", "ridge"):
                    preds = np.empty_like(y_all)
                    naive = np.empty_like(y_all)
                    for tr, te in LeaveOneOut().split(X_np):
                        if name == "xgboost":
                            est = XGBRegressor(
                                n_estimators=200,
                                max_depth=3,
                                learning_rate=0.08,
                                subsample=0.9,
                                random_state=self.seed,
                                verbosity=0,
                            )
                        else:
                            est = make_pipeline(
                                StandardScaler(), Ridge(alpha=1.0, random_state=None)
                            )
                        est.fit(X_np[tr], y_all[tr])
                        preds[te] = est.predict(X_np[te])
                        if grp is not None:
                            cls = grp.iloc[te[0]]
                            same = grp.iloc[tr] == cls
                            naive[te] = (
                                y_all[tr][same.to_numpy()].mean()
                                if same.any()
                                else y_all[tr].mean()
                            )
                        else:
                            naive[te] = y_all[tr].mean()

                    mae = float(mean_absolute_error(y_all, preds))
                    naive_mae = float(mean_absolute_error(y_all, naive))
                    results.append(
                        FitResult(
                            target=target,
                            feature_set=feature_set,
                            estimator=name,
                            mae=mae,
                            r2=float(r2_score(y_all, preds)),
                            naive_mae=naive_mae,
                            beats_naive=bool(mae < naive_mae),
                            n_train=int(len(y_all)),
                        )
                    )

        return pd.DataFrame([r.__dict__ for r in results])

    def fit_final(self) -> None:
        """Fit on all rows, per target per feature set, for prediction and SHAP."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        frame = self.training_frame()
        for target in TARGETS:
            y = frame[target].to_numpy(dtype=float)
            for feature_set in ("FULL", "LAUNCH"):
                X = self._design_matrix(frame, feature_set)
                self.encoders[feature_set] = list(X.columns)
                xgb = XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    random_state=self.seed,
                    verbosity=0,
                )
                xgb.fit(X.to_numpy(dtype=float), y)
                self.fitted[(target, feature_set, "xgboost")] = xgb

                ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                ridge.fit(X.to_numpy(dtype=float), y)
                self.fitted[(target, feature_set, "ridge")] = ridge

    def explain(self, target: str, feature_set: str = "LAUNCH"):
        """SHAP values for the tree model. Returns (shap_values, X)."""
        import shap

        key = (target, feature_set, "xgboost")
        if key not in self.fitted:
            raise PolicyModelViolation("call fit_final() before explain()")
        X = self._design_matrix(self.training_frame(), feature_set)
        explainer = shap.TreeExplainer(self.fitted[key])
        return explainer.shap_values(X), X

    # -- the operational output -----------------------------------------

    def predict_launch(
        self, attributes: Dict[str, Any], estimator: str = "xgboost"
    ) -> Dict[str, float]:
        """Recommend an opening policy for a SKU with no demand history.

        Refuses on a missing attribute rather than imputing one. A launch policy
        built on a silently-defaulted feature is worse than no recommendation:
        the planner cannot see that it happened.
        """
        required = [c for c in ATTRIBUTE_FEATURES]
        missing = [c for c in required if c not in attributes]
        if missing:
            raise PolicyModelViolation(
                f"cannot recommend a launch policy — missing attributes: "
                f"{missing}. Supply them or accept no recommendation; this "
                f"model does not impute."
            )
        leaked = [c for c in HISTORY_DERIVED_FEATURES if c in attributes]
        if leaked:
            raise PolicyModelViolation(
                f"{leaked} are derived from demand history and cannot exist for "
                f"a launch. Use the FULL model on an established SKU instead."
            )

        row = pd.DataFrame([{c: attributes[c] for c in required}])
        X = pd.get_dummies(
            row, columns=[c for c in ("category", "abc_class") if c in row.columns]
        )
        cols = self.encoders.get("LAUNCH")
        if cols is None:
            raise PolicyModelViolation("call fit_final() before predict_launch()")
        X = X.reindex(columns=cols, fill_value=0)

        out: Dict[str, float] = {}
        for target in TARGETS:
            est = self.fitted[(target, "LAUNCH", estimator)]
            out[target] = float(est.predict(X.to_numpy(dtype=float))[0])
        return out
