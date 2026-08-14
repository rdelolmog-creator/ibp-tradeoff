"""Step 6 — MVD trade-off engine.

Forward-simulates one production line over `schema.meta.horizon_months` under a
given set of the four decision levers, and returns the four economic outputs.

Design constraints this module is built to satisfy (see decision-log.md):

D-038  Forward-simulating, not history-replaying. History is used only to
       characterise demand; it is never recomputed.
D-041  Deterministic. One demand path, no RNG anywhere. Lost sales are derived
       analytically from `irreducible_volatility_cv` via the standard normal
       loss function, not by sampling futures.
D-040  `min_run_hours` is expected to move three outputs at once. Nothing here
       special-cases that; it must emerge.
§9     Circularity guardrail. `forecast_bias_correction` enters in exactly one
       place — `project_plan()` — and moves the PLAN. Demand is never a
       function of any lever, and no cost is ever a direct function of a lever
       value. Costs change only because quantities change.

Nothing is hard-coded: no paths, no SKU ids, no line ids, no rates. Every
economic constant resolves to a key in config/assumptions.yaml, and every
structural expectation to config/schema.yaml. Contains no Colab-specific
calls, so app.py imports it unchanged at Step 10.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WEEKS_PER_MONTH = 52.0 / 12.0


class EngineViolation(Exception):
    """Raised when a lever, contract or capacity invariant is breached."""


# ---------------------------------------------------------------------------
# Levers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeverSettings:
    """The four decision levers. Ranges are validated against assumptions.yaml.

    Scope is frozen at four (architecture §5). A fifth lever does not belong
    here; it belongs in a conversation about what comes out.
    """

    service_target: float
    inventory_cover_weeks: float
    forecast_bias_correction: float
    min_run_hours: float

    _RANGE_KEYS = {
        "service_target": "service_target_by_abc",
        "inventory_cover_weeks": "inventory_cover_weeks",
        "forecast_bias_correction": "forecast_bias_correction",
        "min_run_hours": "min_run_hours",
    }

    def validate(self, assumptions: Dict[str, Any]) -> "LeverSettings":
        levers = assumptions["levers"]
        for field_name, key in self._RANGE_KEYS.items():
            lo, hi = (float(v) for v in levers[key]["range"])
            value = float(getattr(self, field_name))
            if not (lo - 1e-12 <= value <= hi + 1e-12):
                raise EngineViolation(
                    f"{field_name}={value} is outside assumptions.levers."
                    f"{key}.range = [{lo}, {hi}]"
                )
        return self

    @classmethod
    def defaults(
        cls, assumptions: Dict[str, Any], category: str, abc_class: str = "A"
    ) -> "LeverSettings":
        """Slider defaults, read from config — never literals in code."""
        abc = assumptions["abc"][abc_class]
        return cls(
            service_target=float(abc["service_floor"]),
            inventory_cover_weeks=float(abc["target_cover_weeks"]),
            forecast_bias_correction=0.0,
            min_run_hours=float(assumptions["categories"][category]["min_run_hours"]),
        ).validate(assumptions)

    def replace(self, **kwargs: float) -> "LeverSettings":
        data = {
            "service_target": self.service_target,
            "inventory_cover_weeks": self.inventory_cover_weeks,
            "forecast_bias_correction": self.forecast_bias_correction,
            "min_run_hours": self.min_run_hours,
        }
        data.update(kwargs)
        return LeverSettings(**data)

    def as_dict(self) -> Dict[str, float]:
        return {
            "service_target": self.service_target,
            "inventory_cover_weeks": self.inventory_cover_weeks,
            "forecast_bias_correction": self.forecast_bias_correction,
            "min_run_hours": self.min_run_hours,
        }


# ---------------------------------------------------------------------------
# line_master — declared in schema.yaml, produced by nothing until now
# ---------------------------------------------------------------------------


def build_line_master(
    assumptions: Dict[str, Any], schema: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Build `canonical.line_master` from config.

    schema.yaml declares this table but no pipeline step emits it, because line
    economics are assumptions, not observed data. Storing it as a generated
    artefact would create a second place to change them. It is therefore built
    at runtime, and validated against the schema declaration so the two cannot
    drift apart silently.
    """
    plants = assumptions["plants"]
    cats = assumptions["categories"]
    econ = assumptions["plant_economics"]

    rows: List[Dict[str, Any]] = []
    for line_id, cfg in assumptions["lines"].items():
        plant_id = cfg["plant"]
        cat = cfg["category"]
        c = cats[cat]
        rows.append(
            {
                "line_id": line_id,
                "plant_id": plant_id,
                "source_system": plants[plant_id]["source_system"],
                "category": cat,
                "units_per_hour": float(c["line_speed_units_hr"]),
                "hours_available_month": float(econ["scheduled_hours_per_line_month"]),
                "changeover_hours": float(c["changeover_hours"]),
                "changeover_cash_eur": float(c["changeover_cash_eur"]),
                "equipment_age_years": float(cfg["equipment_age_years"]),
                "complexity_index": float(cfg["complexity_index"]),
                "labour_rate_eur_hr": float(econ["labour_rate_eur_hr"]),
                "yield_rate": float(c["yield_rate"]),
                "schedule_adherence": float(econ["schedule_adherence"]),
            }
        )
    out = pd.DataFrame(rows)

    if schema is not None:
        declared = list(schema["canonical"]["line_master"]["columns"])
        missing = [c for c in declared if c not in out.columns]
        if missing:
            raise EngineViolation(
                f"build_line_master() does not satisfy schema.canonical."
                f"line_master; missing {missing}"
            )
    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """One lever setting, fully evaluated.

    `assumption_fingerprint` travels with every result so no number can be
    shown without the assumption set that produced it (architecture §10).
    """

    line_id: str
    levers: LeverSettings
    sku_month: pd.DataFrame
    line_month: pd.DataFrame
    totals: Dict[str, float]
    assumption_fingerprint: str

    def summary_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"line_id": self.line_id}
        row.update(self.levers.as_dict())
        row.update(self.totals)
        row["assumption_fingerprint"] = self.assumption_fingerprint
        return row


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TradeOffEngine:
    """Levers in, four economic outputs out, for one line over the horizon."""

    def __init__(
        self,
        assumptions: Dict[str, Any],
        schema: Dict[str, Any],
        clean_master: pd.DataFrame,
        sku_master: pd.DataFrame,
        demand_characteristics: pd.DataFrame,
        flagged_sku_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.assumptions = assumptions
        self.schema = schema
        self.horizon_months = int(schema["meta"]["horizon_months"])

        self.clean_master = clean_master.copy()
        self.clean_master["month"] = pd.to_datetime(self.clean_master["month"])

        self.sku_master = sku_master.set_index("sku_id", drop=False).copy()
        self.demand_characteristics = demand_characteristics.copy()
        if self.demand_characteristics.index.name != "sku_id":
            self.demand_characteristics = self.demand_characteristics.set_index("sku_id")

        # D-035: SKUs whose bias estimate Step 5a could not confidently
        # recover receive zero correction whatever the lever says. Taken as
        # given from Step 5a's own verdicts; never re-derived here.
        self.flagged_sku_ids = set(flagged_sku_ids or [])

        self.line_master = build_line_master(assumptions, schema).set_index(
            "line_id", drop=False
        )

        self._validate_inputs()
        self.assumption_fingerprint = self._fingerprint()

    # -- setup ----------------------------------------------------------

    def _validate_inputs(self) -> None:
        skus = set(self.clean_master["sku_id"])
        missing_master = skus - set(self.sku_master.index)
        if missing_master:
            raise EngineViolation(
                f"{len(missing_master)} sku_id in clean_master have no row in "
                f"sku_master: {sorted(missing_master)[:5]}"
            )
        missing_dc = skus - set(self.demand_characteristics.index)
        if missing_dc:
            raise EngineViolation(
                f"{len(missing_dc)} sku_id in clean_master have no row in "
                f"demand_characteristics: {sorted(missing_dc)[:5]}"
            )
        forbidden = {"category_chronic_bias", "sku_idiosyncratic_bias",
                     "total_base_chronic_bias"}
        leaked = forbidden & set(self.demand_characteristics.columns)
        if leaked:
            raise EngineViolation(
                f"ground-truth columns present in demand_characteristics: "
                f"{sorted(leaked)}. The engine must never see encoded truth."
            )

    def _fingerprint(self) -> str:
        payload = json.dumps(self.assumptions, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        version = self.assumptions.get("meta", {}).get("assumptions_version", "?")
        return f"v{version}:{digest}"

    def line_skus(self, line_id: str) -> List[str]:
        """SKUs allocated to a line. Static allocation — see O-10, open."""
        col = "line_id" if "line_id" in self.sku_master.columns else "line_code"
        return sorted(self.sku_master.index[self.sku_master[col] == line_id].tolist())

    def _line_cfg(self, line_id: str) -> pd.Series:
        if line_id not in self.line_master.index:
            raise EngineViolation(f"Unknown line_id {line_id!r}")
        return self.line_master.loc[line_id]

    # -- 1. demand ------------------------------------------------------

    def project_demand(self, line_id: str) -> pd.DataFrame:
        """A deterministic forward demand path per SKU, from history alone.

        Classical decomposition on the observed series — seasonal index by
        calendar month, OLS trend on the deseasonalised series, level from the
        trailing year. Promo months are excluded from level and trend so a
        promo calendar does not masquerade as growth.

        Deliberately reads NOTHING from assumptions.yaml. The generator's own
        seasonality amplitude, phase and trend parameters live there; using
        them would hand the engine the answer the data is supposed to carry.
        This function takes no levers, and must not: demand is exogenous
        (§9 circularity guardrail).
        """
        skus = self.line_skus(line_id)
        hist = self.clean_master[self.clean_master["sku_id"].isin(skus)].copy()
        hist = hist.sort_values(["sku_id", "month"])

        last_month = hist["month"].max()
        future_months = pd.date_range(
            last_month + pd.offsets.MonthBegin(1),
            periods=self.horizon_months,
            freq="MS",
        )

        rows: List[Dict[str, Any]] = []
        for sku_id, g in hist.groupby("sku_id", sort=True):
            g = g.sort_values("month")
            y = g["actual_units"].astype(float).to_numpy()
            months = g["month"].to_numpy()
            promo = g["promo_flag"].astype(bool).to_numpy()
            cal = pd.DatetimeIndex(months).month.to_numpy()

            # seasonal index: ratio to a centred 12-month moving average
            s = pd.Series(y)
            cma = s.rolling(12, center=True, min_periods=12).mean()
            ratio = (s / cma).to_numpy()
            idx = np.ones(13, dtype=float)
            for m in range(1, 13):
                sel = (cal == m) & np.isfinite(ratio) & (~promo)
                if sel.any():
                    idx[m] = float(np.nanmean(ratio[sel]))
            core = idx[1:]
            core = np.where(np.isfinite(core) & (core > 0), core, 1.0)
            core = core / core.mean()          # normalise: a full year is neutral
            idx[1:] = core

            deseas = y / idx[cal]

            # level and trend from non-promo months only
            ok = ~promo
            t = np.arange(len(y), dtype=float)
            if ok.sum() >= 4:
                slope, intercept = np.polyfit(t[ok], deseas[ok], 1)
            else:                                # pragma: no cover - guard only
                slope, intercept = 0.0, float(deseas.mean())

            n = len(y)
            for h, month in enumerate(future_months, start=1):
                base = intercept + slope * (n - 1 + h)
                base = max(base, 0.0)
                rows.append(
                    {
                        "sku_id": sku_id,
                        "month": month,
                        "horizon_month": h,
                        "seasonal_index": float(idx[month.month]),
                        "demand_units": float(base * idx[month.month]),
                    }
                )
        return pd.DataFrame(rows)

    # -- 2. plan --------------------------------------------------------

    def project_plan(self, demand: pd.DataFrame, levers: LeverSettings) -> pd.DataFrame:
        """The forecast the planner works to, after bias correction.

        THE ONLY PLACE `forecast_bias_correction` ACTS. The uncorrected plan
        carries the chronic bias Step 5a detected; the lever removes a share of
        it. Demand is untouched. Every downstream cost moves only because the
        plan quantity moved — which is exactly what the circularity test in
        tests/test_engine.py asserts.

        `chronic_bias_l1` is in log units (log forecast - log actual), so it is
        converted with expm1 before use. `chronic_bias_pct_l1` is NOT read: it
        lives in the diagnostics file, outside Step 5a's fixed nine-column
        contract.
        """
        c = float(levers.forecast_bias_correction)
        out = demand.copy()
        bias_log = self.demand_characteristics["chronic_bias_l1"].reindex(
            out["sku_id"]
        ).to_numpy(dtype=float)
        bias_pct = np.expm1(np.nan_to_num(bias_log, nan=0.0))

        flagged = out["sku_id"].isin(self.flagged_sku_ids).to_numpy()
        applied = np.where(flagged, 0.0, c)          # D-035

        out["detected_bias_pct"] = bias_pct
        out["bias_correction_applied"] = applied
        out["plan_units"] = out["demand_units"].to_numpy() * (
            1.0 + (1.0 - applied) * bias_pct
        )
        return out

    # -- 3. simulate ----------------------------------------------------

    def simulate(
        self, line_id: str, levers: LeverSettings
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Roll the line forward month by month under one lever setting."""
        levers.validate(self.assumptions)
        cfg = self._line_cfg(line_id)
        econ = self.assumptions["plant_economics"]

        units_per_hour = float(cfg["units_per_hour"])
        yield_rate = float(cfg["yield_rate"])
        adherence = float(cfg["schedule_adherence"])
        changeover_hours = float(cfg["changeover_hours"])
        hours_available = float(cfg["hours_available_month"])
        overtime_step = float(econ["overtime_step_hours"])
        max_overtime = float(econ["max_overtime_hours_month"])

        demand = self.project_demand(line_id)
        plan = self.project_plan(demand, levers)

        skus = self.line_skus(line_id)
        z = _service_z(levers.service_target)
        cover_months = float(levers.inventory_cover_weeks) / WEEKS_PER_MONTH
        min_run_units = float(levers.min_run_hours) * units_per_hour

        cv = self.demand_characteristics["irreducible_volatility_cv"].reindex(skus)
        cv = cv.fillna(cv.median()).to_dict()
        moq = self.sku_master.loc[skus, "moq_units"].astype(float).to_dict()
        std_cost = self.sku_master.loc[skus, "std_cost_eur"].astype(float).to_dict()
        shelf_life_months = (
            self.sku_master.loc[skus, "shelf_life_days"].astype(float) / 30.44
        ).to_dict()

        # Opening position: the last observed closing stock. History supplies
        # the starting point; it is not recomputed (D-038).
        last_month = self.clean_master["month"].max()
        opening = (
            self.clean_master[
                (self.clean_master["month"] == last_month)
                & (self.clean_master["sku_id"].isin(skus))
            ]
            .set_index("sku_id")["stock_close_units"]
            .astype(float)
            .to_dict()
        )
        # FIFO ageing layers: [age_in_months, units]
        layers: Dict[str, List[List[float]]] = {
            s: ([[0.0, float(opening.get(s, 0.0))]] if opening.get(s, 0.0) > 0 else [])
            for s in skus
        }

        plan_ix = plan.set_index(["sku_id", "month"])
        months = sorted(plan["month"].unique())

        sku_rows: List[Dict[str, Any]] = []
        line_rows: List[Dict[str, Any]] = []
        backlog: Dict[str, float] = {s: 0.0 for s in skus}

        for month in months:
            # ---- 3a. gross requirement per SKU ---------------------------
            requests: Dict[str, float] = {}
            month_ctx: Dict[str, Dict[str, float]] = {}
            for s in skus:
                row = plan_ix.loc[(s, month)]
                plan_units = float(row["plan_units"])
                demand_units = float(row["demand_units"])
                stock_open = sum(l[1] for l in layers[s])

                # Order-up-to level: cycle stock from the cover lever plus
                # safety stock from the service lever. Deliberately NOT
                # "cover + this month's demand" — that top-up rule makes the
                # line effectively unstockable and pins service at 100%,
                # which would leave the lost-sales output dead.
                # O-14. Safety stock absorbs VOLATILITY, not BIAS, so it is
                # sized off demand — not off the bias-carrying plan. Step 5a
                # decomposes irreducible volatility, chronic bias and random
                # error precisely so they can be treated separately; sizing the
                # buffer off the plan re-mixes what the estimator just took
                # apart. It also carries a perverse implication: a planner who
                # over-forecasts by 20% would automatically receive more service
                # protection than one who forecasts accurately, rewarding the
                # behaviour this tool exists to expose.
                #
                # Cycle stock still scales with the plan. That is correct —
                # cover is a policy applied to the planning signal.
                safety_units = z * float(cv[s]) * demand_units
                target_stock = cover_months * plan_units + safety_units
                # Backlog is capacity-cut volume deferred from a prior month.
                # It is CONSUMED by the request that carries it — without this
                # reset it compounds forever and production diverges from
                # demand. Latent since Step 6: L3 never breaches capacity, so
                # `cut` was always zero and the bug never fired (D-058).
                need = max(0.0, target_stock - stock_open) + backlog[s]
                backlog[s] = 0.0

                if need > 0:
                    batch = max(need, float(moq[s]), min_run_units)
                else:
                    batch = 0.0
                requests[s] = batch
                month_ctx[s] = {
                    "plan_units": plan_units,
                    "demand_units": demand_units,
                    "detected_bias_pct": float(row["detected_bias_pct"]),
                    "bias_correction_applied": float(row["bias_correction_applied"]),
                    "stock_open_units": stock_open,
                    "safety_stock_units": safety_units,
                    "target_stock_units": target_stock,
                }

            # ---- 3b. capacity, changeovers, overtime ---------------------
            active = [s for s in skus if requests[s] > 0]
            changeover_count = len(active)          # every run needs a setup
            total_changeover_hours = changeover_count * changeover_hours
            run_hours = {s: requests[s] / units_per_hour for s in skus}
            total_run_hours = sum(run_hours.values())
            workload = total_run_hours + total_changeover_hours

            overtime_hours = 0.0
            if workload > hours_available:
                excess = workload - hours_available
                overtime_hours = min(
                    math.ceil(excess / overtime_step) * overtime_step, max_overtime
                )
            effective = hours_available + overtime_hours

            capacity_cut_units = 0.0
            if workload > effective + 1e-9 and active:
                # Same remedy as the generator: scale every run back pro-rata.
                available_run_hours = max(0.0, effective - total_changeover_hours)
                scale = (
                    min(1.0, available_run_hours / total_run_hours)
                    if total_run_hours > 0
                    else 0.0
                )
                for s in skus:
                    cut = requests[s] * (1.0 - scale)
                    capacity_cut_units += cut
                    requests[s] *= scale
                    run_hours[s] *= scale
                    backlog[s] += cut
                total_run_hours = sum(run_hours.values())
                workload = total_run_hours + total_changeover_hours

            # ---- 3c. produce, ship, age ----------------------------------
            for s in skus:
                ctx = month_ctx[s]
                produced = requests[s] * yield_rate * adherence
                if produced > 0:
                    layers[s].append([0.0, produced])
                available = sum(l[1] for l in layers[s])

                demand_units = ctx["demand_units"]
                sigma = float(cv[s]) * demand_units
                lost_units = _expected_shortfall(available, demand_units, sigma)
                shipped = min(available, max(0.0, demand_units - lost_units))

                # The flat disposition share (assumptions.stockout_disposition)
                # is the reordering adjustment: only the LOST share is charged.
                # The substituted and backordered remainder is absorbed outside
                # the model — a documented simplification, not a claim that the
                # model tracks all volume. A full order-book mechanism was
                # specified and rejected as disproportionate (D-057).
                remaining = shipped
                for layer in layers[s]:              # FIFO
                    if remaining <= 0:
                        break
                    take = min(layer[1], remaining)
                    layer[1] -= take
                    remaining -= take

                # write-off: stock past its usable life
                life = float(shelf_life_months[s])
                cat = str(self.sku_master.loc[s, "category"])
                trigger = float(
                    self.assumptions["categories"][cat]["writeoff_trigger_pct_life_left"]
                )
                max_age = life * (1.0 - trigger)
                writeoff_units = sum(l[1] for l in layers[s] if l[0] >= max_age)
                layers[s] = [l for l in layers[s] if l[0] < max_age and l[1] > 1e-9]
                for layer in layers[s]:
                    layer[0] += 1.0

                stock_close = sum(l[1] for l in layers[s])

                # excess / slow-moving provision (assumptions.inventory_risk)
                # Cover is measured against DEMAND, never against the plan.
                # Measuring it against the plan would let
                # forecast_bias_correction change this cost through a purely
                # definitional channel — lower the plan, and identical stock
                # is reclassified as excess. That is precisely the circular
                # coupling the §9 guardrail forbids.
                excess_units = _excess_units(
                    stock_close, demand_units, self.assumptions
                )
                provision_pct = float(
                    self.assumptions.get("inventory_risk", {}).get(
                        "excess_provision_pct", 0.0
                    )
                )

                gm = float(self.sku_master.loc[s, "gross_margin_eur"])
                lost_share = float(
                    self.assumptions["stockout_disposition"][cat]["lost"]
                )
                carrying = float(self.assumptions["finance"]["carrying_rate_monthly"])
                sc = float(std_cost[s])

                sku_rows.append(
                    {
                        "line_id": line_id,
                        "sku_id": s,
                        "month": month,
                        "demand_units": demand_units,
                        "plan_units": ctx["plan_units"],
                        "detected_bias_pct": ctx["detected_bias_pct"],
                        "bias_correction_applied": ctx["bias_correction_applied"],
                        "safety_stock_units": ctx["safety_stock_units"],
                        "target_stock_units": ctx["target_stock_units"],
                        "stock_open_units": ctx["stock_open_units"],
                        "production_units": produced,
                        "run_hours": run_hours[s],
                        "shipped_units": shipped,
                        "lost_units": lost_units,
                        "backlog_units": backlog[s],
                        "writeoff_units": writeoff_units,
                        "excess_units": excess_units,
                        "stock_close_units": stock_close,
                        "stock_value_eur": stock_close * sc,
                        "lost_sales_eur": lost_units * lost_share * gm,
                        "excess_obsolescence_eur": (
                            writeoff_units * sc + excess_units * provision_pct * sc
                        ),
                        "working_capital_cost_eur": stock_close * sc * carrying,
                    }
                )

            conversion_cost = (
                float(econ["fixed_absorption_eur_line_month"])
                + total_run_hours * float(cfg["labour_rate_eur_hr"])
                + changeover_count * float(cfg["changeover_cash_eur"])
                + overtime_hours
                * float(cfg["labour_rate_eur_hr"])
                * float(econ["overtime_premium_pct"])
            )
            line_rows.append(
                {
                    "line_id": line_id,
                    "month": month,
                    "run_hours": total_run_hours,
                    "changeover_count": changeover_count,
                    "changeover_hours": total_changeover_hours,
                    "workload_hours": workload,
                    "hours_available": hours_available,
                    "overtime_hours": overtime_hours,
                    "capacity_shortfall_units": capacity_cut_units,
                    "utilisation": workload / hours_available if hours_available else np.nan,
                    "conversion_cost_eur": conversion_cost,
                }
            )

        return pd.DataFrame(sku_rows), pd.DataFrame(line_rows)

    # -- 4. scenario / sweep --------------------------------------------

    def run_scenario(self, line_id: str, levers: LeverSettings) -> ScenarioResult:
        sku_month, line_month = self.simulate(line_id, levers)

        demand_total = float(sku_month["demand_units"].sum())
        lost_total = float(sku_month["lost_units"].sum())
        totals = {
            "lost_sales_eur": float(sku_month["lost_sales_eur"].sum()),
            "excess_obsolescence_eur": float(sku_month["excess_obsolescence_eur"].sum()),
            "conversion_cost_eur": float(line_month["conversion_cost_eur"].sum()),
            "working_capital_cost_eur": float(
                sku_month["working_capital_cost_eur"].sum()
            ),
        }
        totals["total_economic_cost_eur"] = float(sum(totals.values()))
        totals["service_achieved"] = (
            1.0 - lost_total / demand_total if demand_total > 0 else np.nan
        )
        totals["utilisation_mean"] = float(line_month["utilisation"].mean())
        totals["production_units"] = float(sku_month["production_units"].sum())
        totals["mean_stock_units"] = float(
            sku_month.groupby("month")["stock_close_units"].sum().mean()
        )
        return ScenarioResult(
            line_id=line_id,
            levers=levers,
            sku_month=sku_month,
            line_month=line_month,
            totals=totals,
            assumption_fingerprint=self.assumption_fingerprint,
        )

    def sweep(
        self,
        line_id: str,
        lever_name: str,
        values: Sequence[float],
        base_levers: LeverSettings,
    ) -> pd.DataFrame:
        """Move one lever, hold the other three. The MVD gate table."""
        if lever_name not in base_levers.as_dict():
            raise EngineViolation(f"Unknown lever {lever_name!r}")
        rows = []
        for v in values:
            res = self.run_scenario(
                line_id, base_levers.replace(**{lever_name: float(v)})
            )
            row = {"lever_name": lever_name, "lever_value": float(v)}
            row.update(res.totals)
            row["assumption_fingerprint"] = res.assumption_fingerprint
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service_z(service_target: float) -> float:
    """Inverse standard normal CDF. Abramowitz & Stegun 26.2.23.

    Implemented locally so src/ depends on pandas/numpy only — scipy is not a
    Streamlit Cloud dependency this project needs to carry for one function.
    """
    p = min(max(float(service_target), 1e-6), 1 - 1e-6)
    if p < 0.5:
        return -_service_z(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3
    return t - num / den


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_shortfall(available: float, demand: float, sigma: float) -> float:
    """E[max(0, D - available)] for D ~ N(demand, sigma).

    The standard normal loss function. This is how the analytical route (D-041)
    turns Step 5a's `irreducible_volatility_cv` into expected lost units
    without simulating a single random future.
    """
    if sigma <= 0:
        return max(0.0, demand - available)
    k = (available - demand) / sigma
    loss = _norm_pdf(k) - k * (1.0 - _norm_cdf(k))
    return max(0.0, sigma * loss)


def _excess_units(
    stock_close: float, demand_units: float, assumptions: Dict[str, Any]
) -> float:
    """Stock beyond the slow-moving threshold, in units.

    Architecture §2 names this output "excess / slow-moving risk", which is
    wider than shelf-life obsolescence. On a long-life category the write-off
    trigger never fires inside a 12-month horizon, so without this term the
    output would be structurally dead rather than merely small. Threshold and
    provision rate are config, not literals — see assumptions.inventory_risk.
    """
    cfg = assumptions.get("inventory_risk", {})
    threshold = cfg.get("excess_cover_months_threshold")
    if threshold is None or demand_units <= 0:
        return 0.0
    allowed = float(threshold) * demand_units
    return max(0.0, stock_close - allowed)
