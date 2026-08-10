#!/usr/bin/env python3
"""
Synthetic raw-data generator for the IBP trade-off model.

The YAML files are the contract:
- raw output columns come only from schema.yaml -> raw.*
- behavioural/economic mechanisms read only from assumptions.yaml
- known raw defects are injected only in inject_defects()/emit()
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import math
import re
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml


class SyntheticDataGenerator:
    """Create deterministic synthetic FMCG history from the two YAML contracts."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        schema_path: str | Path | None = None,
        assumptions_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.repo_root = self.repo_root.resolve()

        self.schema_path, self.assumptions_path = self._resolve_contract_paths(
            schema_path=schema_path,
            assumptions_path=assumptions_path,
        )
        self.schema = self._read_yaml(self.schema_path)
        self.assumptions = self._read_yaml(self.assumptions_path)

        self._validate_contracts()
        self.seed = int(self.assumptions["meta"]["seed"])  # assumptions.meta.seed

        self.portfolio: pd.DataFrame | None = None
        self.demand: pd.DataFrame | None = None
        self.forecast: pd.DataFrame | None = None
        self.supply: pd.DataFrame | None = None
        self.inventory: pd.DataFrame | None = None
        self.schedule_summary: pd.DataFrame | None = None
        self.ground_truth: pd.DataFrame | None = None
        self.raw_frames: Dict[str, pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # Contract loading / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return loaded

    def _resolve_contract_paths(
        self,
        schema_path: str | Path | None,
        assumptions_path: str | Path | None,
    ) -> Tuple[Path, Path]:
        schema_candidate = Path(schema_path) if schema_path is not None else Path("config/schema.yaml")
        assumptions_candidate = (
            Path(assumptions_path)
            if assumptions_path is not None
            else Path("config/assumptions.yaml")
        )

        resolved_schema = (
            (self.repo_root / schema_candidate).resolve()
            if not schema_candidate.is_absolute()
            else schema_candidate.resolve()
        )
        resolved_assumptions = (
            (self.repo_root / assumptions_candidate).resolve()
            if not assumptions_candidate.is_absolute()
            else assumptions_candidate.resolve()
        )

        if not resolved_schema.is_file():
            raise FileNotFoundError(f"Schema contract not found: {resolved_schema}")
        if not resolved_assumptions.is_file():
            raise FileNotFoundError(f"Assumptions contract not found: {resolved_assumptions}")

        return resolved_schema, resolved_assumptions

    def _validate_contracts(self) -> None:
        schema_meta = self.schema["meta"]
        assumptions = self.assumptions

        if schema_meta["project"] != assumptions.get("meta", {}).get("project", schema_meta["project"]):
            # assumptions.yaml has no project field in the current contract; do not require one.
            raise ValueError("Project identifiers conflict between contracts.")

        expected_skus = int(schema_meta["scope"]["skus"])
        abc_total = sum(float(v["sku_count_share"]) for v in assumptions["abc"].values())
        if not np.isclose(abc_total, 1.0):
            raise ValueError("assumptions.abc.*.sku_count_share must sum to 1.")

        volume_total = sum(float(v["volume_share"]) for v in assumptions["abc"].values())
        if not np.isclose(volume_total, 1.0):
            raise ValueError("assumptions.abc.*.volume_share must sum to 1.")

        if expected_skus <= 0:
            raise ValueError("schema.meta.scope.skus must be positive.")

        forecast_lags_schema = list(schema_meta["scope"]["forecast_lags"])
        forecast_lags_assumptions = list(assumptions["forecast"]["lags_months"])
        if forecast_lags_schema != forecast_lags_assumptions:
            raise ValueError("Forecast lags disagree between schema.yaml and assumptions.yaml.")

        bias_amp = assumptions["forecast"]["chronic_bias"]["lag_amplification"]
        if len(bias_amp) != len(forecast_lags_assumptions):
            raise ValueError("forecast.chronic_bias.lag_amplification length must match lags.")

        for abc_class, values in assumptions["forecast"]["random_error_cv_by_abc"].items():
            if len(values) != len(forecast_lags_assumptions):
                raise ValueError(
                    f"forecast.random_error_cv_by_abc.{abc_class} length must match lags."
                )

        line_categories = {v["category"] for v in assumptions["lines"].values()}
        if not line_categories.issubset(assumptions["categories"]):
            raise ValueError("Every assumptions.lines.*.category must exist in assumptions.categories.")

        for line_id, line_cfg in assumptions["lines"].items():
            if line_cfg["plant"] not in assumptions["plants"]:
                raise ValueError(f"Unknown plant on line {line_id}: {line_cfg['plant']}")

        for cat, split in assumptions["stockout_disposition"].items():
            if cat == "lost_share_slider_range":
                continue
            total = sum(float(v) for v in split.values())
            if not np.isclose(total, 1.0):
                raise ValueError(f"stockout_disposition.{cat} must sum to 1.")

        # assumptions.guardrails.* — fail closed if the two non-circularity flags are not true.
        guardrails = assumptions["guardrails"]
        if not guardrails["no_direct_forecast_error_to_cost_coefficient"]:
            raise ValueError("Forecast-error-to-cost guardrail must be enabled.")
        if not guardrails["no_direct_forecast_error_to_writeoff_coefficient"]:
            raise ValueError("Forecast-error-to-writeoff guardrail must be enabled.")

        # The requested raw outputs are exactly the three tables below.
        for raw_key in ("sku_master", "system_a", "system_b"):
            if raw_key not in self.schema["raw"]:
                raise ValueError(f"schema.raw.{raw_key} is required.")

    def _rng(self, label: str) -> np.random.Generator:
        """Stable, method-local RNG so reruns/call order do not change draws."""
        label_hash = int.from_bytes(
            hashlib.sha256(label.encode("utf-8")).digest()[:8],
            byteorder="little",
            signed=False,
        )
        sequence = np.random.SeedSequence([self.seed, label_hash])
        return np.random.default_rng(sequence)

    @staticmethod
    def _round_half(values: Any) -> Any:
        return np.round(np.asarray(values, dtype=float) * 2.0) / 2.0

    @staticmethod
    def _lognormal_multiplier(
        rng: np.random.Generator, cv: float, size: int
    ) -> np.ndarray:
        """Positive mean-one multiplicative noise with the requested coefficient of variation."""
        if cv < 0:
            raise ValueError("Coefficient of variation cannot be negative.")
        if cv == 0:
            return np.ones(size, dtype=float)
        sigma2 = math.log1p(cv * cv)
        sigma = math.sqrt(sigma2)
        mu = -0.5 * sigma2
        return rng.lognormal(mean=mu, sigma=sigma, size=size)

    def _history_months(self) -> pd.DatetimeIndex:
        history_months = int(self.schema["meta"]["history_months"])
        seed_text = str(self.assumptions["meta"]["seed"])  # assumptions.meta.seed
        if len(seed_text) < 6:
            raise ValueError(
                "No calendar anchor exists in the contracts. The current contract encodes "
                "YYYYMM in the first six digits of assumptions.meta.seed; seed is too short."
            )
        year = int(seed_text[:4])
        month = int(seed_text[4:6])
        if not (1 <= month <= 12):
            raise ValueError(
                "No calendar anchor exists in the contracts. The current contract requires "
                "the first six digits of assumptions.meta.seed to form a valid YYYYMM."
            )
        end_month = pd.Timestamp(year=year, month=month, day=1)
        return pd.date_range(end=end_month, periods=history_months, freq="MS")

    # ------------------------------------------------------------------
    # S — build_portfolio
    # ------------------------------------------------------------------

    def build_portfolio(self) -> pd.DataFrame:
        rng = self._rng("build_portfolio")

        n_skus = int(self.schema["meta"]["scope"]["skus"])
        lines_cfg = self.assumptions["lines"]
        categories_cfg = self.assumptions["categories"]
        abc_cfg = self.assumptions["abc"]

        line_ids = sorted(lines_cfg)
        if not line_ids:
            raise ValueError("assumptions.lines cannot be empty.")

        # Allocate the schema.meta.scope.skus count across the contracted lines,
        # evenly because no line-level SKU-count parameter exists in assumptions.yaml.
        base, remainder = divmod(n_skus, len(line_ids))
        sku_lines: list[str] = []
        for position, line_id in enumerate(line_ids):
            sku_lines.extend([line_id] * (base + (position < remainder)))
        sku_lines = list(np.asarray(sku_lines)[rng.permutation(len(sku_lines))])

        # assumptions.abc.*.sku_count_share — exact portfolio count split.
        abc_classes = list(abc_cfg)
        raw_counts = np.array(
            [float(abc_cfg[a]["sku_count_share"]) * n_skus for a in abc_classes],
            dtype=float,
        )
        counts = np.floor(raw_counts).astype(int)
        remainder_count = n_skus - int(counts.sum())
        if remainder_count:
            fractional_order = np.argsort(-(raw_counts - counts))
            for idx in fractional_order[:remainder_count]:
                counts[idx] += 1
        sku_abc: list[str] = []
        for abc_class, count in zip(abc_classes, counts):
            sku_abc.extend([abc_class] * int(count))
        sku_abc = list(np.asarray(sku_abc)[rng.permutation(len(sku_abc))])

        rows: list[dict[str, Any]] = []
        class_counts = {a: sku_abc.count(a) for a in abc_classes}
        for ordinal, (line_id, abc_class) in enumerate(zip(sku_lines, sku_abc), start=1):
            line_cfg = lines_cfg[line_id]
            category = line_cfg["category"]
            cat_cfg = categories_cfg[category]
            plant_id = line_cfg["plant"]
            source_system = self.assumptions["plants"][plant_id]["source_system"]

            # assumptions.categories.<category>.net_price_eur_per_unit
            price_eur = float(cat_cfg["net_price_eur_per_unit"])

            # assumptions.categories.<category>.gross_margin_pct
            std_cost_eur = price_eur * (1.0 - float(cat_cfg["gross_margin_pct"]))

            # assumptions.categories.<category>.shelf_life_weeks
            shelf_life_days = int(round(float(cat_cfg["shelf_life_weeks"]) * 7.0))

            # No independent MOQ parameter exists. Derive it only from contracted
            # category mechanics: one changeover-duration equivalent of nominal output.
            # assumptions.categories.<category>.line_speed_units_hr
            # assumptions.categories.<category>.changeover_hours
            moq_units = (
                float(cat_cfg["line_speed_units_hr"]) * float(cat_cfg["changeover_hours"])
            )

            # assumptions.categories.<category>.min_run_hours
            # assumptions.categories.<category>.line_speed_units_hr
            min_run_units = (
                float(cat_cfg["line_speed_units_hr"]) * float(cat_cfg["min_run_hours"])
            )

            # schema.raw.sku_master.case_size requires varying CS pack sizes, while
            # assumptions.yaml has no case-size parameter. Derive a deterministic
            # transport/UOM-only case count from two existing category parameters;
            # it never enters economic calibration or any cost mechanism.
            # assumptions.categories.<category>.min_run_hours
            # assumptions.categories.<category>.changeover_hours
            run_hours_integer = max(1, int(round(float(cat_cfg["min_run_hours"]))))
            changeover_hours_integer = max(
                1, int(round(float(cat_cfg["changeover_hours"])))
            )
            case_size_derived = run_hours_integer + (ordinal % changeover_hours_integer)

            # assumptions.abc.<class>.volume_share and sku_count_share determine
            # relative SKU baseline volume; equal weight within each class.
            relative_volume_weight = (
                float(abc_cfg[abc_class]["volume_share"]) / class_counts[abc_class]
            )

            rows.append(
                {
                    "sku_id": f"S{ordinal:04d}",
                    "source_system": str(source_system),
                    "category": str(category),
                    "abc_class": str(abc_class),
                    "plant_id": str(plant_id),
                    "line_id": str(line_id),
                    "price_eur": price_eur,
                    "std_cost_eur": std_cost_eur,
                    "shelf_life_days": shelf_life_days,
                    "case_size_derived": int(case_size_derived),
                    "moq_units": float(moq_units),
                    "min_run_units": float(min_run_units),
                    "relative_volume_weight": float(relative_volume_weight),
                }
            )

        portfolio = pd.DataFrame(rows).sort_values("sku_id").reset_index(drop=True)

        if len(portfolio) != n_skus or portfolio["sku_id"].duplicated().any():
            raise AssertionError("Portfolio SKU count/uniqueness failed.")

        self.portfolio = portfolio
        return portfolio.copy()

    # ------------------------------------------------------------------
    # S — build_demand
    # ------------------------------------------------------------------

    def build_demand(self) -> pd.DataFrame:
        if self.portfolio is None:
            raise RuntimeError("Call build_portfolio() before build_demand().")

        rng = self._rng("build_demand")
        months = self._history_months()
        categories_cfg = self.assumptions["categories"]
        abc_cfg = self.assumptions["abc"]

        rows: list[dict[str, Any]] = []
        months_per_year = 12  # calendar identity, not a model parameter

        for sku in self.portfolio.itertuples(index=False):
            cat_cfg = categories_cfg[sku.category]

            # assumptions.categories.<category>.promo_events_per_sku_year
            promo_events = int(cat_cfg["promo_events_per_sku_year"])
            if not 0 <= promo_events <= months_per_year:
                raise ValueError(
                    f"promo_events_per_sku_year must be within a 12-month block: {sku.category}"
                )
            promo_flags = np.zeros(len(months), dtype=bool)
            for block_start in range(0, len(months), months_per_year):
                block_end = min(block_start + months_per_year, len(months))
                block_size = block_end - block_start
                events = min(promo_events, block_size)
                selected = rng.choice(block_size, size=events, replace=False)
                promo_flags[block_start + selected] = True

            for month_index, month in enumerate(months):
                # assumptions.categories.<category>.seasonality_amplitude
                amplitude = float(cat_cfg["seasonality_amplitude"])
                # assumptions.categories.<category>.seasonality_phase_months
                phase_months = float(cat_cfg.get("seasonality_phase_months", 0.0))
                seasonality = 1.0 + amplitude * math.sin(
                    2.0
                    * math.pi
                    * ((month_index - phase_months) % months_per_year)
                    / months_per_year
                )

                # assumptions.categories.<category>.annual_trend
                annual_trend = float(cat_cfg["annual_trend"])
                trend = (1.0 + annual_trend) ** (month_index / months_per_year)

                # assumptions.categories.<category>.promo_uplift_multiplier
                promo_multiplier = (
                    float(cat_cfg["promo_uplift_multiplier"])
                    if promo_flags[month_index]
                    else 1.0
                )

                raw_signal = (
                    float(sku.relative_volume_weight)
                    * seasonality
                    * trend
                    * promo_multiplier
                )
                rows.append(
                    {
                        "sku_id": sku.sku_id,
                        "month": month,
                        "source_system": sku.source_system,
                        "category": sku.category,
                        "abc_class": sku.abc_class,
                        "plant_id": sku.plant_id,
                        "line_id": sku.line_id,
                        "price_eur": float(sku.price_eur),
                        "std_cost_eur": float(sku.std_cost_eur),
                        "promo_flag": bool(promo_flags[month_index]),
                        "_raw_signal": raw_signal,
                    }
                )

        demand = pd.DataFrame(rows)

        # assumptions.calibration.target_annual_net_sales_eur
        target_annual_sales = float(
            self.assumptions["calibration"]["target_annual_net_sales_eur"]
        )
        history_years = len(months) / months_per_year
        signal_sales = float((demand["_raw_signal"] * demand["price_eur"]).sum())
        if signal_sales <= 0:
            raise ValueError("Demand signal calibration denominator must be positive.")
        baseline_scale = target_annual_sales * history_years / signal_sales
        demand["deterministic_demand_units"] = demand["_raw_signal"] * baseline_scale

        # assumptions.abc.<class>.demand_noise_cv
        # Use a positive, mean-one lognormal multiplier whose CV is exactly the
        # contracted value; this avoids creating artificial near-zero demand via clipping.
        actual_multiplier = np.empty(len(demand), dtype=float)
        for abc_class, idx in demand.groupby("abc_class").groups.items():
            cv = float(abc_cfg[abc_class]["demand_noise_cv"])
            positions = np.asarray(list(idx), dtype=int)
            actual_multiplier[positions] = self._lognormal_multiplier(
                rng, cv=cv, size=len(positions)
            )
        demand["demand_noise_multiplier"] = actual_multiplier
        demand["realised_demand_units"] = (
            demand["deterministic_demand_units"] * demand["demand_noise_multiplier"]
        )

        demand = demand.drop(columns=["_raw_signal"]).sort_values(
            ["sku_id", "month"]
        ).reset_index(drop=True)
        self.demand = demand
        return demand.copy()

    # ------------------------------------------------------------------
    # S — build_forecast
    # ------------------------------------------------------------------

    def build_forecast(self) -> pd.DataFrame:
        if self.demand is None or self.portfolio is None:
            raise RuntimeError("Call build_portfolio() and build_demand() before build_forecast().")

        rng = self._rng("build_forecast")
        fc_cfg = self.assumptions["forecast"]
        lags = list(fc_cfg["lags_months"])
        category_bias_cfg = fc_cfg["chronic_bias"]["by_category"]
        idio_sd_cfg = fc_cfg["chronic_bias"]["sku_idiosyncratic_sd_by_abc"]
        lag_amp = list(fc_cfg["chronic_bias"]["lag_amplification"])
        random_cv_cfg = fc_cfg["random_error_cv_by_abc"]

        # assumptions.forecast.chronic_bias.by_category — the contracted category
        # values are the category-level chronic draws and remain fixed for 36 months.
        category_bias_draw = {k: float(v) for k, v in category_bias_cfg.items()}

        truth_rows: list[dict[str, Any]] = []
        sku_bias: dict[str, float] = {}
        sku_idio: dict[str, float] = {}

        for sku in self.portfolio.itertuples(index=False):
            # assumptions.forecast.chronic_bias.sku_idiosyncratic_sd_by_abc
            idio = float(rng.normal(loc=0.0, scale=float(idio_sd_cfg[sku.abc_class])))
            total_base = category_bias_draw[sku.category] + idio
            sku_idio[sku.sku_id] = idio
            sku_bias[sku.sku_id] = total_base

            truth = {
                "sku_id": sku.sku_id,
                "category": sku.category,
                "abc_class": sku.abc_class,
                "category_chronic_bias": category_bias_draw[sku.category],
                "sku_idiosyncratic_bias": idio,
                "total_base_chronic_bias": total_base,
            }
            # assumptions.forecast.chronic_bias.lag_amplification
            for lag, amplification in zip(lags, lag_amp):
                truth[f"effective_chronic_bias_l{lag}"] = total_base * float(amplification)
            truth_rows.append(truth)

        forecast = self.demand.copy()

        # CRITICAL: every forecast below starts from deterministic_demand_units.
        # realised_demand_units / demand_noise_multiplier are never used in the formula.
        for lag_index, (lag, amplification) in enumerate(zip(lags, lag_amp)):
            random_multiplier = np.empty(len(forecast), dtype=float)
            for abc_class, idx in forecast.groupby("abc_class").groups.items():
                # assumptions.forecast.random_error_cv_by_abc.<class>[lag]
                cv = float(random_cv_cfg[abc_class][lag_index])
                positions = np.asarray(list(idx), dtype=int)
                random_multiplier[positions] = self._lognormal_multiplier(
                    rng, cv=cv, size=len(positions)
                )

            chronic_component = forecast["sku_id"].map(sku_bias).astype(float) * float(
                amplification
            )
            forecast[f"forecast_l{lag}_units"] = np.maximum(
                0.0,
                forecast["deterministic_demand_units"]
                * (1.0 + chronic_component)
                * random_multiplier,
            )

        self.forecast = forecast.sort_values(["sku_id", "month"]).reset_index(drop=True)
        self.ground_truth = pd.DataFrame(truth_rows).sort_values("sku_id").reset_index(drop=True)
        return self.forecast.copy()

    # ------------------------------------------------------------------
    # S — build_supply
    # ------------------------------------------------------------------

    def build_supply(self) -> pd.DataFrame:
        if self.forecast is None or self.portfolio is None:
            raise RuntimeError("Call build_forecast() before build_supply().")

        categories_cfg = self.assumptions["categories"]
        abc_cfg = self.assumptions["abc"]
        plant_econ = self.assumptions["plant_economics"]
        lines_cfg = self.assumptions["lines"]
        lags = list(self.assumptions["forecast"]["lags_months"])
        planning_lag = lags[0]

        # Calendar conversion only; policy inputs remain assumptions.abc.*.target_cover_weeks.
        weeks_per_month = (365.25 / 7.0) / 12.0

        fc = self.forecast.copy()
        portfolio_idx = self.portfolio.set_index("sku_id")

        # Initial planning stock is the contracted cover policy applied to L1 forecast.
        initial_plan_stock: dict[str, float] = {}
        first_month = fc["month"].min()
        for row in fc.loc[fc["month"] == first_month].itertuples(index=False):
            cover = float(abc_cfg[row.abc_class]["target_cover_weeks"])
            initial_plan_stock[row.sku_id] = (
                float(getattr(row, f"forecast_l{planning_lag}_units"))
                * cover
                / weeks_per_month
            )
        plan_stock = initial_plan_stock.copy()

        production_records: list[dict[str, Any]] = []
        schedule_records: list[dict[str, Any]] = []
        prior_last_sku: dict[str, str | None] = {line_id: None for line_id in lines_cfg}

        months = sorted(fc["month"].unique())
        for month in months:
            month_ts = pd.Timestamp(month)
            for line_id in sorted(lines_cfg):
                line_cfg = lines_cfg[line_id]
                category = line_cfg["category"]
                cat_cfg = categories_cfg[category]
                line_rows = fc[(fc["month"] == month_ts) & (fc["line_id"] == line_id)].copy()
                line_rows = line_rows.sort_values("sku_id")

                # assumptions.categories.<category>.line_speed_units_hr
                line_speed = float(cat_cfg["line_speed_units_hr"])
                # assumptions.categories.<category>.yield_rate
                yield_rate = float(cat_cfg["yield_rate"])
                # assumptions.plant_economics.schedule_adherence
                adherence = float(plant_econ["schedule_adherence"])
                # assumptions.categories.<category>.min_run_hours
                min_run_units = line_speed * float(cat_cfg["min_run_hours"])

                run_plan: list[dict[str, Any]] = []
                for row in line_rows.itertuples(index=False):
                    sku_id = row.sku_id
                    forecast_units = float(getattr(row, f"forecast_l{planning_lag}_units"))
                    # assumptions.abc.<class>.target_cover_weeks
                    cover_weeks = float(abc_cfg[row.abc_class]["target_cover_weeks"])
                    target_close = forecast_units * cover_weeks / weeks_per_month
                    opening_plan = float(plan_stock.get(sku_id, 0.0))
                    good_units_needed = max(
                        0.0, forecast_units + target_close - opening_plan
                    )

                    if good_units_needed > 0.0:
                        # assumptions.categories.<category>.yield_rate
                        # assumptions.plant_economics.schedule_adherence
                        gross_units_needed = good_units_needed / (yield_rate * adherence)
                        scheduled_units = max(gross_units_needed, min_run_units)
                        run_hours = scheduled_units / line_speed
                        run_plan.append(
                            {
                                "sku_id": sku_id,
                                "scheduled_units": scheduled_units,
                                "run_hours": run_hours,
                                "forecast_units": forecast_units,
                                "target_close": target_close,
                                "opening_plan": opening_plan,
                            }
                        )
                    else:
                        run_plan.append(
                            {
                                "sku_id": sku_id,
                                "scheduled_units": 0.0,
                                "run_hours": 0.0,
                                "forecast_units": forecast_units,
                                "target_close": target_close,
                                "opening_plan": opening_plan,
                            }
                        )

                active_runs = [r for r in run_plan if r["scheduled_units"] > 0.0]
                active_runs = sorted(active_runs, key=lambda r: r["sku_id"])

                changeovers = 0
                if active_runs:
                    previous = prior_last_sku[line_id]
                    for run in active_runs:
                        if previous is not None and previous != run["sku_id"]:
                            changeovers += 1
                        previous = run["sku_id"]
                    prior_last_sku[line_id] = active_runs[-1]["sku_id"]

                total_run_hours = float(sum(r["run_hours"] for r in active_runs))
                # assumptions.categories.<category>.changeover_hours
                changeover_hours_each = float(cat_cfg["changeover_hours"])
                total_changeover_hours = changeovers * changeover_hours_each
                workload_hours = total_run_hours + total_changeover_hours

                # assumptions.plant_economics.scheduled_hours_per_line_month
                regular_capacity = float(plant_econ["scheduled_hours_per_line_month"])
                # assumptions.plant_economics.overtime_step_hours
                overtime_step = float(plant_econ["overtime_step_hours"])
                # assumptions.plant_economics.max_overtime_hours_month
                max_overtime_hours = float(plant_econ["max_overtime_hours_month"])
                overtime_hours = 0.0
                if workload_hours > regular_capacity:
                    excess = workload_hours - regular_capacity
                    overtime_hours = min(
                        math.ceil(excess / overtime_step) * overtime_step,
                        max_overtime_hours,
                    )
                effective_capacity = regular_capacity + overtime_hours

                capacity_shortfall_units = 0.0
                if workload_hours > effective_capacity + 1e-9 and active_runs:
                    available_run_hours = max(
                        0.0, effective_capacity - total_changeover_hours
                    )
                    run_scale = min(
                        1.0,
                        available_run_hours / total_run_hours
                        if total_run_hours > 0.0
                        else 0.0,
                    )
                    for run in active_runs:
                        original_scheduled_units = float(run["scheduled_units"])
                        run["scheduled_units"] = original_scheduled_units * run_scale
                        run["run_hours"] = float(run["run_hours"]) * run_scale
                        capacity_shortfall_units += (
                            original_scheduled_units - float(run["scheduled_units"])
                        )
                    total_run_hours = float(sum(r["run_hours"] for r in active_runs))
                    workload_hours = total_run_hours + total_changeover_hours

                active_lookup = {r["sku_id"]: r for r in active_runs}
                for row in line_rows.itertuples(index=False):
                    run = active_lookup.get(row.sku_id)
                    if run is None:
                        production_units = 0.0
                        scheduled_units = 0.0
                        run_hours = 0.0
                    else:
                        scheduled_units = float(run["scheduled_units"])
                        run_hours = float(run["run_hours"])
                        # assumptions.categories.<category>.yield_rate
                        # assumptions.plant_economics.schedule_adherence
                        production_units = scheduled_units * yield_rate * adherence

                    production_units = float(self._round_half(production_units))
                    production_records.append(
                        {
                            "sku_id": row.sku_id,
                            "month": month_ts,
                            "production_units": production_units,
                            "scheduled_input_units": scheduled_units,
                            "run_hours": run_hours,
                            "sched_adherence": adherence,
                            "yield_rate": yield_rate,
                        }
                    )

                    # Planner stock evolves on forecast and expected realised production;
                    # it never reads realised demand.
                    opening_plan = float(plan_stock.get(row.sku_id, 0.0))
                    forecast_units = float(
                        getattr(row, f"forecast_l{planning_lag}_units")
                    )
                    plan_stock[row.sku_id] = max(
                        0.0, opening_plan + production_units - forecast_units
                    )

                # Conversion cost uses operational mechanisms only. There is no
                # forecast-error term anywhere in this calculation.
                # assumptions.plant_economics.fixed_absorption_eur_line_month
                fixed_absorption = float(plant_econ["fixed_absorption_eur_line_month"])
                # assumptions.plant_economics.labour_rate_eur_hr
                labour_rate = float(plant_econ["labour_rate_eur_hr"])
                # assumptions.categories.<category>.changeover_cash_eur
                changeover_cash = float(cat_cfg["changeover_cash_eur"])
                # assumptions.plant_economics.overtime_premium_pct
                overtime_premium = float(plant_econ["overtime_premium_pct"])

                conversion_cost = (
                    fixed_absorption
                    + total_run_hours * labour_rate
                    + changeovers * changeover_cash
                    + overtime_hours * labour_rate * overtime_premium
                )

                schedule_records.append(
                    {
                        "line_id": line_id,
                        "plant_id": line_cfg["plant"],
                        "month": month_ts,
                        "run_hours": total_run_hours,
                        "changeover_count": int(changeovers),
                        "changeover_hours": total_changeover_hours,
                        "workload_hours": workload_hours,
                        "regular_capacity_hours": regular_capacity,
                        "overtime_hours": overtime_hours,
                        "effective_capacity_hours": effective_capacity,
                        "capacity_shortfall_units": capacity_shortfall_units,
                        "conversion_cost_eur": conversion_cost,
                    }
                )

        production = pd.DataFrame(production_records)
        supply = self.forecast.merge(
            production,
            on=["sku_id", "month"],
            how="left",
            validate="one_to_one",
        )

        self.supply = supply.sort_values(["sku_id", "month"]).reset_index(drop=True)
        self.schedule_summary = pd.DataFrame(schedule_records).sort_values(
            ["line_id", "month"]
        ).reset_index(drop=True)
        return self.supply.copy()

    # ------------------------------------------------------------------
    # S — roll_inventory
    # ------------------------------------------------------------------

    def roll_inventory(self) -> pd.DataFrame:
        if self.supply is None:
            raise RuntimeError("Call build_supply() before roll_inventory().")

        abc_cfg = self.assumptions["abc"]
        lags = list(self.assumptions["forecast"]["lags_months"])
        planning_lag = lags[0]
        weeks_per_month = (365.25 / 7.0) / 12.0

        data = self.supply.copy().sort_values(["sku_id", "month"]).reset_index(drop=True)
        rolled_rows: list[dict[str, Any]] = []

        for sku_id, group in data.groupby("sku_id", sort=True):
            group = group.sort_values("month")
            first = group.iloc[0]
            # assumptions.abc.<class>.target_cover_weeks
            cover = float(abc_cfg[first["abc_class"]]["target_cover_weeks"])
            initial_open = float(
                self._round_half(
                    float(first[f"forecast_l{planning_lag}_units"]) * cover / weeks_per_month
                )
            )
            stock_open = initial_open

            for row in group.itertuples(index=False):
                production = float(self._round_half(row.production_units))
                latent_demand = float(self._round_half(row.realised_demand_units))
                available = stock_open + production
                actual_units = min(latent_demand, available)
                actual_units = float(self._round_half(actual_units))
                stock_close = float(self._round_half(available - actual_units))
                if stock_close < 0.0 and abs(stock_close) <= 0.5:
                    stock_close = 0.0
                if stock_close < 0.0:
                    raise AssertionError("Negative stock after roll-forward.")

                out = row._asdict()
                out["realised_demand_units"] = latent_demand
                out["stock_open_units"] = float(stock_open)
                out["actual_units"] = actual_units
                out["lost_units"] = max(0.0, latent_demand - actual_units)
                out["stock_close_units"] = stock_close
                rolled_rows.append(out)
                stock_open = stock_close

        inventory = pd.DataFrame(rolled_rows).sort_values(
            ["sku_id", "month"]
        ).reset_index(drop=True)

        self.inventory = inventory
        return inventory.copy()

    # ------------------------------------------------------------------
    # S — inject_defects
    # ------------------------------------------------------------------

    def inject_defects(self) -> Dict[str, pd.DataFrame]:
        if self.inventory is None or self.portfolio is None:
            raise RuntimeError("Call roll_inventory() before inject_defects().")

        rng = self._rng("inject_defects")
        raw_schema = self.schema["raw"]
        portfolio = self.portfolio.copy()
        inventory = self.inventory.copy()

        # Build local source-system codes only now, so collision is a raw-layer defect.
        code_map: dict[str, str] = {}
        for source_system, group in portfolio.groupby("source_system", sort=True):
            group = group.sort_values("sku_id")
            width = max(1, len(str(len(group))))
            for local_ordinal, sku_id in enumerate(group["sku_id"], start=1):
                code_map[sku_id] = str(local_ordinal).zfill(width)

        # Map source-system labels to the UOM values in schema.raw.sku_master.
        source_values = list(
            raw_schema["sku_master"]["columns"]["source_system"].get("values", [])
        )
        uom_values = list(raw_schema["sku_master"]["columns"]["uom"].get("values", []))
        if len(source_values) != len(uom_values):
            raise ValueError(
                "schema.raw.sku_master source-system values and UOM values must align "
                "for the two-source raw defect contract."
            )
        source_uom = dict(zip(source_values, uom_values))

        master_rows: list[dict[str, Any]] = []
        fx_gbp_eur = float(self.assumptions["finance"]["fx"]["gbp_eur"])  # assumptions.finance.fx.gbp_eur

        for idx, sku in enumerate(portfolio.itertuples(index=False)):
            uom = source_uom[sku.source_system]
            is_each = uom == raw_schema["system_a"]["columns"]["SALES_QTY"]["uom"]
            case_size = 1 if is_each else int(sku.case_size_derived)

            # Free-text category defect is applied only to raw export.
            if idx % 3 == 0:
                raw_category = sku.category.upper()
            elif idx % 3 == 1:
                raw_category = sku.category.capitalize() + " "
            else:
                raw_category = sku.category.lower()

            price_source = float(sku.price_eur)
            cost_source = float(sku.std_cost_eur)
            moq_source = float(sku.moq_units)
            min_run_source = float(sku.min_run_units)

            if not is_each:
                # assumptions.finance.fx.gbp_eur; raw B values are GBP per CS.
                price_source = price_source * case_size / fx_gbp_eur
                cost_source = cost_source * case_size / fx_gbp_eur
                moq_source = moq_source / case_size
                min_run_source = min_run_source / case_size

            master_rows.append(
                {
                    "source_system": sku.source_system,
                    "source_sku_code": code_map[sku.sku_id],
                    "description": f"{sku.category.replace('_', ' ')} {code_map[sku.sku_id]}",
                    "category": raw_category,
                    "abc_class": sku.abc_class,
                    "plant_code": sku.plant_id,
                    "line_code": sku.line_id,
                    "uom": uom,
                    "case_size": int(case_size),
                    "price": price_source,
                    "std_cost": cost_source,
                    "shelf_life_days": int(sku.shelf_life_days),
                    "moq": moq_source,
                    "min_run_qty": min_run_source,
                }
            )

        sku_master = pd.DataFrame(master_rows)

        inv = inventory.merge(
            portfolio[["sku_id", "source_system", "case_size_derived"]],
            on=["sku_id", "source_system"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_portfolio"),
        )
        inv["source_sku_code"] = inv["sku_id"].map(code_map)

        a_plants = set(raw_schema["system_a"]["covers"])
        b_plants = set(raw_schema["system_b"]["covers"])

        a = inv[inv["plant_id"].isin(a_plants)].copy()
        b = inv[inv["plant_id"].isin(b_plants)].copy()

        # Raw System A: EA, EUR, YYYYMM int, 0-100 rates.
        a_out = pd.DataFrame(
            {
                "MATERIAL": a["source_sku_code"].astype(str),
                "PLANT": a["plant_id"].astype(str),
                "PERIOD": a["month"].dt.strftime("%Y%m").astype(int),
                "SALES_QTY": a["actual_units"].astype(float),
                "FCST_L1": a["forecast_l1_units"].astype(float),
                "FCST_L2": a["forecast_l2_units"].astype(float),
                "FCST_L3": a["forecast_l3_units"].astype(float),
                "PROD_QTY": a["production_units"].astype(float),
                "STOCK_CLOSE": a["stock_close_units"].astype(float),
                "SCHED_ADH_PCT": a["sched_adherence"].astype(float) * 100.0,
                "YIELD_PCT": a["yield_rate"].astype(float) * 100.0,
                "PROMO_FLAG": a["promo_flag"].astype(int),
                "CURRENCY": raw_schema["system_a"]["columns"]["CURRENCY"]["constant"],
            }
        )

        # Raw System B: CS, GBP, DD/MM/YYYY string, 0-1 rates.
        # Use source master case sizes, not an A-shaped intermediate rename.
        b_case_size = b["sku_id"].map(
            portfolio.set_index("sku_id")["case_size_derived"]
        ).astype(float)
        b_out = pd.DataFrame(
            {
                "item_no": b["source_sku_code"].astype(str),
                "site": b["plant_id"].astype(str),
                "month_end": b["month"]
                .dt.to_period("M")
                .dt.to_timestamp("M")
                .dt.strftime("%d/%m/%Y"),
                "shipped": b["actual_units"].astype(float) / b_case_size,
                "fcst_m1": b["forecast_l1_units"].astype(float) / b_case_size,
                "fcst_m2": b["forecast_l2_units"].astype(float) / b_case_size,
                "fcst_m3": b["forecast_l3_units"].astype(float) / b_case_size,
                "produced": b["production_units"].astype(float) / b_case_size,
                "stock_eom": b["stock_close_units"].astype(float) / b_case_size,
                "adherence": b["sched_adherence"].astype(float),
                "yield_rate": b["yield_rate"].astype(float),
                "promo": np.where(b["promo_flag"].astype(bool), "Y", "N"),
                "ccy": raw_schema["system_b"]["columns"]["ccy"]["constant"],
            }
        )

        # schema.raw.known_defects.null_volume contains the only null-rate contract.
        null_defect = self._known_defect_text("null_volume")
        match = re.search(r"~?\s*([0-9]+(?:\.[0-9]+)?)\s*%", null_defect)
        if match is None:
            raise ValueError("Could not parse null-volume rate from schema.raw.known_defects.")
        null_rate = float(match.group(1)) / 100.0

        for frame, column in ((a_out, "SALES_QTY"), (b_out, "shipped")):
            null_count = int(round(len(frame) * null_rate))
            if null_count > 0:
                chosen = rng.choice(frame.index.to_numpy(), size=null_count, replace=False)
                frame.loc[chosen, column] = np.nan

        # Exact raw-column contract; no extras are allowed.
        sku_master = self._coerce_raw_frame("sku_master", sku_master)
        a_out = self._coerce_raw_frame("system_a", a_out)
        b_out = self._coerce_raw_frame("system_b", b_out)

        # Confirm the cross-system SKU-code collision defect exists.
        collision_counts = (
            sku_master.groupby("source_sku_code")["source_system"].nunique()
        )
        if not (collision_counts > 1).any():
            raise AssertionError("Known defect duplicate_sku_codes was not created.")

        self.raw_frames = {
            "sku_master": sku_master,
            "system_a": a_out,
            "system_b": b_out,
        }

        # Ground truth must be joinable to the raw layer. It keys on the internal
        # sku_id, which by design appears in no raw file; without the source keys
        # the decomposition cannot be scored against what was actually drawn.
        # The surrogate key matches schema.canonical.sku_master.sku_id:
        # {source_system}-{source_sku_code}. Written to _truth/ only — never raw.
        source_system_map = (
            self.portfolio.set_index("sku_id")["source_system"].astype(str).to_dict()
        )
        truth = self.ground_truth.copy()
        truth.insert(1, "source_system", truth["sku_id"].map(source_system_map))
        truth.insert(2, "source_sku_code", truth["sku_id"].map(code_map))
        if truth["source_sku_code"].isna().any() or truth["source_system"].isna().any():
            raise AssertionError("Ground truth could not be keyed to the raw layer.")
        truth["surrogate_sku_id"] = (
            truth["source_system"] + "-" + truth["source_sku_code"].astype(str)
        )
        self.ground_truth = truth

        return {k: v.copy() for k, v in self.raw_frames.items()}

    def _known_defect_text(self, defect_name: str) -> str:
        for item in self.schema["raw"]["known_defects"]:
            if defect_name in item:
                return str(item[defect_name])
        raise KeyError(f"Missing schema.raw.known_defects entry: {defect_name}")

    def _coerce_raw_frame(self, table_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        spec = self.schema["raw"][table_key]["columns"]
        expected_columns = list(spec)
        if set(frame.columns) != set(expected_columns):
            missing = [c for c in expected_columns if c not in frame.columns]
            unexpected = [c for c in frame.columns if c not in expected_columns]
            raise ValueError(
                f"Raw column mismatch for {table_key}; missing={missing}, unexpected={unexpected}"
            )

        frame = frame.loc[:, expected_columns].copy()
        for column, column_spec in spec.items():
            dtype = column_spec["dtype"]
            if dtype == "string":
                frame[column] = frame[column].astype("string")
            elif dtype == "int":
                if frame[column].isna().any():
                    raise ValueError(f"Non-nullable int raw column has nulls: {table_key}.{column}")
                frame[column] = frame[column].astype("int64")
            elif dtype == "float":
                frame[column] = frame[column].astype("float64")
            else:
                raise ValueError(f"Unsupported raw dtype {dtype!r} for {table_key}.{column}")
        return frame

    # ------------------------------------------------------------------
    # S — emit
    # ------------------------------------------------------------------

    def emit(self) -> Dict[str, Path]:
        if self.raw_frames is None:
            raise RuntimeError("Call inject_defects() before emit().")
        if self.ground_truth is None:
            raise RuntimeError("Ground truth is unavailable; call build_forecast() first.")

        outputs: Dict[str, Path] = {}
        for table_key in ("system_a", "system_b", "sku_master"):
            relative_file = Path(self.schema["raw"][table_key]["file"])
            output_path = (self.repo_root / relative_file).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_frames[table_key].to_csv(
                output_path,
                index=False,
                lineterminator="\n",
                na_rep="",
                float_format="%.6f",
            )
            outputs[table_key] = output_path

        # The truth path is derived from the schema-declared raw data directory.
        raw_parent = (self.repo_root / Path(self.schema["raw"]["sku_master"]["file"])).parent
        truth_path = raw_parent.parent / "_truth" / "ground_truth.csv"
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        self.ground_truth.to_csv(
            truth_path,
            index=False,
            lineterminator="\n",
            float_format="%.10f",
        )
        outputs["ground_truth"] = truth_path.resolve()
        return outputs

    # ------------------------------------------------------------------
    # Validation / orchestration
    # ------------------------------------------------------------------

    def validation_summary(self) -> Dict[str, Any]:
        if self.inventory is None or self.schedule_summary is None:
            raise RuntimeError("Call roll_inventory() before validation_summary().")

        inv = self.inventory
        schedule = self.schedule_summary
        calibration = self.assumptions["calibration"]
        lags = list(self.assumptions["forecast"]["lags_months"])

        history_years = int(self.schema["meta"]["history_months"]) / 12.0
        annual_net_sales = float((inv["actual_units"] * inv["price_eur"]).sum() / history_years)
        target_sales = float(calibration["target_annual_net_sales_eur"])

        realised_utilisation = float(
            schedule["workload_hours"].sum() / schedule["regular_capacity_hours"].sum()
        )
        target_utilisation = float(calibration["target_line_utilisation"])

        mape_rows: list[dict[str, Any]] = []
        lag1_ranges = calibration["validation_checks"].get("forecast_mape_by_abc_lag1", {})
        for abc_class in self.assumptions["abc"]:
            subset = inv[inv["abc_class"] == abc_class]
            valid = subset["realised_demand_units"] > 0
            for lag in lags:
                mape = float(
                    (
                        (
                            subset.loc[valid, f"forecast_l{lag}_units"]
                            - subset.loc[valid, "realised_demand_units"]
                        ).abs()
                        / subset.loc[valid, "realised_demand_units"]
                    ).mean()
                )
                contracted_range = lag1_ranges.get(abc_class) if lag == lags[0] else None
                passed = (
                    None
                    if contracted_range is None
                    else float(contracted_range[0]) <= mape <= float(contracted_range[1])
                )
                mape_rows.append(
                    {
                        "abc_class": abc_class,
                        "lag": lag,
                        "mape": mape,
                        "range": contracted_range,
                        "pass": passed,
                    }
                )

        roll_rule = next(
            rule["stock_rollforward"]
            for rule in self.schema["validation"]["business_rules"]
            if "stock_rollforward" in rule
        )
        tolerance_match = re.search(r"tolerance\s+([0-9]+(?:\.[0-9]+)?)", str(roll_rule))
        if tolerance_match is None:
            raise ValueError("Could not parse stock-rollforward tolerance from schema.yaml.")
        tolerance = float(tolerance_match.group(1))
        roll_diff = (
            inv["stock_open_units"]
            + inv["production_units"]
            - inv["actual_units"]
            - inv["stock_close_units"]
        ).abs()
        max_roll_diff = float(roll_diff.max())
        roll_pass = max_roll_diff <= tolerance

        # assumptions.calibration.validation_checks.conversion_cost_pct_of_cogs
        cogs = float((inv["actual_units"] * inv["std_cost_eur"]).sum())
        conversion_cost = float(schedule["conversion_cost_eur"].sum())
        conversion_pct_cogs = conversion_cost / cogs if cogs > 0 else float("nan")
        conversion_range = calibration["validation_checks"].get(
            "conversion_cost_pct_of_cogs"
        )
        conversion_pass = (
            float(conversion_range[0]) <= conversion_pct_cogs <= float(conversion_range[1])
            if conversion_range is not None
            else None
        )

        stockout_split_pass = all(
            np.isclose(sum(float(v) for v in split.values()), 1.0)
            for cat, split in self.assumptions["stockout_disposition"].items()
            if cat != "lost_share_slider_range"
        )

        summary = {
            "annual_net_sales_eur": annual_net_sales,
            "target_annual_net_sales_eur": target_sales,
            "realised_line_utilisation": realised_utilisation,
            "target_line_utilisation": target_utilisation,
            "mape": mape_rows,
            "stock_rollforward_max_abs_diff": max_roll_diff,
            "stock_rollforward_tolerance": tolerance,
            "stock_rollforward_pass": roll_pass,
            "conversion_cost_pct_of_cogs": conversion_pct_cogs,
            "conversion_cost_range": conversion_range,
            "conversion_cost_pass": conversion_pass,
            "stockout_split_sums_to_one": stockout_split_pass,
        }

        print("\nVALIDATION SUMMARY")
        print("-" * 72)
        print(
            f"Annual net sales: EUR {annual_net_sales:,.0f} "
            f"vs target EUR {target_sales:,.0f}"
        )
        print(
            f"Realised line utilisation: {realised_utilisation:.3f} "
            f"vs target {target_utilisation:.3f}"
        )
        for line_id, grp in schedule.groupby("line_id", sort=True):
            util = float(grp["workload_hours"].sum() / grp["regular_capacity_hours"].sum())
            print(f"  {line_id}: utilisation {util:.3f}")
        if realised_utilisation < target_utilisation - 0.05:
            print(
                "  Remediation: utilisation is too low; reduce "
                "categories.<category>.line_speed_units_hr or raise "
                "calibration.target_annual_net_sales_eur. The script will not self-correct."
            )
        elif realised_utilisation > target_utilisation + 0.05:
            print(
                "  Remediation: utilisation is too high; raise "
                "categories.<category>.line_speed_units_hr or reduce "
                "calibration.target_annual_net_sales_eur. The script will not self-correct."
            )

        print("Emergent MAPE by ABC and lag (against latent realised demand):")
        for row in mape_rows:
            if row["range"] is None:
                range_text = "no YAML validation range"
                pass_text = "N/A"
            else:
                lo, hi = row["range"]
                range_text = f"range [{float(lo):.3f}, {float(hi):.3f}]"
                pass_text = "PASS" if row["pass"] else "FAIL"
            print(
                f"  ABC {row['abc_class']} L{row['lag']}: "
                f"{row['mape']:.3f} | {range_text} | {pass_text}"
            )

        print(
            f"Stock roll-forward max abs diff: {max_roll_diff:.6f} "
            f"(tolerance {tolerance:.3f}) | {'PASS' if roll_pass else 'FAIL'}"
        )
        if conversion_range is not None:
            print(
                f"Conversion cost / COGS: {conversion_pct_cogs:.3f} "
                f"vs [{float(conversion_range[0]):.3f}, {float(conversion_range[1]):.3f}] "
                f"| {'PASS' if conversion_pass else 'FAIL'}"
            )
        print(
            "Stockout disposition sums to one: "
            + ("PASS" if stockout_split_pass else "FAIL")
        )
        return summary

    def run(self) -> Dict[str, Path]:
        self.build_portfolio()
        self.build_demand()
        self.build_forecast()
        self.build_supply()
        self.roll_inventory()
        self.validation_summary()
        self.inject_defects()
        outputs = self.emit()
        print("Wrote:")
        for key, path in outputs.items():
            print(f"  {key}: {path}")
        return outputs


if __name__ == "__main__":
    SyntheticDataGenerator().run()
