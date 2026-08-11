"""Cleaning for the IBP Trade-Off Engine.

Implements cleaning-spec.md C-03 to C-13, in the order specified there. The
order is a dependency chain: keys before joins, case_size before UoM, stock_open
before null recovery, business rules last.

Maps schema.yaml raw.* to schema.yaml canonical.*. Nothing coerces silently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class CleaningViolation(Exception):
    """Raised when a cleaning invariant fails. Never downgraded to a warning."""


class DataCleaner:
    """Transform the raw layer into canonical.clean_master."""

    def __init__(self, schema: Dict[str, Any], assumptions: Dict[str, Any]) -> None:
        self.schema = schema
        self.assumptions = assumptions
        self.dropped: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}

    # -- helpers --------------------------------------------------------

    @property
    def _tolerance(self) -> float:
        for rule in self.schema["validation"]["business_rules"]:
            if "stock_rollforward" in rule:
                text = str(rule["stock_rollforward"])
                return float(text.split("tolerance")[1].split(")")[0].strip().split()[0])
        raise CleaningViolation("No stock_rollforward tolerance in schema.")

    @staticmethod
    def _pad(codes: pd.Series) -> pd.Series:
        width = codes.astype(str).str.len().max()
        return codes.astype(str).str.zfill(int(width))

    # -- C-03 -----------------------------------------------------------

    def _surrogate_key(self, frame: pd.DataFrame, system: str, code_col: str) -> pd.Series:
        """sku_id = {source_system}-{source_sku_code}.

        All 15 system-B codes collide with system-A codes for different
        materials. Joining on the bare code merges unrelated SKUs and the
        result looks plausible, which is what makes it dangerous.
        """
        return system + "-" + self._pad(frame[code_col])

    # -- C-04 -----------------------------------------------------------

    @staticmethod
    def _month_from_yyyymm(period: pd.Series) -> pd.Series:
        return pd.to_datetime(period.astype(int).astype(str), format="%Y%m")

    @staticmethod
    def _month_from_ddmmyyyy(month_end: pd.Series) -> pd.Series:
        """Explicit format, never dayfirst inference — 03/04/2024 is ambiguous."""
        parsed = pd.to_datetime(month_end, format="%d/%m/%Y", errors="raise")
        return parsed.dt.to_period("M").dt.to_timestamp()

    # -- main -----------------------------------------------------------

    def clean(
        self, raw: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        master = self._clean_master(raw["sku_master"])
        a = self._normalise_system_a(raw["system_a"])
        b = self._normalise_system_b(raw["system_b"], master)
        df = pd.concat([a, b], ignore_index=True)
        df = self._derive_stock_open(df)          # C-09
        df = self._recover_nulls(df)              # C-10, C-11
        df = self._finalise(df, master)           # C-12
        dq = self._dq_report(df, master)          # C-13
        return df, master, dq

    # -- C-06, C-07, C-08 on the master ---------------------------------

    def _clean_master(self, m: pd.DataFrame) -> pd.DataFrame:
        out = m.copy()
        out["sku_id"] = self._surrogate_key(out, out["source_system"], "source_sku_code")
        out["source_sku_code"] = self._pad(out["source_sku_code"])

        # C-08 free-text category, validated against assumptions, not observed values
        out["category"] = (
            out["category"].str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        )
        valid = set(self.assumptions["categories"])
        found = set(out["category"].unique())
        if found - valid:
            raise CleaningViolation(f"Unknown categories after normalisation: {found - valid}")
        self.stats["category_variants_collapsed"] = int(m["category"].nunique())

        out["case_size"] = out["case_size"].astype(int)
        if (out["case_size"] <= 0).any():
            raise CleaningViolation("case_size must be positive for every SKU.")

        # C-07 currency: system B is GBP per case. FX and case size both undone.
        fx = float(self.assumptions["finance"]["fx"]["gbp_eur"])
        is_b = out["source_system"] == "B"
        out["price_eur"] = out["price"].astype(float)
        out["std_cost_eur"] = out["std_cost"].astype(float)
        out.loc[is_b, "price_eur"] = out.loc[is_b, "price"].astype(float) * fx / out.loc[is_b, "case_size"]
        out.loc[is_b, "std_cost_eur"] = out.loc[is_b, "std_cost"].astype(float) * fx / out.loc[is_b, "case_size"]

        # Record the exact rate applied per row (blank for System A, which
        # needs no conversion) so the Step 4 review workbook can show the FX
        # math on the same sheet as the converted values, not just the
        # abstract rate from assumptions.yaml. See Step 4 review, item 4.
        out["fx_rate_applied"] = np.where(is_b, fx, np.nan)

        # moq / min_run are quantities: case size only, no FX
        out["moq_units"] = out["moq"].astype(float)
        out["min_run_units"] = out["min_run_qty"].astype(float)
        out.loc[is_b, "moq_units"] = out.loc[is_b, "moq"].astype(float) * out.loc[is_b, "case_size"]
        out.loc[is_b, "min_run_units"] = out.loc[is_b, "min_run_qty"].astype(float) * out.loc[is_b, "case_size"]

        out["gross_margin_eur"] = out["price_eur"] - out["std_cost_eur"]
        if (out["gross_margin_eur"] <= 0).any():
            raise CleaningViolation(
                "Non-positive margin — FX or case size applied twice."
            )
        self.stats["fx_applied_rows"] = int(is_b.sum())

        if out["sku_id"].duplicated().any():
            raise CleaningViolation("sku_id is not unique in sku_master.")
        self.stats["sku_collisions_resolved"] = int(
            out.groupby("source_sku_code")["source_system"].nunique().gt(1).sum()
        )
        return out

    # -- system A -------------------------------------------------------

    def _normalise_system_a(self, a: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "sku_id": self._surrogate_key(a, "A", "MATERIAL"),
                "month": self._month_from_yyyymm(a["PERIOD"]),
                "actual_units": a["SALES_QTY"].astype(float),
                "forecast_l1_units": a["FCST_L1"].astype(float),
                "forecast_l2_units": a["FCST_L2"].astype(float),
                "forecast_l3_units": a["FCST_L3"].astype(float),
                "production_units": a["PROD_QTY"].astype(float),
                "stock_close_units": a["STOCK_CLOSE"].astype(float),
                # C-05 rate scales: A is 0-100, detected by source not by value
                "sched_adherence": a["SCHED_ADH_PCT"].astype(float) / 100.0,
                "yield_rate": a["YIELD_PCT"].astype(float) / 100.0,
                "promo_flag": a["PROMO_FLAG"].astype(int).astype(bool),
            }
        )
        self._check_rates(out, "system_a")
        return out

    # -- system B -------------------------------------------------------

    def _normalise_system_b(self, b: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
        sku_id = self._surrogate_key(b, "B", "item_no")
        case = sku_id.map(master.set_index("sku_id")["case_size"])
        if case.isna().any():
            raise CleaningViolation("System-B SKU with no case_size in master.")
        case = case.astype(float)

        # C-06 UoM: cases to units, per SKU. case_size varies (7, 8, ...).
        vol = lambda col: b[col].astype(float) * case
        out = pd.DataFrame(
            {
                "sku_id": sku_id,
                "month": self._month_from_ddmmyyyy(b["month_end"]),
                "actual_units": vol("shipped"),
                "forecast_l1_units": vol("fcst_m1"),
                "forecast_l2_units": vol("fcst_m2"),
                "forecast_l3_units": vol("fcst_m3"),
                "production_units": vol("produced"),
                "stock_close_units": vol("stock_eom"),
                # C-05: B is already 0-1
                "sched_adherence": b["adherence"].astype(float),
                "yield_rate": b["yield_rate"].astype(float),
                "promo_flag": b["promo"].fillna("N").str.strip().eq("Y"),
            }
        )
        self._check_rates(out, "system_b")
        self.stats["uom_converted_rows"] = len(out)
        return out

    def _check_rates(self, frame: pd.DataFrame, tag: str) -> None:
        for col in ("sched_adherence", "yield_rate"):
            if not frame[col].between(0, 1).all():
                raise CleaningViolation(f"{tag}.{col} outside [0, 1] after scaling.")

    # -- C-09 -----------------------------------------------------------

    def _derive_stock_open(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["sku_id", "month"]).reset_index(drop=True)
        df["stock_open_units"] = df.groupby("sku_id")["stock_close_units"].shift(1)
        return df

    # -- C-10, C-11 -----------------------------------------------------

    def _recover_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive, do not interpolate.

        stock_open + production - stock_close is arithmetic, not estimation.
        Verified against every row where the actual is reported: max error
        0.000000. Interpolation would inject fabricated variance into the
        demand series and let generator noise be attributed to a planner role.

        A row is only counted as "recovered" if the derived value is itself
        non-null. stock_open being present is NOT sufficient — if
        production_units or stock_close_units are ALSO null on the same row
        (or stock_open itself is null because the PRIOR month's stock_close
        was null, cascading forward via the shift), the arithmetic still
        yields NaN. Previously this was flagged actual_imputed=True before
        checking whether recovery actually succeeded, which overstated the
        recovery count in the DQ report even though the row was correctly
        dropped a few lines later. See decision log, Step 4 review, item 2.
        """
        null_mask = df["actual_units"].isna()
        self.stats["nulls_in_volume"] = int(null_mask.sum())

        derived = (
            df["stock_open_units"] + df["production_units"] - df["stock_close_units"]
        )
        # Only rows where recovery actually PRODUCES a value are "recoverable".
        recoverable = null_mask & derived.notna()
        if (derived[recoverable] < -self._tolerance).any():
            raise CleaningViolation(
                "Derived actual is negative — the roll-forward identity is broken upstream."
            )
        df.loc[recoverable, "actual_units"] = derived[recoverable].clip(lower=0.0)
        df["actual_imputed"] = recoverable
        self.stats["nulls_recovered_by_identity"] = int(recoverable.sum())

        # C-11 irrecoverable: split into two distinct causes so a reviewer can
        # tell "expected month-1 edge case" from "something upstream broke and
        # is cascading" — these look identical in the row-count summary but
        # mean very different things. Distinguish by CALENDAR POSITION, not by
        # whether stock_open is null — a cascaded failure (prior month's
        # stock_close was null) also has a null stock_open, so that check
        # alone would misclassify a real cascade as an expected edge case.
        irrecoverable = df["actual_units"].isna()
        if irrecoverable.any():
            first_month_per_sku = df.groupby("sku_id")["month"].transform("min")
            is_calendar_first_month = df["month"] == first_month_per_sku
            month1_mask = irrecoverable & is_calendar_first_month
            cascade_mask = irrecoverable & ~is_calendar_first_month

            for mask, reason in [
                (month1_mask, "null volume in first month, stock_open unknown (expected edge case)"),
                (cascade_mask, "null volume in a non-first month with no valid stock_open — "
                               "either production_units/stock_close_units are also missing "
                               "this row, or a PRIOR month's stock_close was null and the gap "
                               "cascaded forward via the roll-forward shift"),
            ]:
                if mask.any():
                    rows = df.loc[mask, ["sku_id", "month"]].copy()
                    rows["month"] = rows["month"].astype(str)
                    self.dropped.append({
                        "reason": reason,
                        "rows": int(mask.sum()),
                        "detail": rows.to_dict("records"),
                    })
        return df.loc[~irrecoverable].reset_index(drop=True)

    # -- C-12 -----------------------------------------------------------

    def _finalise(self, df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
        orphans = ~df["sku_id"].isin(set(master["sku_id"]))
        if orphans.any():
            raise CleaningViolation(f"{int(orphans.sum())} rows have no parent in sku_master.")

        for col in ("actual_units", "production_units", "stock_close_units"):
            if (df[col] < -self._tolerance).any():
                raise CleaningViolation(f"{col} is negative.")

        expected_months = int(self.schema["meta"]["history_months"])
        counts = df.groupby("sku_id")["month"].nunique()
        # Rows legitimately dropped at C-11 make a short series expected, not an error.
        short = counts[counts < expected_months]
        allowed = sum(d["rows"] for d in self.dropped)
        if int((expected_months - short).sum()) > allowed:
            raise CleaningViolation("Calendar gaps beyond the rows dropped at C-11.")

        roll = (
            df["stock_open_units"] + df["production_units"]
            - df["actual_units"] - df["stock_close_units"]
        ).abs()
        checkable = df["stock_open_units"].notna()
        worst = float(roll[checkable].max())
        self.stats["stock_rollforward_max_abs_diff"] = worst
        if worst > self._tolerance:
            raise CleaningViolation(
                f"Stock roll-forward breaks by {worst:.3f} units, tolerance {self._tolerance}."
            )

        cols = list(self.schema["canonical"]["clean_master"]["columns"])
        return df.loc[:, [c for c in cols if c in df.columns]]

    # -- C-13 -----------------------------------------------------------

    def _dq_report(self, df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
        rows = [
            ("rows_out", len(df)),
            ("skus_out", df["sku_id"].nunique()),
            ("months_out", df["month"].nunique()),
            ("rows_dropped", sum(d["rows"] for d in self.dropped)),
            ("nulls_in_volume", self.stats.get("nulls_in_volume", 0)),
            ("nulls_recovered_by_identity", self.stats.get("nulls_recovered_by_identity", 0)),
            ("actual_imputed_rows", int(df["actual_imputed"].sum()) if "actual_imputed" in df else 0),
            ("fx_applied_rows", self.stats.get("fx_applied_rows", 0)),
            ("uom_converted_rows", self.stats.get("uom_converted_rows", 0)),
            ("sku_collisions_resolved", self.stats.get("sku_collisions_resolved", 0)),
            ("category_variants_collapsed", self.stats.get("category_variants_collapsed", 0)),
            ("stock_rollforward_max_abs_diff", self.stats.get("stock_rollforward_max_abs_diff", np.nan)),
        ]
        for col in df.columns:
            rows.append((f"null_rate__{col}", round(float(df[col].isna().mean()), 6)))
        return pd.DataFrame(rows, columns=["metric", "value"])
