"""Step 10 — app logic, kept entirely separate from Streamlit.

Contains NO `import streamlit` anywhere in this file — that is what keeps it
testable with plain pytest exactly like every other src/ module (asserted by
test_no_streamlit_import_in_app_logic).

Rewritten for D-071/D-072: the app now drives PER-CLASS levers directly
(matching the engine's real capability and the dashboard mockup agreed with
the user), and capacity warnings are computed LIVE from the actual scenario
being viewed — not looked up in a static Step 8 table. Every number traces to
engine.run_scenario() or a Step 9 reporter function; this file adds no
economic logic of its own.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.cleaner import DataCleaner
from src.engine import LeverSettings, TradeOffEngine
from src.ingest import DataIngestor
from src.portfolio_impact import get_flagged_skus
from src.reporter import (
    ReporterViolation,
    avoidable_cost_view,
    class_breakdown_view,
    line_policy_brief,
    owner_view,
    require_artefact,
)

ABC_CLASSES = ("A", "B", "C")

APP_DATA_FILES = {
    "demand_characteristics": "app_data/demand_characteristics.csv",
    "censoring_diagnostics": "app_data/censoring_diagnostics.csv",
    "line_results": "app_data/line_results.csv",
    "sku_level": "app_data/sku_level.csv",
    "portfolio_summary": "app_data/step08_portfolio_summary.csv",
    "lever_consistency": "app_data/step08_lever_consistency.csv",
}


class AppLogicViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(repo_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """assumptions.yaml + schema.yaml. Not cached — trivial cost, and caching
    config risks serving stale config after a git-triggered redeploy."""
    assumptions = yaml.safe_load(
        open(os.path.join(repo_root, "config", "assumptions.yaml"))
    )
    schema = yaml.safe_load(open(os.path.join(repo_root, "config", "schema.yaml")))
    return assumptions, schema


# ---------------------------------------------------------------------------
# Regenerated data
# ---------------------------------------------------------------------------


def build_clean_data(
    repo_root: str, data_root: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate clean_master and sku_master via the SAME
    DataIngestor/DataCleaner path every notebook uses.

    Streamlit Cloud mounts the repo READ-ONLY. generate_data.py is imported
    and run IN-PROCESS (not via subprocess, whose CalledProcessError hides
    the real traceback) and writes to a WRITABLE temp directory, never into
    repo_root — the fix for the deploy failure this project hit the first
    time this ran on Streamlit Cloud.
    """
    if data_root is None:
        data_root = os.path.join(tempfile.gettempdir(), "ibp-tradeoff-data")

    raw_dir = os.path.join(data_root, "data", "raw")
    if not os.path.isdir(raw_dir):
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from generate_data import SyntheticDataGenerator

        SyntheticDataGenerator(
            repo_root=data_root,
            schema_path=os.path.join(repo_root, "config", "schema.yaml"),
            assumptions_path=os.path.join(repo_root, "config", "assumptions.yaml"),
        ).run()

    ing = DataIngestor(
        repo_root=repo_root, data_root=os.path.join(data_root, "data")
    )
    clean_master, sku_master, _ = DataCleaner(ing.schema, ing.assumptions).clean(
        ing.load()
    )
    return clean_master, sku_master


# ---------------------------------------------------------------------------
# Committed app data
# ---------------------------------------------------------------------------


def load_app_data(repo_root: str) -> Dict[str, pd.DataFrame]:
    """Load the six committed app_data/ files, naming the missing one if any
    is absent — reuses src.reporter.require_artefact's contract."""
    out: Dict[str, pd.DataFrame] = {}
    for key, rel_path in APP_DATA_FILES.items():
        path = require_artefact(os.path.join(repo_root, rel_path))
        out[key] = pd.read_csv(path)
    return out


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------


def build_engine(
    assumptions: Dict[str, Any],
    schema: Dict[str, Any],
    clean_master: pd.DataFrame,
    sku_master: pd.DataFrame,
    app_data: Dict[str, pd.DataFrame],
) -> TradeOffEngine:
    demand_characteristics = app_data["demand_characteristics"].set_index("sku_id")
    censoring_diagnostics = app_data["censoring_diagnostics"]
    flagged = get_flagged_skus(censoring_diagnostics)
    return TradeOffEngine(
        assumptions, schema, clean_master, sku_master, demand_characteristics, flagged
    )


def fingerprint_status(
    engine: TradeOffEngine, app_data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Compare the live engine's fingerprint against the committed Step 7
    output's. WARNS on mismatch, never blocks — a stale reference table is
    still better than no table, but the mismatch must be visible."""
    live = engine.assumption_fingerprint
    static_series = app_data["line_results"].get("assumption_fingerprint")
    static = (
        str(static_series.iloc[0])
        if static_series is not None and len(static_series)
        else "unknown"
    )
    return {"match": live == static, "live": live, "static": static}


# ---------------------------------------------------------------------------
# Lever presets
# ---------------------------------------------------------------------------


def base_levers_for_line(
    engine: TradeOffEngine, assumptions: Dict[str, Any], line_id: str
) -> LeverSettings:
    """The true per-class default (D-071/D-072's 'Base') — each class at its
    OWN class's default cover/service, not class A applied to everyone."""
    skus = engine.line_skus(line_id)
    category = str(engine.sku_master.loc[skus[0], "category"])
    return LeverSettings.defaults_per_class(assumptions, category)


def optimal_levers_for_line(
    line_results: pd.DataFrame, line_id: str
) -> LeverSettings:
    """The jointly-feasible optimum for one line, from Step 7's committed
    search_all_lines() output. Only meaningful for a SINGLE line: different
    lines have independently-found optima, so 'jump to optimal' does not
    generalise cleanly to a multi-line selection — the app restricts the
    jump action to a single selected line."""
    row = line_results[line_results["line_id"] == line_id]
    if row.empty:
        raise AppLogicViolation(
            f"no committed optimal policy for line {line_id!r} — "
            f"app_data/line_results.csv may be stale or incomplete"
        )
    r = row.iloc[0]
    return LeverSettings(
        service_target={c: float(r[f"service_{c}"]) for c in ABC_CLASSES},
        inventory_cover_weeks={c: float(r[f"cover_{c}"]) for c in ABC_CLASSES},
        forecast_bias_correction=float(r["bias_correction"]),
        min_run_hours=float(r["min_run_hours"]),
    )


# ---------------------------------------------------------------------------
# Live multi-line scenario — the core of the app
# ---------------------------------------------------------------------------


def run_scenario_for_lines(
    engine: TradeOffEngine,
    line_ids: List[str],
    levers: LeverSettings,
    assumptions: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Run ONE per-class LeverSettings across every selected line and
    aggregate. This is genuinely live — every number here comes from a real
    run_scenario() call made just now, not a cached or precomputed table.

    Returns the combined avoidable-cost view (summed across lines), the
    concatenated per-SKU sku_month for every selected line, and a per-line
    capacity read (workload, shortfall) so the app can warn precisely.
    """
    if not line_ids:
        raise AppLogicViolation("at least one line must be selected")

    per_line_views: List[Dict[str, float]] = []
    sku_frames: List[pd.DataFrame] = []
    capacity_rows: List[Dict[str, Any]] = []

    for line_id in line_ids:
        result = engine.run_scenario(line_id, levers)
        view = avoidable_cost_view(result, assumptions, schema)
        per_line_views.append(view)

        sku_month = result.sku_month.copy()
        sku_month["line_id"] = line_id
        sku_month["abc_class"] = sku_month["sku_id"].map(
            engine.sku_master["abc_class"]
        )
        sku_frames.append(sku_month)

        lm = result.line_month
        capacity_rows.append(
            {
                "line_id": line_id,
                "workload_hours_max": float(lm["workload_hours"].max()),
                "hours_available": float(lm["hours_available"].iloc[0]),
                "overtime_hours_total": float(lm["overtime_hours"].sum()),
                "capacity_shortfall_total": float(
                    lm["capacity_shortfall_units"].sum()
                ),
            }
        )

    combined: Dict[str, float] = {}
    for key in per_line_views[0]:
        if key in ("line_id", "assumption_fingerprint"):
            continue
        combined[key] = sum(v[key] for v in per_line_views)
    combined["line_id"] = "+".join(line_ids) if len(line_ids) > 1 else line_ids[0]
    combined["assumption_fingerprint"] = per_line_views[0]["assumption_fingerprint"]

    return {
        "avoidable_view": combined,
        "sku_month": pd.concat(sku_frames, ignore_index=True),
        "capacity": pd.DataFrame(capacity_rows),
    }


def live_capacity_warnings(capacity_df: pd.DataFrame) -> List[str]:
    """Warnings computed from the LIVE scenario just run — not a static
    lookup. A line is flagged if this exact combination of levers produced
    any unmet production requirement on it."""
    warnings: List[str] = []
    for _, row in capacity_df.iterrows():
        if row["capacity_shortfall_total"] > 1e-6:
            warnings.append(
                f"{row['line_id']}: this combination breaches capacity — "
                f"peak workload {row['workload_hours_max']:.0f}h against "
                f"{row['hours_available']:.0f}h available "
                f"(+{row['overtime_hours_total']:.0f}h overtime used), with "
                f"{row['capacity_shortfall_total']:.0f} units short."
            )
    return warnings


# ---------------------------------------------------------------------------
# Reporter wrappers
# ---------------------------------------------------------------------------


def owner_table(
    avoidable_view: Dict[str, float], assumptions: Dict[str, Any]
) -> pd.DataFrame:
    return owner_view(avoidable_view, assumptions)


def by_class_breakdown(
    sku_month: pd.DataFrame, levers: LeverSettings
) -> pd.DataFrame:
    """Real per-class economics for the CURRENT live scenario, in the shape
    src.reporter.class_breakdown_view expects.

    cover_weeks/service_target are derived here from the levers actually
    used for this scenario — engine.sku_month does not carry them, since the
    engine reasons in target_stock_units, not the lever value itself.
    Conversion cost is not computed per SKU by the engine, so it is
    intentionally absent here — shown once per line elsewhere in the app,
    never split (D-066).

    sku_month arrives as one row per SKU PER MONTH — it must be summed to one
    row per SKU first, or class_breakdown_view's n_skus would count SKU-
    months rather than SKUs.
    """
    per_sku = (
        sku_month.groupby(["sku_id", "line_id", "abc_class"])
        .agg(
            lost_sales_eur=("lost_sales_eur", "sum"),
            excess_obsolescence_eur=("excess_obsolescence_eur", "sum"),
            working_capital_cost_eur=("working_capital_cost_eur", "sum"),
        )
        .reset_index()
    )
    per_sku["cover_weeks"] = per_sku["abc_class"].map(
        lambda c: levers.resolve("inventory_cover_weeks", c)
    )
    per_sku["service_target"] = per_sku["abc_class"].map(
        lambda c: levers.resolve("service_target", c)
    )
    per_sku["line_conversion_cost_eur"] = pd.NA
    return class_breakdown_view(per_sku)


def line_policy_table(line_results: pd.DataFrame) -> pd.DataFrame:
    return line_policy_brief(line_results)
