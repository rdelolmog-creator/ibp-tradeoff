"""Step 10 — app logic, kept entirely separate from Streamlit.

Contains NO `import streamlit` anywhere in this file — that is what keeps it
testable with plain pytest exactly like every other src/ module, and it is
asserted by test_no_streamlit_import_in_app_logic.

Every function here either (a) regenerates a cheap, deterministic artefact
the same way Steps 4/6/7/8 already do, or (b) forwards to
src.engine / src.reporter and adds nothing. No new economic logic. If a
function in this file computes a cost rather than calling something that
already computes it, that is a scope violation (D-072's own discipline).
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from src.cleaner import DataCleaner
from src.engine import LeverSettings, TradeOffEngine
from src.ingest import DataIngestor
from src.portfolio_impact import get_flagged_skus
from src.reporter import (
    ReporterViolation,
    avoidable_cost_view,
    capacity_warning,
    owner_view,
    policy_brief,
    require_artefact,
)

APP_DATA_FILES = {
    "demand_characteristics": "app_data/demand_characteristics.csv",
    "censoring_diagnostics": "app_data/censoring_diagnostics.csv",
    "optimal_policy": "app_data/optimal_policy.csv",
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
# Regenerated data (D-072: cheap enough to derive on demand)
# ---------------------------------------------------------------------------


def build_clean_data(
    repo_root: str, data_root: str = "data_primary"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate clean_master and sku_master via the SAME
    DataIngestor/DataCleaner path Steps 4/6/7/8 use. No duplicated cleaning
    logic. Deterministic (seeded), safe to do at every cold start.
    """
    raw_dir = os.path.join(repo_root, data_root, "raw")
    if not os.path.isdir(raw_dir):
        subprocess.run(
            ["python", "generate_data.py"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        generated = os.path.join(repo_root, "data")
        target = os.path.join(repo_root, data_root)
        if os.path.isdir(generated) and not os.path.isdir(target):
            os.rename(generated, target)

    ing = DataIngestor(repo_root=repo_root, data_root=data_root)
    clean_master, sku_master, _ = DataCleaner(ing.schema, ing.assumptions).clean(
        ing.load()
    )
    return clean_master, sku_master


# ---------------------------------------------------------------------------
# Committed app data (D-072: Step 5a/7/8 outputs, not regenerated)
# ---------------------------------------------------------------------------


def load_app_data(repo_root: str) -> Dict[str, pd.DataFrame]:
    """Load the five committed app_data/ files.

    Raises with a clear message naming the missing file if any is absent —
    reuses src.reporter.require_artefact's contract rather than reimplementing
    it, so the failure message style is consistent across every step.
    """
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
    """Construct TradeOffEngine from committed + regenerated inputs.

    Flagged SKUs come from get_flagged_skus — the same function every
    notebook uses, not a reimplementation.
    """
    demand_characteristics = app_data["demand_characteristics"].set_index("sku_id")
    censoring_diagnostics = app_data["censoring_diagnostics"]
    flagged = get_flagged_skus(censoring_diagnostics)
    return TradeOffEngine(
        assumptions, schema, clean_master, sku_master, demand_characteristics, flagged
    )


def fingerprint_status(
    engine: TradeOffEngine, app_data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Compare the live engine's assumption fingerprint against the one
    recorded in the committed Step 7 output.

    The app WARNS on mismatch, never blocks (D-072) — a stale reference table
    is still better than no table, but the person using it must see the
    mismatch rather than trust it silently.
    """
    live = engine.assumption_fingerprint
    static_series = app_data["optimal_policy"].get("assumption_fingerprint")
    static = (
        str(static_series.iloc[0])
        if static_series is not None and len(static_series)
        else "unknown"
    )
    return {"match": live == static, "live": live, "static": static}


# ---------------------------------------------------------------------------
# Scenario view — thin forward to engine + reporter
# ---------------------------------------------------------------------------


def scenario_view(
    engine: TradeOffEngine,
    line_id: str,
    levers: LeverSettings,
    assumptions: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, float]:
    """Run one scenario and format it. Calls engine.run_scenario then
    src.reporter.avoidable_cost_view — does NOT recompute anything either of
    those already do. This function only assembles inputs and forwards.
    """
    result = engine.run_scenario(line_id, levers)
    return avoidable_cost_view(result, assumptions, schema)


def capacity_check(
    line_id: str, cover_weeks: float, portfolio_summary_df: pd.DataFrame
) -> Optional[str]:
    """Thin wrapper around src.reporter.capacity_warning."""
    row = portfolio_summary_df.set_index("line_id").loc[line_id]
    return capacity_warning(line_id, cover_weeks, row)


def owner_table(
    avoidable_view: Dict[str, float], assumptions: Dict[str, Any]
) -> pd.DataFrame:
    """Thin wrapper around src.reporter.owner_view."""
    return owner_view(avoidable_view, assumptions)


def top_policy_moves(
    optimal_policy_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    assumptions: Dict[str, Any],
    top_n: int = 10,
) -> pd.DataFrame:
    """Thin wrapper around src.reporter.policy_brief, with the default-lever
    lookup built from assumptions.abc — same construction Step 9's notebook
    uses."""
    default_levers_by_class = {
        c: {
            "inventory_cover_weeks": assumptions["abc"][c]["target_cover_weeks"],
            "service_target": assumptions["abc"][c]["service_floor"],
        }
        for c in assumptions["abc"]
    }
    return policy_brief(optimal_policy_df, sku_master, default_levers_by_class, top_n)
