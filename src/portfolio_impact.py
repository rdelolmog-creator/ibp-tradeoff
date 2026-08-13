"""Step 5b — scoring and portfolio-impact reporting.

Does no estimation. Reports on results Step 5a's blind estimator already
produced, joined against business context (category, plant, line, price)
that estimator never saw by design.

score_against_ground_truth() is validation-only and must never be called
from, or exposed to, the blind chat that wrote demand_characterization.py —
ground_truth.csv does not exist outside this Project's own environment.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def score_against_ground_truth(
    demand_characteristics: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Join Step 5a's estimates to the known-only-here ground truth.

    Returns one row per matched SKU with encoded and recovered bias side
    by side. Category-level and correlation summaries are left to the
    caller (they're presentation, not this function's job per E-4).
    """
    truth = ground_truth.set_index("surrogate_sku_id")
    joined = demand_characteristics.join(
        truth[["category", "category_chronic_bias", "sku_idiosyncratic_bias",
               "total_base_chronic_bias"]],
        how="inner",
    )
    if len(joined) < len(demand_characteristics):
        missing = set(demand_characteristics.index) - set(joined.index)
        raise ValueError(
            f"{len(missing)} SKUs in demand_characteristics did not match "
            f"ground_truth: {sorted(missing)[:5]}..."
        )
    return joined


def concentration_report(
    demand_characteristics: pd.DataFrame,
    sku_master: pd.DataFrame,
    clean_master: pd.DataFrame,
    flagged_sku_ids: List[str],
) -> Dict[str, Any]:
    """Where do the flagged (low-confidence) SKUs concentrate, and how big
    are they in volume and revenue terms.

    flagged_sku_ids is taken as given, not re-derived — this reports on
    Step 5a's own verdicts, it does not recompute or second-guess them.
    """
    flagged_sku_ids = list(flagged_sku_ids)
    unknown = set(flagged_sku_ids) - set(sku_master["sku_id"])
    if unknown:
        raise ValueError(f"flagged_sku_ids not found in sku_master: {sorted(unknown)}")

    master = sku_master.set_index("sku_id")
    n_history_months = clean_master["month"].nunique()
    if n_history_months == 0:
        raise ValueError("clean_master has no months — cannot annualise volume.")

    annual_units = (
        clean_master.groupby("sku_id")["actual_units"].sum() / n_history_months * 12
    )
    m = master.join(annual_units.rename("annual_units"), how="left")
    m["annual_units"] = m["annual_units"].fillna(0.0)
    m["annual_revenue_eur"] = m["annual_units"] * m["price_eur"]
    m["flagged"] = m.index.isin(flagged_sku_ids)

    axis_counts: Dict[str, pd.Series] = {}
    for axis in ("source_system", "plant_code", "line_code", "category", "abc_class"):
        if axis in m.columns:
            axis_counts[axis] = m.loc[m["flagged"], axis].value_counts()

    total_skus = len(m)
    total_vol = m["annual_units"].sum()
    total_rev = m["annual_revenue_eur"].sum()
    flagged_vol = m.loc[m["flagged"], "annual_units"].sum()
    flagged_rev = m.loc[m["flagged"], "annual_revenue_eur"].sum()

    return {
        "n_flagged": len(flagged_sku_ids),
        "n_total": total_skus,
        "sku_share": len(flagged_sku_ids) / total_skus if total_skus else np.nan,
        "volume_share": flagged_vol / total_vol if total_vol else np.nan,
        "revenue_share": flagged_rev / total_rev if total_rev else np.nan,
        "flagged_annual_units": flagged_vol,
        "flagged_annual_revenue_eur": flagged_rev,
        "total_annual_units": total_vol,
        "total_annual_revenue_eur": total_rev,
        "axis_concentration": axis_counts,
        "detail": m,
    }
