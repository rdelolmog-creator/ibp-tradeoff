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

        for cover, service in itertools.product(cover_values, service_values):
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

        default_mask = (
            surface["inventory_cover_weeks"]
            == self.base_levers.inventory_cover_weeks
        ) & (surface["service_target"] == self.base_levers.service_target)
        if default_mask.any():
            default_cost = surface[default_mask].set_index("sku_id")["total_eur"]
        else:  # default not on the grid — evaluate it explicitly
            res = self.engine.run_scenario(self.line_id, self.base_levers)
            default_cost = (
                res.sku_month.groupby("sku_id")[
                    [
                        "lost_sales_eur",
                        "excess_obsolescence_eur",
                        "working_capital_cost_eur",
                    ]
                ]
                .sum()
                .sum(axis=1)
            )

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
