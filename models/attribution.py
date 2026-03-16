"""
Multi-Touch Attribution Engine.

Implements five attribution models:
- First-Touch
- Last-Touch
- Linear
- Time-Decay
- Position-Based (U-Shaped)

All models operate on touchpoint-level DataFrames produced by
SyntheticDataGenerator (or any compatible CSV) and return
attributed revenue per channel.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


MODELS = ["first_touch", "last_touch", "linear", "time_decay", "position_based"]
MODEL_LABELS = {
    "first_touch": "First-Touch",
    "last_touch": "Last-Touch",
    "linear": "Linear",
    "time_decay": "Time-Decay",
    "position_based": "Position-Based",
}


class AttributionEngine:
    """
    Calculate attributed revenue per channel for multiple attribution models.

    Parameters
    ----------
    decay_rate : float
        Half-life (in days) used for the Time-Decay model. A smaller value
        means more credit is concentrated on recent touches. Default: 7.
    position_first_weight : float
        Fraction of credit given to the first touchpoint in Position-Based
        model. Default: 0.40.
    position_last_weight : float
        Fraction of credit given to the last touchpoint in Position-Based
        model. Default: 0.40.
    """

    def __init__(
        self,
        decay_rate: float = 7.0,
        position_first_weight: float = 0.40,
        position_last_weight: float = 0.40,
    ):
        self.decay_rate = decay_rate
        self.position_first_weight = position_first_weight
        self.position_last_weight = position_last_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all attribution models and return a dict of results.

        Parameters
        ----------
        df : pd.DataFrame
            Touchpoint-level dataframe (as produced by SyntheticDataGenerator).

        Returns
        -------
        dict
            Keys are model names; values are DataFrames with columns
            [channel, attributed_revenue, attributed_conversions, cost,
            roi, roas, cost_per_conversion].
        """
        results = {}
        for model in MODELS:
            results[model] = self.run(df, model)
        return results

    def run(self, df: pd.DataFrame, model: str) -> pd.DataFrame:
        """
        Run a single attribution model.

        Parameters
        ----------
        df : pd.DataFrame
            Touchpoint-level dataframe.
        model : str
            One of: first_touch, last_touch, linear, time_decay, position_based.

        Returns
        -------
        pd.DataFrame
            Attribution summary per channel.
        """
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {MODELS}")

        method = getattr(self, f"_{model}")
        attributed = method(df)
        return self._build_summary(df, attributed)

    # ------------------------------------------------------------------
    # Attribution model implementations
    # ------------------------------------------------------------------

    def _first_touch(self, df: pd.DataFrame) -> pd.DataFrame:
        """100% credit to the first touchpoint of each converting journey."""
        converting = self._get_converting_journeys(df)
        first_touches = (
            converting.sort_values("touchpoint_order")
            .groupby("journey_id")
            .first()
            .reset_index()[["journey_id", "channel", "total_revenue"]]
        )
        first_touches["credit"] = 1.0
        first_touches["attributed_revenue"] = first_touches["total_revenue"] * first_touches["credit"]
        return first_touches[["journey_id", "channel", "attributed_revenue"]]

    def _last_touch(self, df: pd.DataFrame) -> pd.DataFrame:
        """100% credit to the last touchpoint of each converting journey."""
        converting = self._get_converting_journeys(df)
        last_touches = (
            converting.sort_values("touchpoint_order")
            .groupby("journey_id")
            .last()
            .reset_index()[["journey_id", "channel", "total_revenue"]]
        )
        last_touches["attributed_revenue"] = last_touches["total_revenue"]
        return last_touches[["journey_id", "channel", "attributed_revenue"]]

    def _linear(self, df: pd.DataFrame) -> pd.DataFrame:
        """Equal credit distributed across all touchpoints of each converting journey."""
        converting = self._get_converting_journeys(df)
        touch_counts = converting.groupby("journey_id")["touchpoint_order"].count().rename("n_touches")
        converting = converting.merge(touch_counts, on="journey_id")
        converting["attributed_revenue"] = converting["total_revenue"] / converting["n_touches"]
        return converting[["journey_id", "channel", "attributed_revenue"]]

    def _time_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exponential decay: touchpoints closer to conversion get more credit.

        Credit weight for touch i = exp(-decay_rate * days_to_conversion_i).
        """
        converting = self._get_converting_journeys(df).copy()

        # Use days_to_conversion if available; otherwise fall back to
        # reverse touchpoint order (0 = last touch).
        if "days_to_conversion" in converting.columns and converting["days_to_conversion"].notna().any():
            converting["decay_days"] = converting["days_to_conversion"].clip(lower=0).fillna(0)
        else:
            # Approximate: reverse order within journey
            converting["decay_days"] = (
                converting.groupby("journey_id")["touchpoint_order"].transform("max")
                - converting["touchpoint_order"]
            )

        converting["weight"] = np.exp(-converting["decay_days"] / self.decay_rate)
        weight_sum = converting.groupby("journey_id")["weight"].transform("sum")
        converting["attributed_revenue"] = (
            converting["weight"] / weight_sum * converting["total_revenue"]
        )
        return converting[["journey_id", "channel", "attributed_revenue"]]

    def _position_based(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        U-Shaped model:
        - First touch gets position_first_weight (default 40%)
        - Last touch gets position_last_weight (default 40%)
        - Remaining credit split equally among middle touches
        """
        converting = self._get_converting_journeys(df).copy()
        touch_counts = converting.groupby("journey_id")["touchpoint_order"].count().rename("n_touches")
        converting = converting.merge(touch_counts, on="journey_id")

        first_w = self.position_first_weight
        last_w = self.position_last_weight
        mid_w = 1.0 - first_w - last_w

        max_order = converting.groupby("journey_id")["touchpoint_order"].transform("max")
        min_order = converting.groupby("journey_id")["touchpoint_order"].transform("min")

        def _weight(row):
            order = row["touchpoint_order"]
            n = row["n_touches"]
            if n == 1:
                return 1.0
            if n == 2:
                if order == row["_min_order"]:
                    return first_w + mid_w / 2
                return last_w + mid_w / 2
            if order == row["_min_order"]:
                return first_w
            if order == row["_max_order"]:
                return last_w
            return mid_w / (n - 2) if n > 2 else 0.0

        converting["_min_order"] = min_order
        converting["_max_order"] = max_order
        converting["weight"] = converting.apply(_weight, axis=1)
        converting["attributed_revenue"] = converting["weight"] * converting["total_revenue"]
        return converting[["journey_id", "channel", "attributed_revenue"]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_converting_journeys(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return touchpoints only for converting journeys, enriched with
        total journey revenue.
        """
        converting_ids = df[df["converted"] == 1]["journey_id"].unique()
        converting = df[df["journey_id"].isin(converting_ids)].copy()

        # Total revenue per journey (sum of revenue column across all touchpoints)
        journey_revenue = (
            df[df["journey_id"].isin(converting_ids)]
            .groupby("journey_id")["revenue"]
            .sum()
            .rename("total_revenue")
        )
        converting = converting.merge(journey_revenue, on="journey_id")
        return converting

    def _build_summary(self, df: pd.DataFrame, attributed: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate attributed revenue per channel and join with cost data
        to compute ROI, ROAS, and cost-per-conversion.
        """
        # Channel-level attributed revenue & conversions
        rev_by_channel = (
            attributed.groupby("channel")["attributed_revenue"]
            .sum()
            .reset_index()
        )

        # Attributed conversions (count journeys where this channel appears)
        conv_by_channel = (
            attributed[attributed["attributed_revenue"] > 0]
            .groupby("channel")["journey_id"]
            .nunique()
            .reset_index()
            .rename(columns={"journey_id": "attributed_conversions"})
        )

        # Total channel cost across ALL touchpoints (not just converting)
        cost_by_channel = (
            df.groupby("channel")["cost"]
            .sum()
            .reset_index()
            .rename(columns={"cost": "total_cost"})
        )

        summary = rev_by_channel.merge(conv_by_channel, on="channel", how="left")
        summary = summary.merge(cost_by_channel, on="channel", how="left")
        summary = summary.fillna(0)

        summary["roi"] = np.where(
            summary["total_cost"] > 0,
            (summary["attributed_revenue"] - summary["total_cost"]) / summary["total_cost"],
            0.0,
        )
        summary["roas"] = np.where(
            summary["total_cost"] > 0,
            summary["attributed_revenue"] / summary["total_cost"],
            0.0,
        )
        summary["cost_per_conversion"] = np.where(
            summary["attributed_conversions"] > 0,
            summary["total_cost"] / summary["attributed_conversions"],
            0.0,
        )

        return summary.round(4)

    # ------------------------------------------------------------------
    # Journey-level helpers used by the dashboard
    # ------------------------------------------------------------------

    @staticmethod
    def get_top_journeys(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Return the most common converting journey paths.

        Parameters
        ----------
        df : pd.DataFrame
            Full touchpoint dataframe.
        top_n : int
            Number of top journeys to return.

        Returns
        -------
        pd.DataFrame
            Columns: journey_path, count, total_revenue, avg_revenue,
            avg_touchpoints.
        """
        converting = df[df["converted"] == 1].copy()
        paths = (
            converting.sort_values("touchpoint_order")
            .groupby("journey_id")
            .agg(
                path=("channel", lambda x: " → ".join(x)),
                n_touches=("touchpoint_order", "count"),
                revenue=("revenue", "sum"),
            )
            .reset_index()
        )
        journey_summary = (
            paths.groupby("path")
            .agg(
                count=("journey_id", "count"),
                total_revenue=("revenue", "sum"),
                avg_touchpoints=("n_touches", "mean"),
            )
            .reset_index()
            .rename(columns={"path": "journey_path"})
        )
        journey_summary["avg_revenue"] = journey_summary["total_revenue"] / journey_summary["count"]
        return (
            journey_summary.sort_values("count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
            .round(2)
        )

    @staticmethod
    def get_channel_sequence_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return statistics about first and last touches for converting journeys.
        """
        converting = df[df["converted"] == 1].copy()
        first_last = (
            converting.sort_values("touchpoint_order")
            .groupby("journey_id")
            .agg(first_touch=("channel", "first"), last_touch=("channel", "last"))
            .reset_index()
        )
        return (
            first_last.groupby(["first_touch", "last_touch"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
