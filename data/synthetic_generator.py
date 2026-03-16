"""
Synthetic Marketing Data Generator for Multi-Touch Attribution.

Generates realistic marketing datasets with:
- Multi-touch user journeys across 5 channels
- Conversion events with revenue
- Seasonality and channel correlation patterns
- Campaign and device segmentation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class SyntheticDataGenerator:
    """Generate synthetic multi-touch marketing attribution data."""

    CHANNELS = ["Email", "Social", "Display", "Search", "Affiliate"]
    CAMPAIGNS = ["Spring_Sale", "Summer_Promo", "Black_Friday", "New_Year", "Back_to_School"]
    DEVICES = ["Desktop", "Mobile", "Tablet"]
    GEOGRAPHIES = ["North", "South", "East", "West", "Central"]

    # Channel-level properties
    CHANNEL_COSTS = {
        "Email": 0.05,
        "Social": 0.80,
        "Display": 1.20,
        "Search": 2.50,
        "Affiliate": 3.00,
    }

    # Base conversion probability per channel position
    CHANNEL_CONVERSION_WEIGHTS = {
        "Email": 0.25,
        "Social": 0.20,
        "Display": 0.15,
        "Search": 0.30,
        "Affiliate": 0.10,
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate(
        self,
        n_users: int = 5000,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        conversion_rate: float = 0.35,
    ) -> pd.DataFrame:
        """
        Generate a synthetic marketing attribution dataset.

        Parameters
        ----------
        n_users : int
            Number of unique users to simulate.
        start_date : str
            Start date of the simulation period (YYYY-MM-DD).
        end_date : str
            End date of the simulation period (YYYY-MM-DD).
        conversion_rate : float
            Fraction of users who convert.

        Returns
        -------
        pd.DataFrame
            Touchpoint-level dataframe with columns:
            user_id, journey_id, touchpoint_order, channel, campaign,
            device, geography, timestamp, cost, converted, revenue,
            days_to_conversion.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_range_days = (end - start).days

        records = []
        n_converters = int(n_users * conversion_rate)
        converter_ids = set(range(1, n_converters + 1))

        for user_id in range(1, n_users + 1):
            will_convert = user_id in converter_ids
            journey_length = self._sample_journey_length(will_convert)

            # Random journey start time
            journey_start = start + timedelta(days=random.randint(0, date_range_days - 30))
            campaign = random.choice(self.CAMPAIGNS)
            device = self._sample_device()
            geography = random.choice(self.GEOGRAPHIES)

            channels = self._sample_channel_sequence(journey_length, will_convert)

            conversion_timestamp = None
            revenue = 0.0
            if will_convert:
                days_offset = random.randint(journey_length, journey_length + 14)
                conversion_timestamp = journey_start + timedelta(days=days_offset)
                revenue = self._sample_revenue(channels)

            for order, channel in enumerate(channels, start=1):
                ts = journey_start + timedelta(hours=order * random.randint(6, 72))
                if ts > end:
                    ts = end - timedelta(hours=1)

                cost = self.CHANNEL_COSTS[channel] * random.uniform(0.8, 1.2)
                days_to_conv = None
                if will_convert and conversion_timestamp:
                    days_to_conv = (conversion_timestamp - ts).days

                records.append(
                    {
                        "user_id": f"U{user_id:06d}",
                        "journey_id": f"J{user_id:06d}",
                        "touchpoint_order": order,
                        "channel": channel,
                        "campaign": campaign,
                        "device": device,
                        "geography": geography,
                        "timestamp": ts,
                        "cost": round(cost, 4),
                        "converted": int(will_convert),
                        "revenue": round(revenue, 2) if order == len(channels) and will_convert else 0.0,
                        "days_to_conversion": days_to_conv,
                    }
                )

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "touchpoint_order"]).reset_index(drop=True)
        return df

    def _sample_journey_length(self, will_convert: bool) -> int:
        """Sample the number of touchpoints in a user journey."""
        if will_convert:
            # Converters tend to have slightly longer journeys
            return np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.20, 0.30, 0.25, 0.12, 0.08])
        return np.random.choice([1, 2, 3, 4, 5], p=[0.30, 0.30, 0.20, 0.12, 0.08])

    def _sample_device(self) -> str:
        """Sample a device type with realistic probabilities."""
        return np.random.choice(self.DEVICES, p=[0.45, 0.45, 0.10])

    def _sample_channel_sequence(self, length: int, will_convert: bool) -> list:
        """
        Sample a sequence of channels for a journey.

        Applies realistic patterns:
        - Search is more likely near conversion
        - Display/Social at the start of funnel
        - Email often mid-funnel
        """
        if length == 1:
            weights = [0.15, 0.20, 0.10, 0.45, 0.10] if will_convert else [0.20, 0.25, 0.20, 0.25, 0.10]
            return [np.random.choice(self.CHANNELS, p=weights)]

        # Top-of-funnel weights (first touchpoint)
        top_weights = [0.20, 0.30, 0.30, 0.15, 0.05]
        # Bottom-of-funnel weights (last touchpoint before conversion)
        bottom_weights = [0.15, 0.15, 0.10, 0.50, 0.10] if will_convert else [0.20, 0.20, 0.20, 0.30, 0.10]
        # Mid-funnel weights
        mid_weights = [0.25, 0.25, 0.20, 0.20, 0.10]

        sequence = []
        for i in range(length):
            if i == 0:
                w = top_weights
            elif i == length - 1:
                w = bottom_weights
            else:
                w = mid_weights
            sequence.append(np.random.choice(self.CHANNELS, p=w))
        return sequence

    def _sample_revenue(self, channels: list) -> float:
        """Sample revenue for a converting journey based on channel mix."""
        base = np.random.lognormal(mean=4.5, sigma=0.8)
        # Search-driven journeys tend to have slightly higher revenue
        if "Search" in channels:
            base *= 1.15
        if "Affiliate" in channels:
            base *= 1.10
        return max(round(base, 2), 5.0)
