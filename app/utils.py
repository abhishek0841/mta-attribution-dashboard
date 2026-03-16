"""Utility functions for the MTA Attribution Dashboard."""

import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data.synthetic_generator import SyntheticDataGenerator
from models.attribution import AttributionEngine, MODEL_LABELS, MODELS


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_synthetic_data(n_users: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate and cache a synthetic marketing dataset."""
    gen = SyntheticDataGenerator(seed=seed)
    return gen.generate(n_users=n_users)


def load_csv_data(uploaded_file) -> pd.DataFrame:
    """Load and validate a user-uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    required_cols = {
        "user_id", "journey_id", "touchpoint_order",
        "channel", "converted", "revenue", "cost",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df.get("timestamp", pd.Timestamp.now()), errors="coerce")
    return df


# ------------------------------------------------------------------
# Attribution calculation
# ------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def compute_attribution(
    _df: pd.DataFrame,
    decay_rate: float = 7.0,
    position_first: float = 0.40,
    position_last: float = 0.40,
) -> dict:
    """Compute all attribution models and return a dict of DataFrames."""
    engine = AttributionEngine(
        decay_rate=decay_rate,
        position_first_weight=position_first,
        position_last_weight=position_last,
    )
    return engine.run_all(_df)


# ------------------------------------------------------------------
# Filter helpers
# ------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    channels: list,
    campaigns: list,
    date_range: tuple,
) -> pd.DataFrame:
    """Apply sidebar filters to the touchpoint dataframe."""
    filtered = df.copy()
    if channels:
        filtered = filtered[filtered["channel"].isin(channels)]
    if campaigns and "campaign" in filtered.columns:
        filtered = filtered[filtered["campaign"].isin(campaigns)]
    if date_range and len(date_range) == 2 and "timestamp" in filtered.columns:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[
            (filtered["timestamp"] >= start) & (filtered["timestamp"] <= end)
        ]
    return filtered


# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------

CHANNEL_COLORS = {
    "Email": "#4C72B0",
    "Social": "#DD8452",
    "Display": "#55A868",
    "Search": "#C44E52",
    "Affiliate": "#8172B2",
}

MODEL_COLORS = {
    "first_touch": "#636EFA",
    "last_touch": "#EF553B",
    "linear": "#00CC96",
    "time_decay": "#AB63FA",
    "position_based": "#FFA15A",
}


def channel_color_sequence() -> list:
    return list(CHANNEL_COLORS.values())


# ------------------------------------------------------------------
# Chart helpers
# ------------------------------------------------------------------


def make_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_map: dict = None,
    text_col: str = None,
) -> go.Figure:
    """Create a styled bar chart."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        color=x,
        color_discrete_map=color_map or CHANNEL_COLORS,
        text=text_col,
    )
    fig.update_layout(
        showlegend=False,
        title_font_size=16,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    fig.update_traces(textposition="outside")
    return fig


def make_grouped_bar_chart(
    results: dict,
    metric: str,
    title: str,
    y_label: str,
) -> go.Figure:
    """Create a grouped bar chart comparing models across channels."""
    channels = sorted(
        set(
            ch
            for df in results.values()
            for ch in df["channel"].tolist()
        )
    )
    fig = go.Figure()
    for model, df in results.items():
        values = []
        for ch in channels:
            row = df[df["channel"] == ch]
            values.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
        fig.add_trace(
            go.Bar(
                name=MODEL_LABELS[model],
                x=channels,
                y=values,
                marker_color=MODEL_COLORS[model],
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        title_font_size=16,
        yaxis_title=y_label,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def make_heatmap(
    results: dict,
    metric: str,
    title: str,
) -> go.Figure:
    """Create a heatmap of metric values (models × channels)."""
    channels = sorted(
        set(
            ch
            for df in results.values()
            for ch in df["channel"].tolist()
        )
    )
    model_names = [MODEL_LABELS[m] for m in results]
    matrix = []
    for model in results:
        row = []
        for ch in channels:
            r = results[model][results[model]["channel"] == ch]
            row.append(float(r[metric].iloc[0]) if not r.empty else 0.0)
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=channels,
            y=model_names,
            colorscale="Blues",
            text=[[f"{v:,.0f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 11},
        )
    )
    fig.update_layout(
        title=title,
        title_font_size=16,
        margin=dict(t=60, b=60, l=120, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ------------------------------------------------------------------
# Export helper
# ------------------------------------------------------------------


def results_to_csv(results: dict) -> bytes:
    """Combine all model results into a single CSV for download."""
    frames = []
    for model, df in results.items():
        frame = df.copy()
        frame.insert(0, "model", MODEL_LABELS[model])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False).encode("utf-8")


def df_to_csv(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
