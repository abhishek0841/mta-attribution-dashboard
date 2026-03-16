"""
Journey Analysis Page
Shows top converting paths, touchpoint sequences, and funnel analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import CHANNEL_COLORS, df_to_csv
from models.attribution import AttributionEngine


def render(df: pd.DataFrame, results: dict):
    st.header("🔄 Journey Analysis")
    st.markdown(
        "Explore the most common converting customer journeys and channel sequences."
    )

    converting = df[df["converted"] == 1].copy()
    if converting.empty:
        st.warning("No converting journeys found in the current data.")
        return

    # Journey length distribution
    journey_lengths = (
        converting.groupby("journey_id")["touchpoint_order"]
        .count()
        .value_counts()
        .sort_index()
        .reset_index()
    )
    journey_lengths.columns = ["Journey Length", "Count"]

    col1, col2 = st.columns(2)
    with col1:
        fig_len = px.bar(
            journey_lengths,
            x="Journey Length",
            y="Count",
            title="Distribution of Journey Lengths (Converting Users)",
            labels={"Journey Length": "# Touchpoints", "Count": "# Users"},
            color_discrete_sequence=["#4C72B0"],
        )
        fig_len.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_len, use_container_width=True)

    with col2:
        # Touchpoint frequency among converters
        touch_freq = (
            converting["channel"]
            .value_counts()
            .reset_index()
        )
        touch_freq.columns = ["Channel", "Appearances"]
        fig_freq = px.bar(
            touch_freq,
            x="Channel",
            y="Appearances",
            color="Channel",
            color_discrete_map=CHANNEL_COLORS,
            title="Channel Appearances in Converting Journeys",
        )
        fig_freq.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_freq, use_container_width=True)

    st.divider()

    # Top converting paths
    st.subheader("Top Converting Journey Paths")
    top_n = st.slider("Number of top journeys to display", 5, 30, 15)
    top_journeys = AttributionEngine.get_top_journeys(df, top_n=top_n)

    fig_paths = px.bar(
        top_journeys.head(top_n),
        x="count",
        y="journey_path",
        orientation="h",
        color="avg_revenue",
        color_continuous_scale="Blues",
        title="Top Converting Paths (by frequency)",
        labels={"count": "# Conversions", "journey_path": "Journey Path", "avg_revenue": "Avg Revenue ($)"},
    )
    fig_paths.update_layout(
        height=max(400, top_n * 30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=250),
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig_paths, use_container_width=True)

    st.dataframe(
        top_journeys.rename(
            columns={
                "journey_path": "Path",
                "count": "Conversions",
                "total_revenue": "Total Revenue ($)",
                "avg_revenue": "Avg Revenue ($)",
                "avg_touchpoints": "Avg Touchpoints",
            }
        ),
        use_container_width=True,
    )

    st.divider()

    # First → Last touch heatmap
    st.subheader("First-Touch → Last-Touch Flow")
    seq_stats = AttributionEngine.get_channel_sequence_stats(df)
    pivot = seq_stats.pivot(index="first_touch", columns="last_touch", values="count").fillna(0)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Blues",
            text=[[f"{int(v)}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig_heat.update_layout(
        title="Conversion Count by First → Last Touchpoint",
        xaxis_title="Last Touch",
        yaxis_title="First Touch",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=80, l=100, r=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # Channel first / last touch stats
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Channels as First Touch")
        first_stats = (
            seq_stats.groupby("first_touch")["count"].sum().reset_index()
        )
        fig_first = px.pie(
            first_stats,
            names="first_touch",
            values="count",
            color="first_touch",
            color_discrete_map=CHANNEL_COLORS,
            title="First-Touch Distribution",
        )
        st.plotly_chart(fig_first, use_container_width=True)

    with col_b:
        st.subheader("Channels as Last Touch")
        last_stats = (
            seq_stats.groupby("last_touch")["count"].sum().reset_index()
        )
        fig_last = px.pie(
            last_stats,
            names="last_touch",
            values="count",
            color="last_touch",
            color_discrete_map=CHANNEL_COLORS,
            title="Last-Touch Distribution",
        )
        st.plotly_chart(fig_last, use_container_width=True)

    st.divider()
    st.download_button(
        "⬇️  Download Top Journeys (CSV)",
        data=df_to_csv(top_journeys),
        file_name="top_converting_journeys.csv",
        mime="text/csv",
    )
