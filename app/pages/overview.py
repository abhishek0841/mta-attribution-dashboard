"""
Attribution Overview Page
Displays side-by-side model comparison with key metrics.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    MODEL_COLORS,
    CHANNEL_COLORS,
    make_grouped_bar_chart,
    results_to_csv,
)
from models.attribution import MODEL_LABELS, MODELS


def render(df: pd.DataFrame, results: dict):
    st.header("📊 Attribution Overview")
    st.markdown(
        "Compare how each attribution model distributes revenue credit across marketing channels."
    )

    if not results:
        st.warning("No attribution results available. Please load data first.")
        return

    # KPI strip
    total_revenue = df[df["converted"] == 1]["revenue"].sum()
    total_conversions = df[df["converted"] == 1]["journey_id"].nunique()
    total_cost = df["cost"].sum()
    overall_roas = total_revenue / total_cost if total_cost > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Conversions", f"{total_conversions:,}")
    c3.metric("Total Ad Spend", f"${total_cost:,.0f}")
    c4.metric("Overall ROAS", f"{overall_roas:.2f}x")

    st.divider()

    # Model selector for quick single-model bar chart
    col_left, col_right = st.columns([1, 3])
    with col_left:
        selected_model = st.selectbox(
            "Select model for details",
            options=MODELS,
            format_func=lambda m: MODEL_LABELS[m],
        )

    with col_right:
        detail_df = results[selected_model].sort_values("attributed_revenue", ascending=False)
        fig_detail = px.bar(
            detail_df,
            x="channel",
            y="attributed_revenue",
            color="channel",
            color_discrete_map=CHANNEL_COLORS,
            title=f"Attributed Revenue — {MODEL_LABELS[selected_model]}",
            labels={"attributed_revenue": "Revenue ($)", "channel": "Channel"},
            text=detail_df["attributed_revenue"].map(lambda v: f"${v:,.0f}"),
        )
        fig_detail.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=40, l=60, r=20),
        )
        fig_detail.update_traces(textposition="outside")
        st.plotly_chart(fig_detail, use_container_width=True)

    st.divider()

    # Multi-model grouped bar
    fig_grouped = make_grouped_bar_chart(
        results,
        metric="attributed_revenue",
        title="Attributed Revenue by Channel & Model",
        y_label="Revenue ($)",
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    st.divider()

    # Revenue share comparison table
    st.subheader("Revenue Share by Model (%)")
    share_rows = {}
    for model, mdf in results.items():
        total = mdf["attributed_revenue"].sum()
        share_rows[MODEL_LABELS[model]] = {
            row["channel"]: f"{row['attributed_revenue'] / total * 100:.1f}%"
            for _, row in mdf.iterrows()
        }
    share_table = pd.DataFrame(share_rows).T
    st.dataframe(share_table, use_container_width=True)

    st.divider()

    # Export
    csv = results_to_csv(results)
    st.download_button(
        label="⬇️  Download All Attribution Results (CSV)",
        data=csv,
        file_name="mta_attribution_results.csv",
        mime="text/csv",
    )
