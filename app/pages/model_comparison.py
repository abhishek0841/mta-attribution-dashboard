"""
Model Comparison Page
Side-by-side heatmaps and difference analysis between attribution models.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from app.utils import MODEL_COLORS, make_heatmap, results_to_csv
from models.attribution import MODEL_LABELS, MODELS


def render(df: pd.DataFrame, results: dict):
    st.header("📈 Model Comparison")
    st.markdown(
        "Understand how attribution model choice affects channel credit allocation "
        "and downstream budget decisions."
    )

    if not results:
        st.warning("No attribution results available.")
        return

    # Revenue heatmap
    st.subheader("Attributed Revenue (Models × Channels)")
    fig_rev = make_heatmap(results, "attributed_revenue", "Attributed Revenue ($)")
    st.plotly_chart(fig_rev, use_container_width=True)

    # ROAS heatmap
    st.subheader("ROAS (Models × Channels)")
    fig_roas = make_heatmap(results, "roas", "Return on Ad Spend (ROAS)")
    st.plotly_chart(fig_roas, use_container_width=True)

    st.divider()

    # Revenue share heatmap (%)
    st.subheader("Revenue Share % (Models × Channels)")
    share_results = {}
    for model, mdf in results.items():
        share_df = mdf.copy()
        total = share_df["attributed_revenue"].sum()
        share_df["attributed_revenue"] = (share_df["attributed_revenue"] / total * 100).round(1)
        share_results[model] = share_df

    fig_share = make_heatmap(share_results, "attributed_revenue", "Revenue Share (%)")
    st.plotly_chart(fig_share, use_container_width=True)

    st.divider()

    # Pairwise model delta
    st.subheader("Pairwise Attribution Difference")
    st.markdown(
        "Select two models to compare attributed revenue differences per channel."
    )

    col1, col2 = st.columns(2)
    with col1:
        model_a = st.selectbox(
            "Model A",
            options=MODELS,
            index=0,
            format_func=lambda m: MODEL_LABELS[m],
            key="model_a",
        )
    with col2:
        model_b = st.selectbox(
            "Model B",
            options=MODELS,
            index=2,
            format_func=lambda m: MODEL_LABELS[m],
            key="model_b",
        )

    if model_a != model_b:
        df_a = results[model_a].set_index("channel")["attributed_revenue"]
        df_b = results[model_b].set_index("channel")["attributed_revenue"]
        diff = (df_a - df_b).reset_index()
        diff.columns = ["channel", "delta"]
        diff["direction"] = diff["delta"].apply(lambda v: "A > B" if v >= 0 else "B > A")

        fig_diff = px.bar(
            diff,
            x="channel",
            y="delta",
            color="direction",
            color_discrete_map={"A > B": "#4C72B0", "B > A": "#C44E52"},
            title=f"Revenue Difference: {MODEL_LABELS[model_a]} vs {MODEL_LABELS[model_b]}",
            labels={"delta": "Revenue Difference ($)", "channel": "Channel"},
            text=diff["delta"].map(lambda v: f"${v:+,.0f}"),
        )
        fig_diff.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_diff.update_traces(textposition="outside")
        st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("Select two different models to see the comparison.")

    st.divider()

    # Line chart: revenue per channel across models
    st.subheader("Revenue per Channel Across All Models")
    channels = sorted(
        set(ch for df_ in results.values() for ch in df_["channel"].tolist())
    )
    selected_channels = st.multiselect(
        "Filter channels", options=channels, default=channels
    )

    line_data = []
    for model, mdf in results.items():
        for _, row in mdf.iterrows():
            if row["channel"] in selected_channels:
                line_data.append(
                    {
                        "Model": MODEL_LABELS[model],
                        "Channel": row["channel"],
                        "Attributed Revenue ($)": row["attributed_revenue"],
                    }
                )
    line_df = pd.DataFrame(line_data)
    if not line_df.empty:
        fig_line = px.line(
            line_df,
            x="Model",
            y="Attributed Revenue ($)",
            color="Channel",
            color_discrete_map={
                ch: col for ch, col in {
                    "Email": "#4C72B0",
                    "Social": "#DD8452",
                    "Display": "#55A868",
                    "Search": "#C44E52",
                    "Affiliate": "#8172B2",
                }.items()
            },
            markers=True,
            title="Attributed Revenue Trajectory Across Models",
        )
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()
    st.download_button(
        "⬇️  Download All Model Results (CSV)",
        data=results_to_csv(results),
        file_name="all_model_attribution.csv",
        mime="text/csv",
    )
