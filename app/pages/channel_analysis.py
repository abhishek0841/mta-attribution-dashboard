"""
Channel Performance Analysis Page
Shows ROI, ROAS, cost per conversion and contribution by channel.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import CHANNEL_COLORS, make_grouped_bar_chart, df_to_csv
from models.attribution import MODEL_LABELS, MODELS


def render(df: pd.DataFrame, results: dict):
    st.header("💰 Channel Performance")
    st.markdown(
        "Analyse return on investment, cost efficiency, and channel contribution "
        "for each attribution model."
    )

    if not results:
        st.warning("No attribution results available.")
        return

    # Model selector
    selected_model = st.selectbox(
        "Attribution Model",
        options=MODELS,
        format_func=lambda m: MODEL_LABELS[m],
        key="channel_model",
    )
    mdf = results[selected_model].sort_values("attributed_revenue", ascending=False)

    # KPI cards
    cols = st.columns(len(mdf))
    for col, (_, row) in zip(cols, mdf.iterrows()):
        col.metric(
            label=row["channel"],
            value=f"${row['attributed_revenue']:,.0f}",
            delta=f"ROAS {row['roas']:.1f}x",
        )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["ROI & ROAS", "Cost Analysis", "Conversions", "Raw Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_roi = px.bar(
                mdf,
                x="channel",
                y="roi",
                color="channel",
                color_discrete_map=CHANNEL_COLORS,
                title="Return on Investment (ROI) by Channel",
                labels={"roi": "ROI", "channel": "Channel"},
                text=mdf["roi"].map(lambda v: f"{v:.2f}"),
            )
            fig_roi.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_roi.update_traces(textposition="outside")
            st.plotly_chart(fig_roi, use_container_width=True)

        with c2:
            fig_roas = px.bar(
                mdf,
                x="channel",
                y="roas",
                color="channel",
                color_discrete_map=CHANNEL_COLORS,
                title="Return on Ad Spend (ROAS) by Channel",
                labels={"roas": "ROAS", "channel": "Channel"},
                text=mdf["roas"].map(lambda v: f"{v:.2f}x"),
            )
            fig_roas.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_roas.update_traces(textposition="outside")
            st.plotly_chart(fig_roas, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_cost = px.bar(
                mdf.sort_values("total_cost", ascending=False),
                x="channel",
                y="total_cost",
                color="channel",
                color_discrete_map=CHANNEL_COLORS,
                title="Total Ad Spend by Channel",
                labels={"total_cost": "Total Cost ($)", "channel": "Channel"},
                text=mdf.sort_values("total_cost", ascending=False)["total_cost"].map(
                    lambda v: f"${v:,.0f}"
                ),
            )
            fig_cost.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig_cost.update_traces(textposition="outside")
            st.plotly_chart(fig_cost, use_container_width=True)

        with c2:
            fig_cpc = px.bar(
                mdf.sort_values("cost_per_conversion", ascending=False),
                x="channel",
                y="cost_per_conversion",
                color="channel",
                color_discrete_map=CHANNEL_COLORS,
                title="Cost per Conversion by Channel",
                labels={"cost_per_conversion": "Cost / Conv ($)", "channel": "Channel"},
            )
            fig_cpc.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cpc, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig_conv = px.bar(
                mdf,
                x="channel",
                y="attributed_conversions",
                color="channel",
                color_discrete_map=CHANNEL_COLORS,
                title="Attributed Conversions by Channel",
                labels={"attributed_conversions": "Conversions", "channel": "Channel"},
                text=mdf["attributed_conversions"].map(lambda v: f"{v:,.0f}"),
            )
            fig_conv.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig_conv.update_traces(textposition="outside")
            st.plotly_chart(fig_conv, use_container_width=True)

        with c2:
            total_rev = mdf["attributed_revenue"].sum()
            fig_pie = px.pie(
                mdf,
                names="channel",
                values="attributed_revenue",
                title="Revenue Contribution Share",
                color="channel",
                color_discrete_map=CHANNEL_COLORS,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab4:
        st.dataframe(
            mdf.rename(
                columns={
                    "attributed_revenue": "Attributed Revenue ($)",
                    "attributed_conversions": "Conversions",
                    "total_cost": "Total Cost ($)",
                    "roi": "ROI",
                    "roas": "ROAS",
                    "cost_per_conversion": "Cost/Conv ($)",
                }
            ),
            use_container_width=True,
        )
        st.download_button(
            "⬇️  Download Channel Data (CSV)",
            data=df_to_csv(mdf),
            file_name=f"channel_performance_{selected_model}.csv",
            mime="text/csv",
        )

    st.divider()

    # Cross-model ROAS comparison
    st.subheader("ROAS Comparison Across All Models")
    fig_multi_roas = make_grouped_bar_chart(
        results, "roas", "ROAS by Channel & Model", "ROAS"
    )
    st.plotly_chart(fig_multi_roas, use_container_width=True)
