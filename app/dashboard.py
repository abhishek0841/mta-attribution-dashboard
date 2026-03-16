"""
Multi-Touch Attribution Dashboard
Main Streamlit application entry point.

Run with:
    streamlit run app/dashboard.py
"""

import sys
import os

# Allow imports from the project root when running via `streamlit run app/dashboard.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
import yaml

from app.utils import (
    load_synthetic_data,
    load_csv_data,
    compute_attribution,
    apply_filters,
)
from app.pages import overview, channel_analysis, journey_analysis, model_comparison
from data.synthetic_generator import SyntheticDataGenerator

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="MTA Attribution Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------------
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG = load_config()


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Data source
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source",
        options=["Synthetic Data", "Upload CSV"],
        index=0,
    )

    uploaded_file = None
    n_users = CONFIG.get("data", {}).get("n_users", 5000)
    if data_source == "Synthetic Data":
        n_users = st.sidebar.slider(
            "Number of users to simulate",
            min_value=500,
            max_value=20000,
            value=n_users,
            step=500,
        )
        seed = st.sidebar.number_input("Random seed", value=42, min_value=0, step=1)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file", type=["csv"], help="See README for required columns."
        )
        seed = 42

    st.sidebar.markdown("---")

    # Model configuration
    st.sidebar.subheader("Model Parameters")
    decay_rate = st.sidebar.slider(
        "Time-Decay half-life (days)",
        min_value=1,
        max_value=30,
        value=CONFIG.get("models", {}).get("time_decay_rate", 7),
        help="Smaller = more credit to recent touches",
    )
    position_first = st.sidebar.slider(
        "Position-Based first-touch weight",
        min_value=0.1,
        max_value=0.6,
        value=float(CONFIG.get("models", {}).get("position_first_weight", 0.40)),
        step=0.05,
    )
    position_last = st.sidebar.slider(
        "Position-Based last-touch weight",
        min_value=0.1,
        max_value=0.6,
        value=float(CONFIG.get("models", {}).get("position_last_weight", 0.40)),
        step=0.05,
    )
    if position_first + position_last > 1.0:
        st.sidebar.error("First + Last weights must not exceed 1.0")
        position_last = 1.0 - position_first

    st.sidebar.markdown("---")

    # Filters (only relevant when data is loaded)
    st.sidebar.subheader("Filters")
    channels = st.sidebar.multiselect(
        "Channels",
        options=SyntheticDataGenerator.CHANNELS,
        default=SyntheticDataGenerator.CHANNELS,
    )
    campaigns = st.sidebar.multiselect(
        "Campaigns",
        options=SyntheticDataGenerator.CAMPAIGNS,
        default=SyntheticDataGenerator.CAMPAIGNS,
    )
    min_date = df["timestamp"].min().date() if not df.empty and "timestamp" in df.columns else None
    max_date = df["timestamp"].max().date() if not df.empty and "timestamp" in df.columns else None
    if min_date and max_date:
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None

    return {
        "data_source": data_source,
        "uploaded_file": uploaded_file,
        "n_users": n_users,
        "seed": seed,
        "decay_rate": decay_rate,
        "position_first": position_first,
        "position_last": position_last,
        "channels": channels,
        "campaigns": campaigns,
        "date_range": date_range,
    }


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    st.title("📊 Multi-Touch Attribution Dashboard")
    st.markdown(
        "A production-ready attribution analytics platform for marketing data scientists. "
        "Compare **First-Touch, Last-Touch, Linear, Time-Decay, and Position-Based** models "
        "using synthetic or your own marketing data."
    )

    # Load base data (synthetic) to populate sidebar filters
    base_df = load_synthetic_data(n_users=5000, seed=42)
    sidebar = render_sidebar(base_df)

    # Resolve data
    if sidebar["data_source"] == "Synthetic Data":
        with st.spinner("Generating synthetic marketing data…"):
            df = load_synthetic_data(n_users=sidebar["n_users"], seed=int(sidebar["seed"]))
        st.sidebar.success(f"✅ Loaded {len(df):,} touchpoints")
    else:
        if sidebar["uploaded_file"] is not None:
            try:
                df = load_csv_data(sidebar["uploaded_file"])
                st.sidebar.success(f"✅ Loaded {len(df):,} touchpoints")
            except Exception as exc:
                st.sidebar.error(f"Error loading file: {exc}")
                df = base_df
        else:
            st.info("👈 Upload a CSV file or switch to synthetic data in the sidebar.")
            df = base_df

    # Apply filters
    date_range = sidebar.get("date_range")
    if date_range and not isinstance(date_range, (list, tuple)):
        date_range = [date_range, date_range]
    filtered_df = apply_filters(
        df,
        channels=sidebar["channels"],
        campaigns=sidebar["campaigns"],
        date_range=date_range,
    )

    if filtered_df.empty:
        st.warning("No data matches the current filters. Adjust the sidebar settings.")
        return

    # Compute attribution
    with st.spinner("Computing attribution models…"):
        results = compute_attribution(
            filtered_df,
            decay_rate=float(sidebar["decay_rate"]),
            position_first=float(sidebar["position_first"]),
            position_last=float(sidebar["position_last"]),
        )

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📑 Navigation",
        options=["Attribution Overview", "Channel Performance", "Journey Analysis", "Model Comparison"],
    )

    if page == "Attribution Overview":
        overview.render(filtered_df, results)
    elif page == "Channel Performance":
        channel_analysis.render(filtered_df, results)
    elif page == "Journey Analysis":
        journey_analysis.render(filtered_df, results)
    elif page == "Model Comparison":
        model_comparison.render(filtered_df, results)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("MTA Attribution Dashboard v1.0 | Built with Streamlit")


if __name__ == "__main__":
    main()
