# Multi-Touch Attribution (MTA) Dashboard

A **production-ready Multi-Touch Attribution analytics platform** for marketing data scientists, built with Streamlit and Python.

## Features

### Attribution Models
| Model | Description |
|-------|-------------|
| **First-Touch** | 100% credit to the first touchpoint |
| **Last-Touch** | 100% credit to the last touchpoint |
| **Linear** | Equal credit across all touchpoints |
| **Time-Decay** | Exponential decay — recent touches receive more credit |
| **Position-Based (U-Shaped)** | 40% first, 40% last, 20% distributed across middle |

### Dashboard Pages
- 📊 **Attribution Overview** — KPIs + model comparison
- 💰 **Channel Performance** — ROI, ROAS, cost per conversion
- 🔄 **Journey Analysis** — Top converting paths, first→last touch heatmap
- 📈 **Model Comparison** — Heatmaps and pairwise difference charts

### Other Features
- Load your own CSV or generate realistic synthetic data
- Real-time attribution calculations
- Interactive filters (date range, channels, campaigns)
- Export results as CSV
- Configurable model parameters (decay rate, position weights)

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/abhishek0841/mta-attribution-dashboard.git
cd mta-attribution-dashboard
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will open at **http://localhost:8501** 🎉

## Project Structure

```
mta-attribution-dashboard/
├── data/
│   ├── __init__.py
│   └── synthetic_generator.py   # Realistic marketing dataset generator
├── models/
│   ├── __init__.py
│   └── attribution.py           # 5 attribution model implementations
├── app/
│   ├── __init__.py
│   ├── dashboard.py             # Main Streamlit entry point
│   ├── utils.py                 # Shared helpers & chart builders
│   └── pages/
│       ├── __init__.py
│       ├── overview.py          # Attribution Overview page
│       ├── channel_analysis.py  # Channel Performance page
│       ├── journey_analysis.py  # Journey Analysis page
│       └── model_comparison.py  # Model Comparison page
├── notebooks/
│   └── mta_analysis_example.ipynb  # Jupyter analysis walkthrough
├── requirements.txt
├── config.yaml                  # Configuration management
└── README.md
```

## Using Your Own Data

Upload a CSV file via the sidebar. The file must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | str | Unique user identifier |
| `journey_id` | str | Unique journey identifier |
| `touchpoint_order` | int | Order of touch within journey (1-based) |
| `channel` | str | Marketing channel name |
| `converted` | int | 1 if journey converted, 0 otherwise |
| `revenue` | float | Revenue attributed to conversion (last touch in journey) |
| `cost` | float | Cost of this touchpoint |
| `timestamp` | datetime | Touchpoint timestamp (optional) |
| `campaign` | str | Campaign name (optional) |
| `device` | str | Device type (optional) |
| `geography` | str | Geographic region (optional) |

## Configuration

Edit `config.yaml` to change defaults:

```yaml
data:
  n_users: 5000
  conversion_rate: 0.35

models:
  time_decay_rate: 7          # half-life in days
  position_first_weight: 0.40
  position_last_weight: 0.40
```

## Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New app"** → select your repo
4. Set **Main file path** to `app/dashboard.py`
5. Click **Deploy** 🚀

## Tech Stack

- **Python 3.9+**
- **Streamlit** — dashboard & UI
- **Pandas + NumPy** — data processing
- **Plotly** — interactive visualizations
- **scikit-learn** — statistical utilities
- **PyYAML** — configuration management

## License

MIT License — see [LICENSE](LICENSE) for details.