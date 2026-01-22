# 🏦 Banking ML Project - Streamlit Demo

Interactive dashboard for exploring machine learning results from the banking project.

## Features

✨ **Customer Selection**: Browse any of the 495 unique customers

📊 **Cluster Profile**: View customer's behavioral segment and comparison with cluster peers

⚠️ **Anomaly Detection**: See top 5 most unusual transactions with anomaly scores

💡 **Saving Recommendations**: Read personalized, rule-based financial advice with clear explanations

## How to Run

### Prerequisites
Ensure the ML pipeline has been executed to generate output files:
- `outputs/clusters.csv` (customer cluster assignments)
- `outputs/anomalies.csv` (transaction anomaly scores)
- `outputs/recommendations.csv` (personalized recommendations)

### Start the App

```bash
# Navigate to project root
cd Bank_ML

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501` in your browser.

## How It Works

### 1. **Select a Customer**
Use the dropdown in the sidebar to choose an AccountID from 1 to 495.

### 2. **View Cluster Information**
- See which cluster the customer belongs to
- Compare their metrics with cluster averages
- Understand behavioral groupings

### 3. **Check Anomalies**
- Browse top 5 most anomalous transactions
- Anomaly scores range from 0 (normal) to 1 (highly unusual)
- Understand what makes transactions unusual

### 4. **Read Recommendations**
- Get personalized financial advice
- Priority levels: HIGH (action needed), MEDIUM (important), LOW (optimization)
- Clear explanations for why each recommendation was generated

## UI Components

| Component | Purpose |
|-----------|---------|
| **Sidebar** | AccountID dropdown for customer selection |
| **Metrics Cards** | Quick stats (cluster, balance, anomaly count) |
| **Cluster Profile** | Detailed segment comparison |
| **Anomalies Table** | Top unusual transactions with scores |
| **Recommendation Card** | Personalized advice with reasons |

## Data Flow

```
Raw Transactions (2,512)
    ↓
Preprocessing & Features
    ↓
├─→ Clustering (K-Means, k=3)
├─→ Anomaly Detection (Isolation Forest)
└─→ Aggregation to Customer Level
    ↓
Rule-Based Recommendations
    ↓
Streamlit Dashboard (This App)
```

## Interpretation Guide

### Anomaly Scores
- **> 0.5**: Highly unusual (rare pattern, amount, or timing)
- **0.3 - 0.5**: Moderately unusual (differs from typical behavior)
- **< 0.3**: Minor deviation (within normal range)

### Recommendation Priorities
- **HIGH**: Financial risk or major optimization opportunity
- **MEDIUM**: Important consideration, some action recommended
- **LOW**: Informational, no urgent action needed

### Cluster Profiles
- **Cluster 0**: One behavioral segment
- **Cluster 1**: Another distinct segment
- **Cluster 2**: Third customer group

## Key Metrics Explained

| Metric | Definition |
|--------|-----------|
| **Average Balance** | Mean account balance across all transactions |
| **Avg Transaction Amount** | Mean spending per transaction |
| **Transaction Frequency** | Transactions per day |
| **Anomaly Count** | Number of unusual transactions detected |
| **Spending vs Cluster** | % difference from cluster average spending |

## Technology Stack

- **Framework**: Streamlit 1.52.2
- **Data**: Pandas, NumPy
- **ML**: scikit-learn (clustering, anomaly detection)
- **Visualization**: Built-in Streamlit components

## Notes

- App loads data once at startup (cached)
- Select different AccountIDs to explore different customers
- All recommendations are explainable rules-based (no black-box predictions)
- Data is read-only (view-only dashboard)

## Project Context

This app demonstrates a complete ML pipeline for a banking course project:
1. **EDA**: Exploratory data analysis (01_eda.ipynb)
2. **Features**: Feature engineering (02_feature_engineering.ipynb)
3. **Clustering**: Customer segmentation (03_customer_clustering.ipynb)
4. **Anomalies**: Fraud detection (04_fraud_detection.ipynb)
5. **Recommendations**: Saving advice (05_saving_recommendation.ipynb)
6. **Demo**: Interactive dashboard (this app)

See project documentation in the root folder for more details.
