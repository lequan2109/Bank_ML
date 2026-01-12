"""
Banking ML Project - Streamlit Demo Application

Interactive demonstration of ML pipeline results:
- Customer clustering (K-Means)
- Transaction anomaly detection (Isolation Forest)
- Rule-based saving recommendations

This simple interface allows exploration of individual customer profiles,
their cluster assignments, anomalous transactions, and personalized recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Banking ML - Customer Insights",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================
@st.cache_data
def load_data():
    """Load all processed datasets."""
    base_path = Path(__file__).parent.parent
    
    clusters_df = pd.read_csv(base_path / "outputs" / "clusters.csv")
    anomalies_df = pd.read_csv(base_path / "outputs" / "anomalies.csv")
    recommendations_df = pd.read_csv(base_path / "outputs" / "recommendations.csv")
    transactions_df = pd.read_csv(base_path / "bank_transactions_data_2.csv")
    
    return clusters_df, anomalies_df, recommendations_df, transactions_df

# Load data
clusters_df, anomalies_df, recommendations_df, transactions_df = load_data()

# ============================================================================
# TITLE & INTRO
# ============================================================================
st.markdown("""
    # 🏦 Banking ML Project - Interactive Demo
    
    **Explore customer profiles, anomalies, and personalized recommendations**
    
    This application demonstrates how machine learning results are used to:
    - Segment customers into behavioral clusters
    - Detect unusual transaction patterns
    - Generate explainable financial recommendations
    """)

st.divider()

# ============================================================================
# SIDEBAR: ACCOUNT SELECTION
# ============================================================================
st.sidebar.markdown("## 🔍 Customer Selection")

# Get unique account IDs
account_ids = sorted(clusters_df['AccountID'].unique())
selected_account = st.sidebar.selectbox(
    "Select AccountID:",
    account_ids,
    help="Choose a customer to explore their profile"
)

# ============================================================================
# MAIN CONTENT: CUSTOMER PROFILE
# ============================================================================
st.markdown(f"### Customer Profile: {selected_account}")

# Get customer data
customer_cluster = clusters_df[clusters_df['AccountID'] == selected_account].iloc[0]
customer_rec = recommendations_df[recommendations_df['AccountID'] == selected_account].iloc[0]
customer_anomalies = anomalies_df[anomalies_df['AccountID'] == selected_account]

# Create 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Cluster Assignment",
        value=f"Cluster {int(customer_cluster['ClusterID'])}",
        help="Customer group based on spending and transaction behavior"
    )

with col2:
    st.metric(
        label="Average Balance",
        value=f"${customer_cluster['average_account_balance']:.2f}",
        help="Mean account balance across all transactions"
    )

with col3:
    st.metric(
        label="Anomalous Transactions",
        value=f"{int(customer_rec['AnomalyCount'])}",
        help="Number of unusual transactions detected"
    )

st.divider()

# ============================================================================
# SECTION 1: CLUSTER PROFILE
# ============================================================================
st.markdown("## 📊 Cluster Profile")

cluster_id = int(customer_cluster['ClusterID'])
cluster_members = clusters_df[clusters_df['ClusterID'] == cluster_id]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Cluster Size",
        value=f"{len(cluster_members)} customers",
        help="Number of customers in this cluster"
    )

with col2:
    st.metric(
        label="Cluster Avg Balance",
        value=f"${cluster_members['average_account_balance'].mean():.2f}",
        help="Average balance for all customers in cluster"
    )

with col3:
    st.metric(
        label="Avg Transaction",
        value=f"${customer_cluster['average_transaction_amount']:.2f}",
        help="Customer's average transaction size"
    )

with col4:
    st.metric(
        label="Transaction Frequency",
        value=f"{customer_cluster['transaction_frequency']:.1f}",
        help="Transactions per day"
    )

st.markdown("""
    **What is clustering?**
    
    K-Means clustering groups customers into 3 behavioral segments based on:
    - Average account balance
    - Average transaction amount
    - Transaction frequency
    - Minimum balance maintained
    
    Customers in the same cluster have similar spending patterns and account management behaviors.
    """)

st.divider()

# ============================================================================
# SECTION 2: ANOMALOUS TRANSACTIONS
# ============================================================================
st.markdown("## ⚠️ Anomalous Transactions")

if len(customer_anomalies) > 0:
    st.markdown(f"""
        **Found {len(customer_anomalies)} anomalous transactions** for this customer.
        
        Anomalies are detected using Isolation Forest algorithm, which identifies transactions
        that deviate significantly from the customer's normal behavior.
        """)
    
    # Sort by anomaly score (descending) and show top 5
    top_anomalies = customer_anomalies.nlargest(5, 'AnomalyScore')[
        ['TransactionID', 'TransactionAmount', 'TransactionType', 'AnomalyScore', 'RiskLevel']
    ].copy()
    
    top_anomalies['AnomalyScore'] = top_anomalies['AnomalyScore'].round(3)
    
    st.dataframe(
        top_anomalies,
        width='stretch',
        hide_index=True,
        column_config={
            "TransactionID": "Transaction ID",
            "TransactionAmount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
            "TransactionType": "Type",
            "AnomalyScore": "Anomaly Score",
            "RiskLevel": "Risk"
        }
    )
    
    st.markdown("""
        **Anomaly Score Interpretation:**
        - **Scores > 0.5**: Highly unusual (rare transaction type, amount, or timing)
        - **Scores 0.3-0.5**: Moderately unusual (different from typical pattern)
        - **Scores < 0.3**: Minor deviation (within normal range with small variations)
        """)
else:
    st.info("✅ No anomalous transactions detected for this customer.")

st.divider()

# ============================================================================
# SECTION 3: SAVING RECOMMENDATIONS
# ============================================================================
st.markdown("## 💡 Personalized Saving Recommendation")

# Color map for priority levels
priority_colors = {
    'HIGH': '#FFB6C1',      # Light red
    'MEDIUM': '#FFD700',    # Gold
    'LOW': '#90EE90'        # Light green
}

priority_color = priority_colors.get(customer_rec['PriorityLevel'], '#E0E0E0')

# Display recommendation with background color
st.markdown(f"""
    <div style="background-color: {priority_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {'#DC143C' if customer_rec['PriorityLevel'] == 'HIGH' else '#FF8C00' if customer_rec['PriorityLevel'] == 'MEDIUM' else '#228B22'};">
        <h4 style="margin-top: 0;">Priority: <strong>{customer_rec['PriorityLevel']}</strong></h4>
        <h5>{customer_rec['RecommendationMessage']}</h5>
        <p><strong>Category:</strong> {customer_rec['RecommendationCategory']}</p>
        <p><strong>Why:</strong> {customer_rec['RecommendationReason']}</p>
    </div>
    """, unsafe_allow_html=True)

# Display supporting metrics
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Savings Potential Score:**")
    st.progress(min(customer_rec['SavingsPotential'] / 100, 1.0))
    st.caption(f"Score: {customer_rec['SavingsPotential']:.1f}/100")

with col2:
    st.markdown("**Spending vs Cluster:**")
    spend_diff = customer_rec['SpendingVsCluster']
    if spend_diff > 0:
        st.caption(f"📈 {spend_diff:.1f}% above cluster average")
    elif spend_diff < 0:
        st.caption(f"📉 {abs(spend_diff):.1f}% below cluster average")
    else:
        st.caption("➡️ In line with cluster average")

st.markdown("""
    **How recommendations are generated:**
    
    This system uses **explainable rules** (no black-box ML) combining:
    1. **Cluster profile**: How customer compares to peers
    2. **Spending patterns**: Average transaction vs cluster baseline
    3. **Anomaly count**: Number of unusual transactions
    4. **Account balance**: Minimum balance maintained
    5. **Transaction frequency**: Activity level
    
    Each recommendation includes specific reasons and actionable advice.
    """)

st.divider()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    ---
    **About This Demo:**
    - Data: 2,512 transactions from 495 unique customers (2023)
    - Clustering: K-Means with k=3 optimal clusters
    - Anomaly Detection: Isolation Forest algorithm
    - Recommendations: Rule-based system with 5 business rules
    - All results are explainable and interpretable
    
    📂 Project files: `outputs/` folder contains clusters.csv, anomalies.csv, and recommendations.csv
    """)

