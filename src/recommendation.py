"""
Recommendation Engine Module

Rule-based financial recommendations based on:
- Customer clustering (customer-level features)
- Anomaly detection results (transaction-level with IsAnomaly flag)

IMPORTANT:
- outputs/anomalies.csv in this project contains ALL transactions
  and uses IsAnomaly to mark anomalous ones.
- Therefore, anomaly counting MUST filter IsAnomaly == True.
"""

__version__ = "0.1.1"
__author__ = "ML Course Project (patched)"

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robust bool conversion for IsAnomaly column (True/False, 0/1, 'true'/'false', etc.)."""
    if s is None:
        return pd.Series([False] * 0)

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    # numeric 0/1
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).eq(1)

    # strings
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(["true", "1", "yes", "y", "t"])


def load_customer_data(
    cluster_file: str,
    anomaly_file: str,
    transaction_file: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load customer clusters, anomalies, and optionally transactions.
    """
    clusters_df = pd.read_csv(cluster_file)
    anomalies_df = pd.read_csv(anomaly_file)

    transactions_df = None
    if transaction_file:
        transactions_df = pd.read_csv(transaction_file)

    print(f"✓ Loaded {len(clusters_df)} customer profiles")
    print(f"✓ Loaded {len(anomalies_df)} transaction rows from anomalies.csv")
    if transactions_df is not None:
        print(f"✓ Loaded {len(transactions_df)} transactions")

    return clusters_df, anomalies_df, transactions_df


def compute_savings_potential(
    customer_row: pd.Series,
    cluster_stats: pd.DataFrame,
    cluster_id: int
) -> Dict[str, float]:
    """
    Compare customer's metrics vs cluster average to estimate savings potential.
    """
    cluster_row = cluster_stats.loc[cluster_id]

    customer_avg_amount = float(customer_row.get("average_transaction_amount", 0))
    cluster_avg_amount = float(cluster_row.get("average_transaction_amount", 0))
    spending_diff = ((customer_avg_amount - cluster_avg_amount) / max(cluster_avg_amount, 1)) * 100

    customer_balance = float(customer_row.get("average_account_balance", 0))
    customer_frequency = float(customer_row.get("transaction_frequency", 0))
    typical_spend_per_transaction = customer_avg_amount
    balance_to_spending = customer_balance / max(typical_spend_per_transaction * customer_frequency, 1)

    customer_freq = float(customer_row.get("transaction_frequency", 0))
    cluster_freq = float(cluster_row.get("transaction_frequency", 0))
    frequency_diff = ((customer_freq - cluster_freq) / max(cluster_freq, 1)) * 100

    overall_score = (spending_diff * 0.5 + frequency_diff * 0.3 + (100 - balance_to_spending * 10) * 0.2)

    return {
        "spending_vs_cluster": float(spending_diff),
        "balance_safety": float(balance_to_spending),
        "frequency_vs_cluster": float(frequency_diff),
        "overall_score": float(max(0, overall_score)),
    }


def apply_recommendation_rules(
    customer_row: pd.Series,
    cluster_stats: pd.DataFrame,
    anomaly_count: int,
    savings_potential: Dict[str, float]
) -> List[Dict[str, str]]:
    """
    Apply business rules to generate explainable recommendations.
    """
    recommendations: List[Dict[str, str]] = []

    cluster_id = int(customer_row["ClusterID"])
    cluster_row = cluster_stats.loc[cluster_id]

    spending_diff = savings_potential["spending_vs_cluster"]
    balance = float(customer_row.get("average_account_balance", 0))
    frequency = float(customer_row.get("transaction_frequency", 0))
    std_amt = float(customer_row.get("std_transaction_amount", 0))
    min_balance = float(customer_row.get("min_account_balance", 0))

    # RULE 1: High spending relative to cluster
    if spending_diff > 30:
        recommendations.append({
            "message": "Consider reducing transaction amounts",
            "reason": f'Your avg transaction (${customer_row["average_transaction_amount"]:.2f}) is {spending_diff:.0f}% above cluster average',
            "priority": "HIGH",
            "metric_value": f"{spending_diff:.1f}% above cluster",
            "category": "SPENDING_CONTROL"
        })

    # RULE 2: Frequent transactions with low balance
    if frequency > float(cluster_row["transaction_frequency"]) + 2 and balance < std_amt * 5:
        recommendations.append({
            "message": "Build emergency fund before increasing transaction frequency",
            "reason": f"High transaction frequency ({frequency:.0f}) with relatively low avg balance (${balance:.2f})",
            "priority": "HIGH",
            "metric_value": f"Balance risk score: {balance/max(float(customer_row['average_transaction_amount']), 1):.1f}x",
            "category": "EMERGENCY_FUND"
        })

    # RULE 3: Very low balance
    if min_balance < 500 and frequency > 3:
        recommendations.append({
            "message": "Establish minimum balance threshold to avoid overdrafts",
            "reason": f"Minimum balance reached ${min_balance:.2f} with frequent transactions",
            "priority": "HIGH",
            "metric_value": f"Min balance: ${min_balance:.2f}",
            "category": "OVERDRAFT_PROTECTION"
        })

    # RULE 4: Irregular spending (anomaly count)
    if anomaly_count >= 5:
        recommendations.append({
            "message": "Review irregular transactions for budgeting patterns",
            "reason": f"Detected {anomaly_count} anomalous transactions indicating inconsistent spending",
            "priority": "MEDIUM",
            "metric_value": f"{anomaly_count} anomalies",
            "category": "SPENDING_PATTERNS"
        })
    elif anomaly_count >= 2:
        recommendations.append({
            "message": "Monitor unusual transactions for budget deviations",
            "reason": f"Found {anomaly_count} anomalous transactions - may indicate unplanned expenses",
            "priority": "MEDIUM",
            "metric_value": f"{anomaly_count} anomalies",
            "category": "SPENDING_PATTERNS"
        })

    # RULE 5: Moderate spending above cluster
    if 10 < spending_diff <= 30:
        recommendations.append({
            "message": "Optimize transaction amounts to align with your spending group",
            "reason": f"Your spending is {spending_diff:.0f}% above similar customers - small changes add up",
            "priority": "MEDIUM",
            "metric_value": f"{spending_diff:.1f}% above cluster",
            "category": "SPENDING_OPTIMIZATION"
        })

    # RULE 6: High frequency
    if frequency > float(cluster_row["transaction_frequency"]) * 1.5:
        recommendations.append({
            "message": "Set up budget tracking for frequent transactions",
            "reason": f"Transaction frequency ({frequency:.0f}) is significantly higher than similar customers",
            "priority": "MEDIUM",
            "metric_value": f"{frequency:.0f} transactions",
            "category": "BUDGET_TRACKING"
        })

    # RULE 7: Positive reinforcement
    balance_score = float(customer_row.get("average_account_balance", 0)) / max(float(customer_row.get("average_transaction_amount", 0)) * frequency, 1)
    if balance_score > 3.0 and anomaly_count < 2:
        recommendations.append({
            "message": "Excellent financial discipline - consider higher-yield savings options",
            "reason": f"Strong balance management with {balance_score:.1f}x typical monthly spending in reserves",
            "priority": "LOW",
            "metric_value": f"Balance ratio: {balance_score:.1f}x",
            "category": "OPTIMIZATION_OPPORTUNITY"
        })

    # RULE 8: Fast transactions with high frequency
    avg_dur = float(customer_row.get("average_transaction_duration", 0))
    if avg_dur < 50 and frequency > float(cluster_row["transaction_frequency"]):
        recommendations.append({
            "message": "Consider using automated tools for high-frequency, quick transactions",
            "reason": f"Quick transaction pattern ({avg_dur:.0f}s avg) with high frequency",
            "priority": "LOW",
            "metric_value": f"{avg_dur:.0f}s avg duration",
            "category": "AUTOMATION_OPPORTUNITY"
        })

    if not recommendations:
        recommendations.append({
            "message": "Continue current spending habits - within normal range for your segment",
            "reason": f"Cluster {cluster_id} member with balanced financial metrics",
            "priority": "LOW",
            "metric_value": "Balanced profile",
            "category": "GENERAL_GUIDANCE"
        })

    return recommendations


def generate_recommendation(
    account_id: str,
    customer_row: pd.Series,
    cluster_stats: pd.DataFrame,
    anomaly_count: int
) -> pd.Series:
    """
    Generate a single recommendation record for one customer.
    """
    cluster_id = int(customer_row["ClusterID"])
    savings = compute_savings_potential(customer_row, cluster_stats, cluster_id)
    recs = apply_recommendation_rules(customer_row, cluster_stats, anomaly_count, savings)

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    top_rec = sorted(recs, key=lambda x: (priority_order[x["priority"]], -savings["overall_score"]))[0]

    return pd.Series({
        "AccountID": account_id,
        "ClusterID": cluster_id,
        "RecommendationMessage": top_rec["message"],
        "RecommendationReason": top_rec["reason"],
        "PriorityLevel": top_rec["priority"],
        "RecommendationCategory": top_rec["category"],
        "SavingsPotential": savings["overall_score"],
        "AnomalyCount": int(anomaly_count),
        "SpendingVsCluster": savings["spending_vs_cluster"],
        "AvgTransactionAmount": float(customer_row.get("average_transaction_amount", 0)),
        "AverageBalance": float(customer_row.get("average_account_balance", 0)),
        "TransactionFrequency": float(customer_row.get("transaction_frequency", 0)),
        "MetricValue": top_rec["metric_value"],
    })


def _count_anomalies_per_customer(
    anomalies_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    transactions_df: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Count anomalies per AccountID correctly.

    Key rule:
    - If anomalies_df contains ALL transactions (common in this project),
      we MUST filter IsAnomaly == True before counting.

    Returns a Series indexed by AccountID with integer counts.
    """
    # preferred: anomalies_df already has AccountID
    if "AccountID" in anomalies_df.columns:
        if "IsAnomaly" in anomalies_df.columns:
            is_anom = _to_bool_series(anomalies_df["IsAnomaly"])
            counts = anomalies_df.loc[is_anom, "AccountID"].value_counts()
        else:
            # fallback: treat all rows as anomalies if no flag exists
            counts = anomalies_df["AccountID"].value_counts()

        # ensure all customers exist with 0 default
        all_ids = pd.Index(clusters_df["AccountID"].astype(str).unique(), name="AccountID")
        return counts.reindex(all_ids, fill_value=0).astype(int)

    # otherwise: map TransactionID -> AccountID using transactions_df if possible
    if transactions_df is not None and "TransactionID" in anomalies_df.columns:
        if not {"TransactionID", "AccountID"}.issubset(transactions_df.columns):
            raise ValueError("transactions_df must contain TransactionID and AccountID to map anomalies to customers.")

        tmp = anomalies_df.copy()
        tmp["TransactionID"] = tmp["TransactionID"].astype(str).str.strip()
        tx = transactions_df[["TransactionID", "AccountID"]].copy()
        tx["TransactionID"] = tx["TransactionID"].astype(str).str.strip()

        mapped = tmp.merge(tx, on="TransactionID", how="left")

        if "IsAnomaly" in mapped.columns:
            is_anom = _to_bool_series(mapped["IsAnomaly"])
            counts = mapped.loc[is_anom, "AccountID"].value_counts()
        else:
            counts = mapped["AccountID"].value_counts()

        all_ids = pd.Index(clusters_df["AccountID"].astype(str).unique(), name="AccountID")
        return counts.reindex(all_ids, fill_value=0).astype(int)

    # fallback: everyone 0
    all_ids = pd.Index(clusters_df["AccountID"].astype(str).unique(), name="AccountID")
    return pd.Series(0, index=all_ids, dtype=int, name="AnomalyCount")


def recommendation_engine(
    clusters_df: Union[pd.DataFrame, str],
    anomalies_df: Union[pd.DataFrame, str],
    transactions_df: Optional[Union[pd.DataFrame, str]] = None
) -> pd.DataFrame:
    """
    Generate recommendations (1 row per customer).
    Accepts DataFrames or file paths.
    """
    # Load from path if needed
    if isinstance(clusters_df, str) or isinstance(anomalies_df, str):
        clusters_df, anomalies_df, transactions_df = load_customer_data(
            clusters_df, anomalies_df, transactions_df if isinstance(transactions_df, str) else None
        )
    elif isinstance(transactions_df, str):
        transactions_df = pd.read_csv(transactions_df)

    # sanity columns for rules
    required_cluster_cols = [
        "AccountID", "ClusterID",
        "average_transaction_amount", "transaction_frequency",
        "average_account_balance", "std_transaction_amount",
        "min_account_balance", "average_transaction_duration"
    ]
    missing = [c for c in required_cluster_cols if c not in clusters_df.columns]
    if missing:
        raise ValueError(f"clusters_df missing required columns for recommendations: {missing}")

    # Count anomalies correctly
    anomaly_counts = _count_anomalies_per_customer(anomalies_df, clusters_df, transactions_df)

    # Cluster statistics
    cluster_stats = clusters_df.groupby("ClusterID")[[
        "average_transaction_amount",
        "transaction_frequency",
        "average_account_balance",
        "std_transaction_amount"
    ]].mean()

    # Generate recommendations
    recommendations = []
    for _, row in clusters_df.iterrows():
        account_id = str(row["AccountID"])
        anomaly_count = int(anomaly_counts.get(account_id, 0))

        rec = generate_recommendation(
            account_id=account_id,
            customer_row=row,
            cluster_stats=cluster_stats,
            anomaly_count=anomaly_count
        )
        recommendations.append(rec)

    recommendations_df = pd.DataFrame(recommendations)

    print(f"\n✓ Generated {len(recommendations_df)} personalized recommendations")
    print("\nRecommendation Priority Distribution:")
    print(recommendations_df["PriorityLevel"].value_counts().sort_index())

    return recommendations_df


def export_recommendations(
    recommendations_df: pd.DataFrame,
    output_file: str
) -> None:
    """Export recommendations to CSV (UTF-8 with BOM for Excel friendliness)."""
    recommendations_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n✓ Exported {len(recommendations_df)} recommendations to {output_file}")
    print(f"  Columns: {', '.join(recommendations_df.columns.tolist())}")
    print("\nTop 5 HIGH priority recommendations:")
    high_priority = recommendations_df[recommendations_df["PriorityLevel"] == "HIGH"]
    for _, rec in high_priority.head(5).iterrows():
        print(f"  • {rec['AccountID']}: {rec['RecommendationMessage']}")
