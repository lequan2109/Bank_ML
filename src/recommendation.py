"""
Recommendation Engine Module

This module implements rule-based financial recommendation systems based on
customer behavior analysis, clustering, and anomaly detection results.

NOT Machine Learning: Pure rule-based system designed for explainability.

Key responsibilities:
- Load customer cluster and anomaly data
- Compute spending and financial metrics
- Apply behavioral rules for recommendations
- Generate explainable recommendations with rationale
- Calculate savings potential per customer
- Export recommendations with reasoning

Functions:
    load_customer_data: Load clusters, anomalies, and transaction data
    compute_savings_potential: Calculate personalized savings opportunity
    apply_recommendation_rules: Apply business rules to customer profile
    generate_recommendation: Create single recommendation with rationale
    recommendation_engine: Orchestrate full recommendation pipeline
    export_recommendations: Save recommendations to CSV
"""

__version__ = "0.1.0"
__author__ = "ML Course Project"

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_customer_data(
    cluster_file: str,
    anomaly_file: str,
    transaction_file: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load customer clusters, anomalies, and transaction data.
    
    Reads customer cluster assignments, anomaly scores, and optionally raw
    transaction data to provide comprehensive customer profiles for recommendation
    generation.
    
    Parameters
    ----------
    cluster_file : str
        Path to clusters.csv (AccountID, ClusterID, behavioral features)
    anomaly_file : str
        Path to anomalies.csv (TransactionID, anomaly scores)
    transaction_file : str, optional
        Path to raw transactions CSV for additional context
        Default: None
    
    Returns
    -------
    clusters_df : pd.DataFrame
        Customer cluster assignments with features
    anomalies_df : pd.DataFrame
        Transaction-level anomaly scores
    transactions_df : pd.DataFrame or None
        Raw transaction data if provided, else None
    
    Examples
    --------
    >>> clusters, anomalies, trans = load_customer_data(
    ...     'clusters.csv', 'anomalies.csv', 'transactions.csv')
    >>> print(f"Loaded {len(clusters)} customer profiles")
    Loaded 495 customer profiles
    """
    clusters_df = pd.read_csv(cluster_file)
    anomalies_df = pd.read_csv(anomaly_file)
    
    transactions_df = None
    if transaction_file:
        transactions_df = pd.read_csv(transaction_file)
    
    print(f"✓ Loaded {len(clusters_df)} customer profiles")
    print(f"✓ Loaded {len(anomalies_df)} transaction anomaly scores")
    if transactions_df is not None:
        print(f"✓ Loaded {len(transactions_df)} transactions")
    
    return clusters_df, anomalies_df, transactions_df


def compute_savings_potential(
    customer_row: pd.Series,
    cluster_stats: pd.DataFrame,
    cluster_id: int
) -> Dict[str, float]:
    """
    Calculate personalized savings potential for a customer.
    
    Compares customer's spending and behavior metrics to their cluster average
    to identify specific areas where savings can be achieved.
    
    Parameters
    ----------
    customer_row : pd.Series
        Single customer's behavioral profile
    cluster_stats : pd.DataFrame
        Statistics for all clusters
    cluster_id : int
        The customer's cluster ID
    
    Returns
    -------
    savings_potential : dict
        Dictionary with keys:
        - 'spending_vs_cluster': % above/below cluster average
        - 'balance_safety': ratio of balance to typical spending
        - 'frequency_vs_cluster': transaction frequency ratio
        - 'overall_score': weighted savings potential score
    
    Examples
    --------
    >>> savings = compute_savings_potential(customer, cluster_stats, cluster_id=0)
    >>> print(f"Spending vs cluster: {savings['spending_vs_cluster']:.1f}%")
    """
    cluster_row = cluster_stats.loc[cluster_id]
    
    # Calculate spending differential
    customer_avg_amount = customer_row['average_transaction_amount']
    cluster_avg_amount = cluster_row['average_transaction_amount']
    spending_diff = ((customer_avg_amount - cluster_avg_amount) / 
                     max(cluster_avg_amount, 1)) * 100
    
    # Calculate balance safety (high balance reduces savings urgency)
    customer_balance = customer_row['average_account_balance']
    customer_frequency = customer_row['transaction_frequency']
    typical_spend_per_transaction = customer_avg_amount
    
    balance_to_spending = customer_balance / max(typical_spend_per_transaction * customer_frequency, 1)
    
    # Calculate frequency differential
    customer_freq = customer_row['transaction_frequency']
    cluster_freq = cluster_row['transaction_frequency']
    frequency_diff = ((customer_freq - cluster_freq) / max(cluster_freq, 1)) * 100
    
    # Overall savings potential (composite score)
    # Positive spending_diff + positive frequency_diff = high savings potential
    overall_score = (spending_diff * 0.5 + frequency_diff * 0.3 + 
                     (100 - balance_to_spending * 10) * 0.2)
    
    return {
        'spending_vs_cluster': spending_diff,
        'balance_safety': balance_to_spending,
        'frequency_vs_cluster': frequency_diff,
        'overall_score': max(0, overall_score)
    }


def apply_recommendation_rules(
    customer_row: pd.Series,
    cluster_stats: pd.DataFrame,
    anomaly_count: int,
    savings_potential: Dict[str, float]
) -> List[Dict[str, str]]:
    """
    Apply behavioral rules to generate personalized recommendations.
    
    Uses customer profile data to determine which financial advice rules apply,
    generating one or more recommendations with clear explanations.
    
    Parameters
    ----------
    customer_row : pd.Series
        Customer behavioral profile
    cluster_stats : pd.DataFrame
        Cluster statistics for comparison
    anomaly_count : int
        Number of anomalous transactions for this customer
    savings_potential : dict
        Calculated savings potential metrics
    
    Returns
    -------
    recommendations : list of dict
        Each dict contains:
        - 'message': Recommendation text
        - 'reason': Explanation of the rule
        - 'priority': 'HIGH', 'MEDIUM', or 'LOW'
        - 'metric_value': Specific value triggering the rule
    
    Examples
    --------
    >>> recs = apply_recommendation_rules(customer, cluster_stats, 5, potential)
    >>> for rec in recs:
    ...     print(f"{rec['priority']}: {rec['message']}")
    """
    recommendations = []
    cluster_id = int(customer_row['ClusterID'])
    cluster_row = cluster_stats.loc[cluster_id]
    
    # RULE 1: High spending relative to cluster
    spending_diff = savings_potential['spending_vs_cluster']
    if spending_diff > 30:
        recommendations.append({
            'message': 'Consider reducing transaction amounts',
            'reason': f'Your avg transaction (${customer_row["average_transaction_amount"]:.2f}) is {spending_diff:.0f}% above cluster average',
            'priority': 'HIGH',
            'metric_value': f'{spending_diff:.1f}% above cluster',
            'category': 'SPENDING_CONTROL'
        })
    
    # RULE 2: Frequent transactions with low balance
    balance = customer_row['average_account_balance']
    frequency = customer_row['transaction_frequency']
    if frequency > cluster_row['transaction_frequency'] + 2 and balance < customer_row['std_transaction_amount'] * 5:
        recommendations.append({
            'message': 'Build emergency fund before increasing transaction frequency',
            'reason': f'High transaction frequency ({frequency:.0f}) with relatively low avg balance (${balance:.2f})',
            'priority': 'HIGH',
            'metric_value': f'Balance risk score: {balance/max(customer_row["average_transaction_amount"], 1):.1f}x',
            'category': 'EMERGENCY_FUND'
        })
    
    # RULE 3: Very low balance (risky)
    min_balance = customer_row['min_account_balance']
    if min_balance < 500 and customer_row['transaction_frequency'] > 3:
        recommendations.append({
            'message': 'Establish minimum balance threshold to avoid overdrafts',
            'reason': f'Minimum balance reached ${min_balance:.2f} with frequent transactions',
            'priority': 'HIGH',
            'metric_value': f'Min balance: ${min_balance:.2f}',
            'category': 'OVERDRAFT_PROTECTION'
        })
    
    # RULE 4: High anomaly count indicates irregular spending
    if anomaly_count >= 5:
        recommendations.append({
            'message': 'Review irregular transactions for budgeting patterns',
            'reason': f'Detected {anomaly_count} anomalous transactions indicating inconsistent spending',
            'priority': 'MEDIUM',
            'metric_value': f'{anomaly_count} anomalies',
            'category': 'SPENDING_PATTERNS'
        })
    elif anomaly_count >= 2:
        recommendations.append({
            'message': 'Monitor unusual transactions for budget deviations',
            'reason': f'Found {anomaly_count} anomalous transactions - may indicate unplanned expenses',
            'priority': 'MEDIUM',
            'metric_value': f'{anomaly_count} anomalies',
            'category': 'SPENDING_PATTERNS'
        })
    
    # RULE 5: Moderate spending vs cluster (room for optimization)
    if 10 < spending_diff <= 30:
        recommendations.append({
            'message': 'Optimize transaction amounts to align with your spending group',
            'reason': f'Your spending is {spending_diff:.0f}% above similar customers - small changes add up',
            'priority': 'MEDIUM',
            'metric_value': f'{spending_diff:.1f}% above cluster',
            'category': 'SPENDING_OPTIMIZATION'
        })
    
    # RULE 6: High frequency - encourage budgeting
    if frequency > cluster_row['transaction_frequency'] * 1.5:
        recommendations.append({
            'message': 'Set up budget tracking for frequent transactions',
            'reason': f'Transaction frequency ({frequency:.0f}) is significantly higher than similar customers',
            'priority': 'MEDIUM',
            'metric_value': f'{frequency:.0f} transactions',
            'category': 'BUDGET_TRACKING'
        })
    
    # RULE 7: Excellent balance management (positive reinforcement)
    balance_score = customer_row['average_account_balance'] / max(customer_row['average_transaction_amount'] * frequency, 1)
    if balance_score > 3.0 and anomaly_count < 2:
        recommendations.append({
            'message': 'Excellent financial discipline - consider higher-yield savings options',
            'reason': f'Strong balance management with {balance_score:.1f}x typical monthly spending in reserves',
            'priority': 'LOW',
            'metric_value': f'Balance ratio: {balance_score:.1f}x',
            'category': 'OPTIMIZATION_OPPORTUNITY'
        })
    
    # RULE 8: Low transaction duration with high frequency (efficiency)
    if customer_row['average_transaction_duration'] < 50 and frequency > cluster_row['transaction_frequency']:
        recommendations.append({
            'message': 'Consider using automated tools for high-frequency, quick transactions',
            'reason': f'Quick transaction pattern ({customer_row["average_transaction_duration"]:.0f}s avg) with high frequency',
            'priority': 'LOW',
            'metric_value': f'{customer_row["average_transaction_duration"]:.0f}s avg duration',
            'category': 'AUTOMATION_OPPORTUNITY'
        })
    
    # If no other rules triggered, provide general guidance
    if not recommendations:
        recommendations.append({
            'message': 'Continue current spending habits - within normal range for your segment',
            'reason': f'Cluster {cluster_id} member with balanced financial metrics',
            'priority': 'LOW',
            'metric_value': 'Balanced profile',
            'category': 'GENERAL_GUIDANCE'
        })
    
    return recommendations


def generate_recommendation(
    account_id: str,
    customer_row: pd.Series,
    cluster_stats: pd.DataFrame,
    anomaly_count: int,
    cluster_profiles: pd.DataFrame
) -> pd.Series:
    """
    Generate complete recommendation record for a single customer.
    
    Orchestrates the recommendation generation process: calculates savings
    potential, applies rules, selects top recommendation, and creates output row.
    
    Parameters
    ----------
    account_id : str
        Customer's account ID (e.g., 'AC00001')
    customer_row : pd.Series
        Customer's behavioral profile
    cluster_stats : pd.DataFrame
        Cluster-level statistics
    anomaly_count : int
        Number of anomalies for this customer
    cluster_profiles : pd.DataFrame
        Customer cluster assignment data
    
    Returns
    -------
    recommendation : pd.Series
        Single row with all recommendation fields:
        - AccountID
        - ClusterID
        - RecommendationMessage
        - RecommendationReason
        - PriorityLevel
        - SavingsPotential
        - AnomalyCount
        - AvgTransactionAmount
        - AverageBalance
        - RecommendationCategory
    
    Examples
    --------
    >>> rec = generate_recommendation('AC00001', customer_row, cluster_stats, 3, profiles)
    >>> print(rec['RecommendationMessage'])
    """
    # Calculate savings potential
    cluster_id = int(customer_row['ClusterID'])
    savings = compute_savings_potential(customer_row, cluster_stats, cluster_id)
    
    # Apply rules
    recs = apply_recommendation_rules(customer_row, cluster_stats, anomaly_count, savings)
    
    # Select top recommendation (highest priority, then highest savings potential)
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    top_rec = sorted(recs, key=lambda x: (priority_order[x['priority']], -savings['overall_score']))[0]
    
    # Create output record
    return pd.Series({
        'AccountID': account_id,
        'ClusterID': cluster_id,
        'RecommendationMessage': top_rec['message'],
        'RecommendationReason': top_rec['reason'],
        'PriorityLevel': top_rec['priority'],
        'RecommendationCategory': top_rec['category'],
        'SavingsPotential': savings['overall_score'],
        'AnomalyCount': anomaly_count,
        'SpendingVsCluster': savings['spending_vs_cluster'],
        'AvgTransactionAmount': customer_row['average_transaction_amount'],
        'AverageBalance': customer_row['average_account_balance'],
        'TransactionFrequency': customer_row['transaction_frequency'],
        'MetricValue': top_rec['metric_value']
    })


def recommendation_engine(
    clusters_df,
    anomalies_df,
    transactions_df = None
) -> pd.DataFrame:
    """
    Complete recommendation generation pipeline.
    
    Generates recommendations by combining customer clusters and anomaly data.
    Accepts both DataFrames and file paths for maximum flexibility.
    
    Parameters
    ----------
    clusters_df : pd.DataFrame or str
        Customer cluster dataframe or path to clusters.csv
    anomalies_df : pd.DataFrame or str
        Transaction anomalies dataframe or path to anomalies.csv
    transactions_df : pd.DataFrame or str, optional
        Transaction data dataframe or path to transactions CSV
        Default: None
    
    Returns
    -------
    recommendations_df : pd.DataFrame
        Complete recommendations with one row per customer
    
    Examples
    --------
    >>> clusters = pd.read_csv('clusters.csv')
    >>> anomalies = pd.read_csv('anomalies.csv')
    >>> recs_df = recommendation_engine(clusters, anomalies)
    >>> print(f"Generated {len(recs_df)} recommendations")
    """
    # Load data if paths provided (for backward compatibility)
    if isinstance(clusters_df, str) or isinstance(anomalies_df, str):
        clusters_df, anomalies_df, transactions_df = load_customer_data(
            clusters_df, anomalies_df, transactions_df
        )
    
    # Count anomalies per customer
    if 'AccountID' in anomalies_df.columns:
        anomaly_counts = anomalies_df['AccountID'].value_counts()
    else:
        # If no AccountID in anomalies, assume transaction-level with TransactionID
        # Map TransactionID back to AccountID via transactions
        if transactions_df is not None:
            trans_merged = anomalies_df.merge(
                transactions_df[['TransactionID', 'AccountID']],
                on='TransactionID',
                how='left'
            )
            anomaly_counts = trans_merged['AccountID'].value_counts()
        else:
            # Default: all customers have 0 anomalies if no mapping possible
            anomaly_counts = pd.Series(0, index=clusters_df['AccountID'])
    
    # Compute cluster statistics for comparison
    cluster_stats = clusters_df.groupby('ClusterID')[[
        'average_transaction_amount', 'transaction_frequency',
        'average_account_balance', 'std_transaction_amount'
    ]].mean()
    
    # Generate recommendations for each customer
    recommendations = []
    for idx, row in clusters_df.iterrows():
        account_id = row['AccountID']
        anomaly_count = anomaly_counts.get(account_id, 0)
        
        rec = generate_recommendation(
            account_id, row, cluster_stats, anomaly_count, clusters_df
        )
        recommendations.append(rec)
    
    recommendations_df = pd.DataFrame(recommendations)
    
    print(f"\n✓ Generated {len(recommendations_df)} personalized recommendations")
    print(f"\nRecommendation Priority Distribution:")
    print(recommendations_df['PriorityLevel'].value_counts().sort_index())
    
    return recommendations_df


def export_recommendations(
    recommendations_df: pd.DataFrame,
    output_file: str
) -> None:
    """
    Export recommendations to CSV file.
    
    Saves recommendation dataframe to CSV with all fields properly formatted
    for downstream business use.
    
    Parameters
    ----------
    recommendations_df : pd.DataFrame
        Complete recommendations dataframe
    output_file : str
        Output file path (e.g., 'recommendations.csv')
    
    Returns
    -------
    None
        Prints confirmation message with file location
    
    Examples
    --------
    >>> export_recommendations(recs_df, 'recommendations.csv')
    ✓ Exported 495 recommendations to recommendations.csv
    """
    recommendations_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Exported {len(recommendations_df)} recommendations to {output_file}")
    print(f"  Columns: {', '.join(recommendations_df.columns.tolist())}")
    print(f"\nTop 5 HIGH priority recommendations:")
    high_priority = recommendations_df[recommendations_df['PriorityLevel'] == 'HIGH']
    for idx, rec in high_priority.head(5).iterrows():
        print(f"  • {rec['AccountID']}: {rec['RecommendationMessage']}")
