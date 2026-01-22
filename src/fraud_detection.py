"""
Anomaly Detection Module

This module implements unsupervised anomaly detection using Isolation Forest to
identify high-risk or unusual transactions without requiring fraud labels.

Key responsibilities:
- Detect anomalous transactions using Isolation Forest
- Handle extreme class imbalance (anomalies are naturally rare)
- Compute anomaly scores and risk indicators
- Perform anomaly analysis and profiling
- Generate transaction-level risk assessments
- Provide interpretable anomaly indicators

Functions:
    prepare_anomaly_features: Select and prepare features for detection
    train_isolation_forest: Train Isolation Forest model
    detect_anomalies: Identify anomalous transactions
    analyze_anomalies: Generate anomaly statistics and profiles
    score_transactions: Compute risk scores for all transactions
    interpret_anomalies: Provide interpretable explanations

Note:
    No binary fraud labels are created. Results are treated as risk indicators only,
    requiring further investigation for business decision-making.
"""

__version__ = "0.1.0"
__author__ = "ML Course Project"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def prepare_anomaly_features(df, feature_list=None):
    """
    Select and prepare features for anomaly detection.
    
    Identifies key transaction features for Isolation Forest analysis.
    Features are selected to capture transaction size, timing, and account activity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level dataframe with raw features
    feature_list : list, optional
        Specific features to use. If None, uses default set:
        ['TransactionAmount', 'TransactionDuration', 'LoginAttempts',
         'AccountBalance', 'TimeBetweenTransactions']
        Default: None
    
    Returns
    -------
    features_prepared : dict
        Dictionary containing:
        - 'X': array of feature values for anomaly detection
        - 'feature_names': list of feature column names
        - 'scaler': fitted StandardScaler object
        - 'df_features': DataFrame with selected features
    
    Examples
    --------
    >>> df = pd.read_csv('transactions.csv')
    >>> prepared = prepare_anomaly_features(df)
    >>> print(f"Features selected: {prepared['feature_names']}")
    >>> print(f"X shape: {prepared['X'].shape}")
    """
    if feature_list is None:
        feature_list = [
            'TransactionAmount',
            'TransactionDuration',
            'LoginAttempts',
            'AccountBalance',
            'TimeBetweenTransactions'
        ]
    
    # Validate all features exist
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataframe: {missing}")
    
    # Extract features
    df_features = df[feature_list].copy()
    
    # Handle missing values (forward fill, then drop remaining)
    df_features = df_features.fillna(method='ffill').dropna()
    
    # Normalize features for Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    print(f"✓ Prepared {len(feature_list)} features for anomaly detection")
    print(f"  Features: {feature_list}")
    print(f"  Shape: {X_scaled.shape}")
    
    prepared = {
        'X': X_scaled,
        'feature_names': feature_list,
        'scaler': scaler,
        'df_features': df_features
    }
    
    return prepared


def train_isolation_forest(X, contamination=0.05, random_state=42):
    """
    Train Isolation Forest model for anomaly detection.
    
    Isolation Forest works by randomly selecting features and split values,
    isolating observations. Anomalies are isolated more quickly (shorter paths).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Normalized feature matrix for training
    contamination : float, optional
        Expected proportion of anomalies in dataset (0.0 to 0.5)
        Default: 0.05 (5% - conservative estimate for transaction data)
    random_state : int, optional
        Random seed for reproducibility
        Default: 42
    
    Returns
    -------
    model_info : dict
        Dictionary containing:
        - 'model': trained IsolationForest object
        - 'contamination': contamination parameter used
        - 'n_samples_train': number of training samples
        - 'n_features': number of features used
    
    Examples
    --------
    >>> X_prepared = prepare_anomaly_features(df)['X']
    >>> model_info = train_isolation_forest(X_prepared, contamination=0.05)
    >>> print(f"Model trained on {model_info['n_samples_train']} transactions")
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    
    model.fit(X)
    
    model_info = {
        'model': model,
        'contamination': contamination,
        'n_samples_train': X.shape[0],
        'n_features': X.shape[1]
    }
    
    print(f"✓ Isolation Forest trained")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Contamination rate: {contamination*100:.1f}%")
    
    return model_info


def detect_anomalies(model, X, threshold_percentile=95):
    """
    Detect anomalous transactions using trained Isolation Forest.
    
    Computes anomaly scores (-1 for anomalies, 1 for normal) and identifies
    transactions in the top percentile of anomaly scores.
    
    Parameters
    ----------
    model : sklearn.ensemble.IsolationForest
        Trained Isolation Forest model
    X : array-like of shape (n_samples, n_features)
        Normalized feature matrix to score
    threshold_percentile : float, optional
        Percentile for anomaly threshold (higher = stricter)
        Default: 95 (top 5% most anomalous transactions)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'predictions': array of -1 (anomaly) / 1 (normal)
        - 'scores': array of anomaly scores
        - 'is_anomaly': boolean mask for anomalies
        - 'anomaly_indices': indices of anomalous transactions
        - 'n_anomalies': count of detected anomalies
        - 'threshold_score': actual threshold value used
    
    Examples
    --------
    >>> model = train_isolation_forest(X_prepared)['model']
    >>> anomalies = detect_anomalies(model, X_prepared)
    >>> print(f"Found {anomalies['n_anomalies']} anomalies")
    """
    # Get predictions and scores
    predictions = model.predict(X)
    scores = model.score_samples(X)
    
    # Lower scores = more anomalous
    # Find threshold at specified percentile
    threshold = np.percentile(scores, 100 - threshold_percentile)
    is_anomaly = scores <= threshold
    
    anomaly_indices = np.where(is_anomaly)[0]
    
    results = {
        'predictions': predictions,
        'scores': scores,
        'is_anomaly': is_anomaly,
        'anomaly_indices': anomaly_indices,
        'n_anomalies': is_anomaly.sum(),
        'threshold_score': threshold,
        'threshold_percentile': threshold_percentile
    }
    
    print(f"✓ Anomalies detected: {is_anomaly.sum()} transactions ({is_anomaly.sum()/len(X)*100:.2f}%)")
    print(f"  Threshold (percentile {threshold_percentile}): {threshold:.4f}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return results


def score_transactions(df, model, X, scores):
    """
    Add anomaly scores to transaction records.
    
    Creates a risk level classification based on anomaly scores and adds
    the scores and risk categories to the original dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original transaction dataframe
    model : sklearn.ensemble.IsolationForest
        Trained model (used for reference)
    X : array-like
        Normalized features used for scoring
    scores : array-like
        Anomaly scores from model
    
    Returns
    -------
    df_scored : pd.DataFrame
        Copy of input dataframe with added columns:
        - 'AnomalyScore': raw anomaly score
        - 'RiskLevel': categorical risk level (Low/Medium/High)
        - 'IsAnomaly': boolean anomaly indicator
    
    Examples
    --------
    >>> df_scored = score_transactions(df, model, X, scores)
    >>> print(df_scored[['TransactionAmount', 'AnomalyScore', 'RiskLevel']])
    """
    df_scored = df.copy()
    
    # Add anomaly score
    df_scored['AnomalyScore'] = pd.to_numeric(scores, errors='coerce')
    
    # Create risk levels based on score percentiles (as per user correction)
    # Lower score => more anomalous => higher risk
    q25 = df_scored["AnomalyScore"].quantile(0.25)
    q50 = df_scored["AnomalyScore"].quantile(0.50)
    
    def categorize_risk(score, q25, q50):
        if score <= q25:
            return 'High'
        elif score <= q50:
            return 'Medium'
        else:
            return 'Low'
    
    df_scored['RiskLevel'] = df_scored['AnomalyScore'].apply(lambda s: categorize_risk(s, q25, q50))
    df_scored['IsAnomaly'] = model.predict(X) == -1
    
    return df_scored


def analyze_anomalies(df_scored, feature_columns, top_n=20):
    """
    Analyze characteristics of anomalous transactions.
    
    Compares anomalies with normal transactions across all features
    to identify distinguishing patterns.
    
    Parameters
    ----------
    df_scored : pd.DataFrame
        Dataframe with anomaly scores and risk levels
    feature_columns : list
        Feature column names used for analysis
    top_n : int, optional
        Number of top anomalies to display
        Default: 20
    
    Returns
    -------
    analysis : dict
        Dictionary containing:
        - 'anomaly_stats': DataFrame with anomaly vs. normal statistics
        - 'top_anomalies': DataFrame of top N most anomalous transactions
        - 'risk_distribution': Series with count by risk level
        - 'summary': Summary statistics
    
    Examples
    --------
    >>> analysis = analyze_anomalies(df_scored, feature_columns)
    >>> print(analysis['risk_distribution'])
    """
    # Separate anomalies and normal transactions
    anomalies = df_scored[df_scored['IsAnomaly']]
    normal = df_scored[~df_scored['IsAnomaly']]
    
    # Compare statistics
    stats_list = []
    for feature in feature_columns:
        stats_list.append({
            'Feature': feature,
            'Normal_Mean': normal[feature].mean(),
            'Normal_Std': normal[feature].std(),
            'Anomaly_Mean': anomalies[feature].mean(),
            'Anomaly_Std': anomalies[feature].std(),
            'Difference': anomalies[feature].mean() - normal[feature].mean(),
            'Pct_Difference': (anomalies[feature].mean() - normal[feature].mean()) / normal[feature].mean() * 100
        })
    
    anomaly_stats = pd.DataFrame(stats_list)
    
    # Top anomalies
    top_anomalies = df_scored.nlargest(top_n, 'AnomalyScore')[
        ['TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDuration', 
         'LoginAttempts', 'AccountBalance', 'AnomalyScore', 'RiskLevel']
    ]
    
    # Risk distribution
    risk_dist = df_scored['RiskLevel'].value_counts()
    
    # Summary
    summary = {
        'total_transactions': len(df_scored),
        'total_anomalies': df_scored['IsAnomaly'].sum(),
        'pct_anomalies': df_scored['IsAnomaly'].sum() / len(df_scored) * 100,
        'anomaly_score_mean': df_scored['AnomalyScore'].mean(),
        'anomaly_score_std': df_scored['AnomalyScore'].std()
    }
    
    analysis = {
        'anomaly_stats': anomaly_stats,
        'top_anomalies': top_anomalies,
        'risk_distribution': risk_dist,
        'summary': summary
    }
    
    return analysis


def visualize_anomalies(df_scored, feature_columns, figsize=(14, 10)):
    """
    Create comprehensive anomaly detection visualizations.
    
    Generates plots showing anomaly score distributions, feature comparisons,
    and risk level breakdowns.
    
    Parameters
    ----------
    df_scored : pd.DataFrame
        Dataframe with anomaly scores and risk levels
    feature_columns : list
        Feature column names for analysis
    figsize : tuple, optional
        Figure size (width, height) in inches
        Default: (14, 10)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing visualizations
    
    Examples
    --------
    >>> fig = visualize_anomalies(df_scored, feature_columns)
    >>> plt.show()
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Anomaly Score Distribution
    ax = axes[0, 0]
    ax.hist(df_scored['AnomalyScore'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(df_scored[df_scored['IsAnomaly']]['AnomalyScore'].min(), 
               color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
    ax.set_xlabel('Anomaly Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Anomaly Scores', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Risk Level Distribution
    ax = axes[0, 1]
    risk_counts = df_scored['RiskLevel'].value_counts()
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    bars = ax.bar(risk_counts.index, risk_counts.values, 
                   color=[colors.get(x, 'gray') for x in risk_counts.index])
    ax.set_xlabel('Risk Level', fontsize=11)
    ax.set_ylabel('Number of Transactions', fontsize=11)
    ax.set_title('Transaction Distribution by Risk Level', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Transaction Amount by Risk Level
    ax = axes[1, 0]
    df_scored.boxplot(column='TransactionAmount', by='RiskLevel', ax=ax,
                      positions=[0, 1, 2], patch_artist=True)
    ax.set_xlabel('Risk Level', fontsize=11)
    ax.set_ylabel('Transaction Amount ($)', fontsize=11)
    ax.set_title('Transaction Amount by Risk Level', fontsize=12, fontweight='bold')
    plt.sca(ax)
    plt.xticks([1, 2, 3], ['Low', 'Medium', 'High'])
    
    # Plot 4: Feature Comparison (Anomaly vs Normal)
    ax = axes[1, 1]
    anomalies = df_scored[df_scored['IsAnomaly']]
    normal = df_scored[~df_scored['IsAnomaly']]
    
    feature_means_normal = [normal[f].mean() for f in feature_columns[:3]]
    feature_means_anomaly = [anomalies[f].mean() for f in feature_columns[:3]]
    
    x = np.arange(len(feature_columns[:3]))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, feature_means_normal, width, label='Normal', alpha=0.7)
    bars2 = ax.bar(x + width/2, feature_means_anomaly, width, label='Anomalous', alpha=0.7)
    
    ax.set_xlabel('Features', fontsize=11)
    ax.set_ylabel('Mean Value (normalized)', fontsize=11)
    ax.set_title('Feature Comparison: Normal vs Anomalous', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('Transaction', 'Txn') for f in feature_columns[:3]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Anomaly Detection Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig
