"""
Clustering Module

This module implements customer segmentation using clustering algorithms.

Key responsibilities:
- Perform unsupervised clustering (K-Means, DBSCAN, Hierarchical)
- Determine optimal number of clusters
- Analyze cluster characteristics and patterns
- Generate cluster labels and assignments
- Visualize cluster distributions
- Profile customer segments

Functions:
    kmeans_clustering: K-Means clustering implementation
    determine_optimal_clusters: Find best cluster count
    analyze_clusters: Generate cluster statistics and insights
    assign_clusters: Assign data points to clusters
    visualize_clusters: Create cluster visualizations
"""

__version__ = "0.1.0"
__author__ = "ML Course Project"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def determine_optimal_clusters(X, k_range=range(2, 11), metrics=['silhouette', 'davies_bouldin']):
    """
    Determine optimal number of clusters using multiple validation metrics.
    
    Tests different k values and evaluates cluster quality using silhouette score
    and Davies-Bouldin index. Returns comprehensive statistics to support k selection.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix for clustering (should be normalized)
    k_range : range or list, optional
        Range of cluster counts to test
        Default: range(2, 11) tests k=2 through k=10
    metrics : list, optional
        Evaluation metrics to compute: 'silhouette', 'davies_bouldin', 'inertia'
        Default: ['silhouette', 'davies_bouldin']
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'silhouette_scores': list of silhouette scores per k
        - 'davies_bouldin_scores': list of Davies-Bouldin scores per k
        - 'inertias': list of inertias per k
        - 'optimal_k_silhouette': k with highest silhouette score
        - 'optimal_k_davies_bouldin': k with lowest Davies-Bouldin score
        - 'k_values': list of tested k values
    
    Examples
    --------
    >>> X_normalized = np.random.randn(100, 5)
    >>> results = determine_optimal_clusters(X_normalized, k_range=range(2, 8))
    >>> print(f"Optimal k: {results['optimal_k_silhouette']}")
    >>> print(f"Silhouette scores: {results['silhouette_scores']}")
    """
    results = {
        'k_values': list(k_range),
        'silhouette_scores': [],
        'davies_bouldin_scores': [],
        'inertias': []
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Silhouette score (higher is better)
        if 'silhouette' in metrics:
            sil_score = silhouette_score(X, labels)
            results['silhouette_scores'].append(sil_score)
        
        # Davies-Bouldin index (lower is better)
        if 'davies_bouldin' in metrics:
            db_score = davies_bouldin_score(X, labels)
            results['davies_bouldin_scores'].append(db_score)
        
        # Inertia (lower is better, for elbow method)
        if 'inertia' in metrics:
            results['inertias'].append(kmeans.inertia_)
    
    # Identify optimal k values
    if results['silhouette_scores']:
        optimal_idx = np.argmax(results['silhouette_scores'])
        results['optimal_k_silhouette'] = k_range[optimal_idx]
        results['optimal_silhouette_score'] = results['silhouette_scores'][optimal_idx]
    
    if results['davies_bouldin_scores']:
        optimal_idx = np.argmin(results['davies_bouldin_scores'])
        results['optimal_k_davies_bouldin'] = k_range[optimal_idx]
        results['optimal_davies_bouldin_score'] = results['davies_bouldin_scores'][optimal_idx]
    
    return results


def kmeans_clustering(X, n_clusters, random_state=42, n_init=10):
    """
    Train K-Means clustering model.
    
    Fits a K-Means model to the data and returns the trained model along with
    cluster labels and silhouette score for quality assessment.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix for clustering (should be normalized)
    n_clusters : int
        Number of clusters to create
    random_state : int, optional
        Random seed for reproducibility
        Default: 42
    n_init : int, optional
        Number of different centroid seeds to try
        Default: 10
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'model': fitted KMeans object
        - 'labels': array of cluster assignments
        - 'inertia': within-cluster sum of squares
        - 'silhouette_score': silhouette coefficient
        - 'centers': cluster centroids
    
    Examples
    --------
    >>> X_normalized = np.random.randn(100, 5)
    >>> results = kmeans_clustering(X_normalized, n_clusters=3)
    >>> print(f"Silhouette Score: {results['silhouette_score']:.4f}")
    >>> print(f"Cluster labels: {results['labels']}")
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    
    results = {
        'model': kmeans,
        'labels': labels,
        'inertia': kmeans.inertia_,
        'silhouette_score': sil_score,
        'centers': kmeans.cluster_centers_
    }
    
    return results


def assign_clusters(df, labels, id_column='AccountID', cluster_column='ClusterID'):
    """
    Assign cluster labels to dataframe records.
    
    Takes clustering results and merges cluster assignments back to original
    dataframe, creating a clean output with cluster IDs linked to records.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with customer records
    labels : array-like
        Cluster assignments from clustering model
    id_column : str, optional
        Column name containing unique record identifiers
        Default: 'AccountID'
    cluster_column : str, optional
        Name for new column containing cluster IDs
        Default: 'ClusterID'
    
    Returns
    -------
    df_assigned : pd.DataFrame
        Copy of input dataframe with new cluster_column added
    
    Examples
    --------
    >>> df = pd.DataFrame({'AccountID': ['AC001', 'AC002'], 'balance': [100, 200]})
    >>> labels = np.array([0, 1])
    >>> df_assigned = assign_clusters(df, labels)
    >>> print(df_assigned[['AccountID', 'ClusterID']])
    """
    df_assigned = df.copy()
    df_assigned[cluster_column] = labels
    
    return df_assigned


def analyze_clusters(df, cluster_column='ClusterID', feature_columns=None):
    """
    Generate cluster statistics and behavioral profiles.
    
    Computes mean values of features for each cluster and generates
    descriptive statistics useful for cluster interpretation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster assignments
    cluster_column : str, optional
        Name of column containing cluster IDs
        Default: 'ClusterID'
    feature_columns : list, optional
        Columns to compute statistics on
        If None, uses all numeric columns except cluster_column
        Default: None
    
    Returns
    -------
    analysis : dict
        Dictionary containing:
        - 'cluster_profiles': DataFrame with mean feature values per cluster
        - 'cluster_sizes': Series with cluster sizes
        - 'cluster_proportions': Series with cluster proportions
        - 'feature_stats': DataFrame with detailed statistics per feature per cluster
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'ClusterID': [0, 0, 1, 1],
    ...     'amount': [100, 150, 200, 250],
    ...     'frequency': [2, 3, 5, 6]
    ... })
    >>> analysis = analyze_clusters(df)
    >>> print(analysis['cluster_profiles'])
    """
    if feature_columns is None:
        feature_columns = [col for col in df.columns 
                          if df[col].dtype != 'object' and col != cluster_column]
    
    # Cluster size and proportions
    cluster_sizes = df[cluster_column].value_counts().sort_index()
    cluster_proportions = (cluster_sizes / len(df) * 100).round(2)
    
    # Cluster profiles (mean values)
    cluster_profiles = df.groupby(cluster_column)[feature_columns].mean()
    
    # Detailed statistics
    cluster_stats = []
    for cluster_id in sorted(df[cluster_column].unique()):
        cluster_data = df[df[cluster_column] == cluster_id][feature_columns]
        for feature in feature_columns:
            stats = {
                'Cluster': cluster_id,
                'Feature': feature,
                'Mean': cluster_data[feature].mean(),
                'Std': cluster_data[feature].std(),
                'Min': cluster_data[feature].min(),
                'Max': cluster_data[feature].max()
            }
            cluster_stats.append(stats)
    
    feature_stats = pd.DataFrame(cluster_stats)
    
    analysis = {
        'cluster_sizes': cluster_sizes,
        'cluster_proportions': cluster_proportions,
        'cluster_profiles': cluster_profiles,
        'feature_stats': feature_stats
    }
    
    return analysis


def visualize_clusters(X, labels, feature_names=None, figsize=(14, 10)):
    """
    Create comprehensive cluster visualizations.
    
    Generates multiple plots showing cluster distributions, including
    2D projections of first two principal components and feature distributions.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Normalized data used for clustering
    labels : array-like
        Cluster assignments from clustering model
    feature_names : list, optional
        Names of features for plot labels
        If None, uses generic names (Feature 0, Feature 1, ...)
        Default: None
    figsize : tuple, optional
        Figure size (width, height) in inches
        Default: (14, 10)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing cluster visualizations
    
    Examples
    --------
    >>> X_normalized = np.random.randn(100, 5)
    >>> labels = np.array([0]*50 + [1]*50)
    >>> fig = visualize_clusters(X_normalized, labels)
    >>> plt.show()
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    n_clusters = len(np.unique(labels))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: 2D scatter plot of first two features colored by cluster
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title('Cluster Distribution (First 2 Features)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Plot 2: Cluster size distribution
    ax = axes[0, 1]
    unique, counts = np.unique(labels, return_counts=True)
    bars = ax.bar(unique, counts, color=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
    ax.set_xlabel('Cluster ID', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(unique)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Cluster proportions pie chart
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    wedges, texts, autotexts = ax.pie(counts, labels=[f'C{i}' for i in unique],
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Cluster Proportions', fontsize=12, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Plot 4: Feature variance per cluster
    ax = axes[1, 1]
    cluster_stds = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_mask = labels == cluster_id
        mean_std = X[cluster_mask].std(axis=0).mean()
        cluster_stds.append(mean_std)
    
    bars = ax.bar(unique, cluster_stds, color=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
    ax.set_xlabel('Cluster ID', fontsize=11)
    ax.set_ylabel('Average Feature Std Dev', fontsize=11)
    ax.set_title('Feature Variability per Cluster', fontsize=12, fontweight='bold')
    ax.set_xticks(unique)
    
    plt.suptitle('Cluster Analysis Overview', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig