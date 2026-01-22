"""
Feature Engineering Module

This module handles feature creation, extraction, and transformation for ML models.

Key responsibilities:
- Create derived features from raw data
- Handle categorical variable encoding
- Perform feature scaling and normalization
- Feature selection and dimensionality reduction
- Generate interaction and polynomial features
- Time-based feature engineering
- Aggregate transaction-level data into customer profiles

Functions:
    create_features: Generate new features from existing data
    encode_categorical: Convert categorical variables to numerical
    scale_features: Normalize/standardize feature values
    select_features: Identify most important features
    aggregate_customer_features: Create customer-level profiles from transactions
    normalize_features: Apply StandardScaler to numerical features
    feature_engineering_pipeline: Complete FE workflow
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

__version__ = "0.1.0"
__author__ = "ML Course Project"


def aggregate_customer_features(df):
    """
    Aggregate transaction-level data into customer-level behavioral profiles.
    
    This function groups all transactions by AccountID and creates customer-level
    features that capture spending patterns, account activity, and financial stability.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level dataframe (output from preprocessing)
        
    Returns
    -------
    pd.DataFrame
        Customer-level profile with one row per unique AccountID
        
    Features Created
    ----------------
    - total_transaction_amount: Sum of all transaction amounts
    - average_transaction_amount: Mean transaction value
    - std_transaction_amount: Standard deviation of amounts
    - transaction_frequency: Count of transactions per customer
    - average_account_balance: Mean account balance
    - min_account_balance: Minimum observed balance
    - max_account_balance: Maximum observed balance
    - average_login_attempts: Mean login attempts per transaction
    - average_transaction_duration: Mean transaction time
    - debit_ratio: Proportion of debit transactions
    
    Examples
    --------
    >>> customer_profiles = aggregate_customer_features(df_transactions)
    >>> print(customer_profiles.shape)
    (500, 15)
    """
    df = df.copy()
    
    # Validate required columns
    required_cols = ['AccountID', 'TransactionAmount', 'AccountBalance', 
                     'LoginAttempts', 'TransactionDuration', 'TransactionType',
                     'CustomerAge', 'CustomerOccupation']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Aggregate transactions to customer level
    customer_features = df.groupby('AccountID').agg({
        'TransactionAmount': ['sum', 'mean', 'std', 'count'],
        'AccountBalance': ['mean', 'min', 'max'],
        'LoginAttempts': 'mean',
        'TransactionDuration': 'mean',
        'CustomerAge': 'first',
        'CustomerOccupation': 'first',
        'TransactionType': lambda x: (x == 'Debit').sum() / len(x)  # Debit ratio
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['_'.join(col).strip('_') for col in customer_features.columns.values]
    
    # Rename columns for clarity
    rename_mapping = {
        'TransactionAmount_sum': 'total_transaction_amount',
        'TransactionAmount_mean': 'average_transaction_amount',
        'TransactionAmount_std': 'std_transaction_amount',
        'TransactionAmount_count': 'transaction_frequency',
        'AccountBalance_mean': 'average_account_balance',
        'AccountBalance_min': 'min_account_balance',
        'AccountBalance_max': 'max_account_balance',
        'LoginAttempts_mean': 'average_login_attempts',
        'TransactionDuration_mean': 'average_transaction_duration',
        'CustomerAge_first': 'customer_age',
        'CustomerOccupation_first': 'customer_occupation',
        'TransactionType_<lambda>': 'debit_ratio'
    }
    
    customer_features.rename(columns=rename_mapping, inplace=True)
    
    # Handle NaN in std (single transaction customers)
    customer_features['std_transaction_amount'] = customer_features['std_transaction_amount'].fillna(0)
    
    print(f"✓ Created customer profiles for {len(customer_features)} unique customers")
    print(f"✓ Features generated: {len(customer_features.columns)}")
    
    return customer_features


def normalize_features(df, feature_columns=None, scaler=None, fit=True):
    """
    Normalize features using StandardScaler (zero mean, unit variance).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_columns : list of str, optional
        Column names to normalize. If None, normalizes all numeric columns
    scaler : StandardScaler, optional
        Pre-fitted scaler object. If provided, uses this for transformation (fit=False)
    fit : bool, default=True
        If True, fits the scaler on the data. If False, assumes scaler is already fitted
        
    Returns
    -------
    tuple of (pd.DataFrame, StandardScaler)
        - Normalized dataframe with scaled features
        - Fitted scaler object (for applying to new data)
        
    Examples
    --------
    >>> df_scaled, scaler = normalize_features(df_customers)
    >>> print(df_scaled.head())
    
    >>> # Apply same scaler to new data
    >>> df_new_scaled, _ = normalize_features(df_new, scaler=scaler, fit=False)
    """
    df = df.copy()
    
    # Auto-detect numeric columns if not provided
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Validate columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")
    
    # Initialize or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
    
    # Fit and transform or just transform
    if fit:
        X_scaled = scaler.fit_transform(df[feature_columns])
    else:
        X_scaled = scaler.transform(df[feature_columns])
    
    # Replace original values with scaled values
    df[feature_columns] = X_scaled
    
    n_features = len(feature_columns)
    print(f"✓ Normalized {n_features} features using StandardScaler")
    
    return df, scaler


def encode_categorical(df, categorical_columns=None, method='onehot'):
    """
    Encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    categorical_columns : list of str, optional
        Columns to encode. If None, auto-detects object/category columns
    method : str, default='onehot'
        Encoding method: 'onehot' or 'label'
        
    Returns
    -------
    pd.DataFrame
        Dataframe with encoded categorical variables
        
    Examples
    --------
    >>> df_encoded = encode_categorical(df, method='onehot')
    """
    df = df.copy()
    
    # Auto-detect categorical columns
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        print("✓ No categorical columns to encode")
        return df
    
    if method == 'onehot':
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        print(f"✓ One-hot encoded {len(categorical_columns)} categorical columns")
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df


def select_features(df, feature_columns, correlation_threshold=0.95):
    """
    Select features based on correlation threshold to reduce multicollinearity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_columns : list of str
        Columns to analyze
    correlation_threshold : float, default=0.95
        Correlation threshold for dropping features
        
    Returns
    -------
    list of str
        Selected feature columns
        
    Examples
    --------
    >>> selected = select_features(df_scaled, behavioral_features)
    >>> print(f"Selected {len(selected)} out of {len(behavioral_features)} features")
    """
    # Calculate correlation matrix
    corr_matrix = df[feature_columns].corr().abs()
    
    # Select upper triangle to avoid duplicates
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop = [col for col in upper_triangle.columns 
               if any(upper_triangle[col] > correlation_threshold)]
    
    selected_features = [col for col in feature_columns if col not in to_drop]
    
    print(f"✓ Selected {len(selected_features)} features (removed {len(to_drop)} correlated features)")
    
    return selected_features


def feature_engineering_pipeline(df, normalize=True, encode_categorical=True, 
                                 select_correlated=False):
    """
    Execute complete feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataframe (preprocessed)
    normalize : bool, default=True
        Whether to normalize numerical features
    encode_categorical : bool, default=True
        Whether to encode categorical variables
    select_correlated : bool, default=False
        Whether to remove highly correlated features
        
    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - Processed dataframe ready for modeling
        - Dictionary containing metadata (scaler, feature names, etc.)
        
    Examples
    --------
    >>> df_final, metadata = feature_engineering_pipeline(df_transactions)
    >>> print(f"Final shape: {df_final.shape}")
    """
    print("="*80)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Step 1: Aggregate to customer level
    print("\n[Step 1/4] Aggregating transactions to customer profiles...")
    customer_features = aggregate_customer_features(df)
    
    # Step 2: Identify numerical and categorical features
    behavioral_features = [col for col in customer_features.columns 
                          if col not in ['AccountID', 'customer_occupation', 'customer_age']]
    
    # Step 3: Normalize features
    df_processed = customer_features.copy()
    scaler = None
    if normalize:
        print("\n[Step 2/4] Normalizing features...")
        df_processed, scaler = normalize_features(df_processed, feature_columns=behavioral_features)
    
    # Step 4: Encode categorical variables
    if encode_categorical:
        print("\n[Step 3/4] Encoding categorical variables...")
        df_processed = globals()['encode_categorical'](df_processed, 
                                                       categorical_columns=['customer_occupation'],
                                                       method='onehot')
    
    # Step 5: Select non-correlated features
    if select_correlated:
        print("\n[Step 4/4] Selecting non-correlated features...")
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        # Remove AccountID from selection
        numeric_cols = [col for col in numeric_cols if col != 'AccountID']
        selected = select_features(df_processed, numeric_cols)
        df_processed = df_processed[['AccountID'] + selected]
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING PIPELINE COMPLETED")
    print("="*80)
    print(f"Final dataset shape: {df_processed.shape}")
    
    # Store metadata
    metadata = {
        'scaler': scaler,
        'behavioral_features': behavioral_features,
        'num_customers': len(customer_features),
        'num_transactions': len(df)
    }
    
    return df_processed, metadata


if __name__ == "__main__":
    """
    Simple test of feature engineering functions
    """
    print("\n" + "="*80)
    print("TESTING FEATURE ENGINEERING MODULE")
    print("="*80 + "\n")
    
    # Import preprocessing module
    from data_preprocessing import preprocess_pipeline
    
    data_path = Path(__file__).parent.parent / "data" / "bank_transactions_data_2.csv"
    
    if data_path.exists():
        # Load and preprocess data
        print("Loading transaction data...\n")
        df_transactions = preprocess_pipeline(
            filepath=data_path,
            datetime_cols=['TransactionDate', 'PreviousTransactionDate'],
            missing_strategy='drop'
        )
        
        # Run full feature engineering pipeline
        print("\n" + "="*80)
        df_final, metadata = feature_engineering_pipeline(df_transactions)
        
        # Display results
        print("\n" + "-"*80)
        print("Final Dataset Sample:")
        print("-"*80)
        print(df_final.head())
        
        print("\n" + "-"*80)
        print("Metadata:")
        print("-"*80)
        print(f"Number of customers: {metadata['num_customers']}")
        print(f"Number of transactions: {metadata['num_transactions']}")
        print(f"Final features: {len(df_final.columns)}")
        
    else:
        print(f"Warning: Data file not found at {data_path}")
        print("Skipping test execution")
