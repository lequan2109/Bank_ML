"""
Data Preprocessing Module

This module handles all data loading, cleaning, validation, and transformation tasks.

Key responsibilities:
- Load data from various sources (CSV, databases, APIs)
- Handle missing values and outliers
- Data type conversions and standardization
- Data validation and quality checks
- Save processed data for downstream modules

Functions:
    load_data: Load raw data from file/source
    clean_missing_values: Handle missing values
    convert_datetime_columns: Convert date columns to datetime format
    create_time_features: Create derived temporal features
"""

import pandas as pd
import numpy as np
from pathlib import Path

__version__ = "0.1.0"
__author__ = "ML Course Project"


def load_data(filepath):
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file to load
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file is not a valid CSV
        
    Examples
    --------
    >>> df = load_data('data/transactions.csv')
    >>> print(df.shape)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix.lower() != '.csv':
        raise ValueError(f"Expected CSV file, got: {filepath.suffix}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded {filepath.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def clean_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    strategy : str, default='drop'
        Strategy for handling missing values:
        - 'drop': Remove rows with missing values
        - 'mean': Fill numeric columns with mean
        - 'forward_fill': Forward fill missing values
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with missing values handled
        
    Raises
    ------
    ValueError
        If strategy is not recognized
        
    Examples
    --------
    >>> df_clean = clean_missing_values(df, strategy='drop')
    >>> print(f"Rows removed: {len(df) - len(df_clean)}")
    """
    df = df.copy()
    
    if strategy not in ['drop', 'mean', 'forward_fill']:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    missing_count_before = df.isnull().sum().sum()
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    missing_count_after = df.isnull().sum().sum()
    print(f"✓ Cleaned missing values: {missing_count_before} → {missing_count_after}")
    
    return df


def convert_datetime_columns(df, datetime_columns=None):
    """
    Convert specified columns to datetime format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    datetime_columns : list of str, optional
        Column names to convert to datetime. If None, attempts to auto-detect
        columns with 'Date' or 'Time' in their names
        
    Returns
    -------
    pd.DataFrame
        Dataframe with datetime columns converted
        
    Examples
    --------
    >>> df = convert_datetime_columns(df, datetime_columns=['TransactionDate', 'PreviousTransactionDate'])
    >>> print(df.dtypes)
    """
    df = df.copy()
    
    # Auto-detect datetime columns if not provided
    if datetime_columns is None:
        datetime_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not datetime_columns:
        print("✓ No datetime columns to convert")
        return df
    
    converted_cols = []
    for col in datetime_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                converted_cols.append(col)
            except Exception as e:
                print(f"⚠ Warning: Could not convert {col}: {str(e)}")
        else:
            print(f"⚠ Warning: Column {col} not found in dataframe")
    
    print(f"✓ Converted {len(converted_cols)} columns to datetime: {converted_cols}")
    return df


def create_time_features(df, date_column='TransactionDate', reference_date_column='PreviousTransactionDate'):
    """
    Create time-based features from datetime columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must contain datetime columns)
    date_column : str, default='TransactionDate'
        Name of the current date column
    reference_date_column : str, default='PreviousTransactionDate'
        Name of the reference date column for time difference calculation
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new time-based features added
        
    Features created
    ----------------
    - TimeBetweenTransactions: Hours between current and previous transaction
    - TransactionHour: Hour of day when transaction occurred
    - TransactionDayOfWeek: Day of week (0=Monday, 6=Sunday)
    - TransactionMonth: Month of transaction
    
    Examples
    --------
    >>> df = create_time_features(df)
    >>> print(df[['TimeBetweenTransactions', 'TransactionHour']].head())
    """
    df = df.copy()
    
    # Validate that date columns exist and are datetime type
    if date_column not in df.columns:
        raise ValueError(f"Column {date_column} not found in dataframe")
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise ValueError(f"Column {date_column} is not datetime type")
    
    # Create TimeBetweenTransactions if reference column exists
    if reference_date_column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[reference_date_column]):
            raise ValueError(f"Column {reference_date_column} is not datetime type")
        
        df['TimeBetweenTransactions'] = (df[date_column] - df[reference_date_column]).dt.total_seconds() / 3600
    
    # Create time-based features from date column
    df['TransactionHour'] = df[date_column].dt.hour
    df['TransactionDayOfWeek'] = df[date_column].dt.dayofweek
    df['TransactionMonth'] = df[date_column].dt.month
    df['TransactionDayOfMonth'] = df[date_column].dt.day
    
    new_features = [
        'TransactionHour', 
        'TransactionDayOfWeek', 
        'TransactionMonth', 
        'TransactionDayOfMonth'
    ]
    
    if reference_date_column in df.columns:
        new_features.insert(0, 'TimeBetweenTransactions')
    
    print(f"✓ Created {len(new_features)} time-based features: {new_features}")
    return df


def preprocess_pipeline(filepath, datetime_cols=None, missing_strategy='drop'):
    """
    Execute the complete preprocessing pipeline.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the input CSV file
    datetime_cols : list of str, optional
        Columns to convert to datetime
    missing_strategy : str, default='drop'
        Strategy for handling missing values
        
    Returns
    -------
    pd.DataFrame
        Fully processed dataframe
        
    Examples
    --------
    >>> df_processed = preprocess_pipeline('data/transactions.csv')
    >>> print(df_processed.info())
    """
    print("="*80)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    print("\n[Step 1/4] Loading data...")
    df = load_data(filepath)
    
    # Step 2: Handle missing values
    print("\n[Step 2/4] Handling missing values...")
    df = clean_missing_values(df, strategy=missing_strategy)
    
    # Step 3: Convert datetime columns
    print("\n[Step 3/4] Converting datetime columns...")
    df = convert_datetime_columns(df, datetime_columns=datetime_cols)
    
    # Step 4: Create time features
    print("\n[Step 4/4] Creating time-based features...")
    df = create_time_features(df, date_column='TransactionDate', reference_date_column=None)
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    """
    Simple test of preprocessing functions
    """
    print("\n" + "="*80)
    print("TESTING DATA PREPROCESSING MODULE")
    print("="*80 + "\n")
    
    # Test with actual data file
    data_path = Path(__file__).parent.parent / "data" / "bank_transactions_data_2.csv"
    
    if data_path.exists():
        # Run full pipeline
        df_processed = preprocess_pipeline(
            filepath=data_path,
            datetime_cols=['TransactionDate', 'PreviousTransactionDate'],
            missing_strategy='drop'
        )
        
        # Display sample results
        print("\n" + "-"*80)
        print("Sample of processed data (first 5 rows):")
        print("-"*80)
        print(df_processed.head())
        
        print("\n" + "-"*80)
        print("Data types after preprocessing:")
        print("-"*80)
        print(df_processed.dtypes)
        
        print("\n" + "-"*80)
        print("New features created:")
        print("-"*80)
        time_features = ['TimeBetweenTransactions', 'TransactionHour', 'TransactionDayOfWeek', 'TransactionMonth']
        available_features = [f for f in time_features if f in df_processed.columns]
        for feat in available_features:
            print(f"  {feat}: range [{df_processed[feat].min():.2f}, {df_processed[feat].max():.2f}]")
        
    else:
        print(f"Warning: Data file not found at {data_path}")
        print("Skipping test execution")
