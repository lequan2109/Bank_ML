# -*- coding: utf-8 -*-
"""
Run full pipeline (no notebooks):
1) Preprocess transactions
2) Customer clustering (KMeans)
3) Fraud/anomaly detection (Isolation Forest)
4) Rule-based recommendations

Outputs:
- outputs/clusters.csv
- outputs/anomalies.csv
- outputs/recommendations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# allow imports like: from src.xxx import ...
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import aggregate_customer_features, normalize_features
from src.clustering import kmeans_clustering, assign_clusters
from src.fraud_detection import (
    prepare_anomaly_features,
    train_isolation_forest,
    detect_anomalies,
)
from src.recommendation import recommendation_engine


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_risk_level(scores: pd.Series) -> pd.Series:
    """
    Correct rule for Isolation Forest score_samples():
    - score càng thấp (càng âm) => càng bất thường => rủi ro cao hơn

    Risk buckets:
      score <= p25 -> High
      p25 < score <= p50 -> Medium
      > p50 -> Low
    """
    s = pd.to_numeric(scores, errors="coerce")
    q25 = s.quantile(0.25)
    q50 = s.quantile(0.50)

    def _cat(x):
        if pd.isna(x):
            return "Low"
        if x <= q25:
            return "High"
        if x <= q50:
            return "Medium"
        return "Low"

    return s.apply(_cat)


def main():
    parser = argparse.ArgumentParser(description="Run Bank ML pipeline and export outputs/*.csv")
    parser.add_argument("--input", default="bank_transactions_data_2.csv", help="Input transactions CSV")
    parser.add_argument("--outputs", default="outputs", help="Output folder")
    parser.add_argument("--missing_strategy", default="drop", choices=["drop", "mean", "forward_fill"])
    parser.add_argument("--k_clusters", type=int, default=3, help="KMeans clusters")
    parser.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination")
    parser.add_argument("--threshold_percentile", type=float, default=95, help="Anomaly threshold percentile (95 => top 5%)")
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve()
    outputs_dir = (PROJECT_ROOT / args.outputs).resolve()
    ensure_dir(outputs_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print("=" * 80)
    print("RUNNING BANK ML PIPELINE")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input        : {input_path}")
    print(f"Outputs      : {outputs_dir}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1) Preprocess
    # -------------------------------------------------------------------------
    print("\n--- [1/4] Preprocessing ---")
    raw_data = preprocess_pipeline(
        str(input_path),
        datetime_cols=["TransactionDate"],
        missing_strategy=args.missing_strategy,
    )

    # Basic column guards (avoid silent wrong outputs)
    must_have = ["TransactionID", "AccountID", "TransactionAmount", "AccountBalance", "LoginAttempts", "TransactionDuration"]
    missing = [c for c in must_have if c not in raw_data.columns]
    if missing:
        raise ValueError(f"Missing required columns in input after preprocessing: {missing}")

    # -------------------------------------------------------------------------
    # 2) Customer clustering
    # -------------------------------------------------------------------------
    print("\n--- [2/4] Customer clustering (KMeans) ---")
    customer_profiles = aggregate_customer_features(raw_data)

    # Use numeric columns only for clustering (exclude AccountID)
    feature_cols = [c for c in customer_profiles.columns if c != "AccountID" and pd.api.types.is_numeric_dtype(customer_profiles[c])]
    if not feature_cols:
        raise ValueError("No numeric features available for clustering.")

    prof_scaled, _scaler = normalize_features(customer_profiles, feature_columns=feature_cols, fit=True)
    X_cluster = prof_scaled[feature_cols].values

    km_res = kmeans_clustering(X_cluster, n_clusters=args.k_clusters, random_state=42, n_init=10)
    clusters_df = assign_clusters(customer_profiles, km_res["labels"], id_column="AccountID", cluster_column="ClusterID")

    clusters_out = outputs_dir / "clusters.csv"
    clusters_df.to_csv(clusters_out, index=False, encoding="utf-8-sig")
    print(f"✓ clusters.csv saved: {clusters_out} | rows={len(clusters_df)} | k={args.k_clusters} | silhouette={km_res['silhouette_score']:.4f}")

    # -------------------------------------------------------------------------
    # 3) Fraud / anomaly detection
    # -------------------------------------------------------------------------
    print("\n--- [3/4] Fraud / anomaly detection (Isolation Forest) ---")

    anomaly_features = ["TransactionAmount", "TransactionDuration", "LoginAttempts", "AccountBalance"]
    if "TimeBetweenTransactions" in raw_data.columns and raw_data["TimeBetweenTransactions"].notna().sum() > 0:
        anomaly_features.append("TimeBetweenTransactions")

    prepared = prepare_anomaly_features(raw_data, feature_list=anomaly_features)
    X_anom = prepared["X"]

    model_info = train_isolation_forest(X_anom, contamination=args.contamination, random_state=42)
    model = model_info["model"]

    det = detect_anomalies(model, X_anom, threshold_percentile=args.threshold_percentile)
    scores = det["scores"]  # score_samples (lower => more anomalous)
    is_anom = det["is_anomaly"]  # boolean mask by percentile threshold (stable rate)

    df_scored = raw_data.copy()
    df_scored["AnomalyScore"] = pd.to_numeric(scores, errors="coerce")
    df_scored["IsAnomaly"] = pd.Series(is_anom).astype(bool).values
    df_scored["RiskLevel"] = compute_risk_level(df_scored["AnomalyScore"])

    # Ensure key columns exist for Streamlit
    if "AccountID" not in df_scored.columns:
        # (Shouldn't happen because raw_data must have it)
        raise ValueError("AccountID missing in df_scored (unexpected).")

    anomalies_out = outputs_dir / "anomalies.csv"

    export_cols = [
        "TransactionID", "AccountID",
        "TransactionDate", "TransactionAmount", "TransactionType",
        "AccountBalance",
        "AnomalyScore", "RiskLevel", "IsAnomaly"
    ]
    export_cols = [c for c in export_cols if c in df_scored.columns]
    df_scored[export_cols].to_csv(anomalies_out, index=False, encoding="utf-8-sig")

    rate = df_scored["IsAnomaly"].mean() * 100
    print(f"✓ anomalies.csv saved: {anomalies_out} | rows={len(df_scored)} | anomaly_rate={rate:.2f}%")

    # -------------------------------------------------------------------------
    # 4) Recommendations (rule-based)
    # -------------------------------------------------------------------------
    print("\n--- [4/4] Recommendations (rule-based) ---")
    recs_df = recommendation_engine(clusters_df, df_scored, raw_data)

    recs_out = outputs_dir / "recommendations.csv"
    recs_df.to_csv(recs_out, index=False, encoding="utf-8-sig")
    print(f"✓ recommendations.csv saved: {recs_out} | rows={len(recs_df)}")

    print("\n🎉 DONE. Outputs generated:")
    print(f" - {clusters_out}")
    print(f" - {anomalies_out}")
    print(f" - {recs_out}")


if __name__ == "__main__":
    main()
