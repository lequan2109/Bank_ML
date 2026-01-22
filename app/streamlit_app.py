# -*- coding: utf-8 -*-
"""
Banking ML Project - Streamlit Demo Application (Ưu tiên 1 + 2 + 3)

ƯU TIÊN 1 (Stability):
- Check file tồn tại + schema tối thiểu
- Fallback an toàn nếu account không có cluster/recommendation/anomaly
- Không crash do .iloc[0]
- Tự merge AccountID vào anomalies nếu file outputs/anomalies.csv bị thiếu

ƯU TIÊN 2 (UX):
- Bộ lọc: Date range, TransactionType, RiskLevel, Min/Max amount, Toggle anomalies/all
- Charts: Timeline amount + highlight anomalies, Distribution AnomalyScore (customer vs global), Customer vs Cluster bar
- Download button: anomalies/profile/recommendation
- Feedback loop: "Not Fraud" => outputs/feedback.csv

ƯU TIÊN 3 (Model Layer & Explainability):
- Balance Trend (Linear Regression)
- Goal-based savings tab
- Deviation Explanation for anomalies
"""

from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# Optional
try:
    import plotly.express as px
except Exception:
    px = None


# =========================
# Page config
# =========================
st.set_page_config(page_title="Phân tích ML Ngân hàng", page_icon="🏦", layout="wide")


# =========================
# Helpers
# =========================
def to_bool_series(s: pd.Series) -> pd.Series:
    """Robust convert IsAnomaly to bool, support True/False, 0/1, 'true'/'false'."""
    if s is None:
        return pd.Series([], dtype=bool)

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).eq(1)

    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(["true", "1", "yes", "y", "t"])


def safe_get_row(df: pd.DataFrame, key_col: str, key_val):
    """Return first matching row as Series, or None."""
    if df is None or df.empty or key_col not in df.columns:
        return None
    m = df[key_col] == key_val
    if not m.any():
        return None
    return df.loc[m].iloc[0]


def safe_datetime_minmax(series: pd.Series):
    s = pd.to_datetime(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    return s.min(), s.max()


def risk_level_order(level: str) -> int:
    order = {"High": 0, "Medium": 1, "Low": 2}
    return order.get(str(level), 99)


def require_columns(df: pd.DataFrame, required: list, df_name: str) -> bool:
    if df is None:
        st.error(f"❌ `{df_name}` đang None.")
        return False
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ `{df_name}` thiếu cột bắt buộc: {missing}.")
        return False
    return True


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def append_feedback(base_path: Path, rows: list[dict]):
    """Append feedback rows to outputs/feedback.csv (create if not exists)."""
    out_path = base_path / "outputs" / "feedback.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fb_new = pd.DataFrame(rows)
    if out_path.exists():
        try:
            fb_old = pd.read_csv(out_path)
            fb = pd.concat([fb_old, fb_new], ignore_index=True)
        except Exception:
            fb = fb_new
    else:
        fb = fb_new
    fb.to_csv(out_path, index=False, encoding="utf-8-sig")


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df.columns = df.columns.astype(str).str.strip()
    return df

# --- New helpers for Priority 3 ---

def compute_balance_trend(tx_df: pd.DataFrame, account_id: str):
    """
    Return dict: slope_per_day, slope_per_month, r2, last_balance, forecast_30d, forecast_90d, model_df
    """
    if tx_df is None or tx_df.empty:
        return None

    need_cols = {"AccountID", "TransactionDate", "AccountBalance"}
    if not need_cols.issubset(tx_df.columns):
        return None

    df = tx_df.copy()
    df["AccountID"] = df["AccountID"].astype(str).str.strip()
    df = df[df["AccountID"] == str(account_id)].copy()

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["AccountBalance"] = pd.to_numeric(df["AccountBalance"], errors="coerce")

    df = df.dropna(subset=["TransactionDate", "AccountBalance"])
    if df.empty or df["TransactionDate"].nunique() < 2:
        return None

    # Use last balance per day (reduce noise)
    df = df.sort_values("TransactionDate")
    daily = df.groupby(df["TransactionDate"].dt.date, as_index=False).tail(1).copy()
    daily["Date"] = pd.to_datetime(daily["TransactionDate"].dt.date)

    t0 = daily["Date"].min()
    daily["t"] = (daily["Date"] - t0).dt.days.astype(int)

    X = daily[["t"]].values
    y = daily["AccountBalance"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))

    slope_per_day = float(model.coef_[0])
    slope_per_month = slope_per_day * 30.0

    last_date = daily["Date"].max()
    last_balance = float(daily.loc[daily["Date"] == last_date, "AccountBalance"].iloc[0])

    def pred(days_ahead: int) -> float:
        t_pred = int((last_date - t0).days + days_ahead)
        return float(model.predict(np.array([[t_pred]]))[0])

    return {
        "slope_per_day": slope_per_day,
        "slope_per_month": slope_per_month,
        "r2": r2,
        "last_balance": last_balance,
        "last_date": last_date,
        "forecast_30d": pred(30),
        "forecast_90d": pred(90),
        "daily_series": daily[["Date", "AccountBalance"]].rename(columns={"AccountBalance": "Balance"}),
    }

def deviation_explanation(tx_row: pd.Series, customer_hist: pd.DataFrame, feature_cols=None, top_k=3):
    """
    Return list of explanations like:
    [{"feature":"LoginAttempts","value":3,"baseline":1.1,"z":2.4,"reason":"cao bất thường"}, ...]
    """
    if feature_cols is None:
        feature_cols = ["TransactionAmount", "TransactionDuration", "LoginAttempts", "AccountBalance"]
        if "TimeBetweenTransactions" in customer_hist.columns:
            feature_cols.append("TimeBetweenTransactions")

    hist = customer_hist.copy()
    out = []
    for f in feature_cols:
        if f not in hist.columns or f not in tx_row.index:
            continue
        h = pd.to_numeric(hist[f], errors="coerce").dropna()
        x = pd.to_numeric(pd.Series([tx_row.get(f)]), errors="coerce").iloc[0]
        if h.empty or pd.isna(x):
            continue

        mu = float(h.mean())
        sd = float(h.std()) if float(h.std()) > 1e-9 else None

        if sd is None:
            z = 0.0
        else:
            z = float((x - mu) / sd)

        direction = "cao" if x > mu else "thấp"
        reason = f"{direction} bất thường so với thói quen (mean={mu:.2f}, std={sd if sd else 0:.2f})"
        out.append({"feature": f, "value": float(x), "baseline_mean": mu, "z": z, "reason": reason})

    # rank by |z|
    out = sorted(out, key=lambda d: abs(d["z"]), reverse=True)
    return out[:top_k]


# =========================
# Load data
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    base_path = Path(__file__).parent.parent

    paths = {
        "clusters": base_path / "outputs" / "clusters.csv",
        "anomalies": base_path / "outputs" / "anomalies.csv",
        "recommendations": base_path / "outputs" / "recommendations.csv",
        "transactions": base_path / "bank_transactions_data_2.csv",
    }

    # 1) Check files exist
    missing_files = [k for k, p in paths.items() if not p.exists()]
    if missing_files:
        st.error(
            "❌ Thiếu file dữ liệu: " + ", ".join(missing_files)
            + "\n\n➡️ Hãy đảm bảo các file tồn tại đúng đường dẫn trong project."
        )
        st.stop()

    # 2) Load
    clusters_df = pd.read_csv(paths["clusters"])
    anomalies_df = pd.read_csv(paths["anomalies"])
    recs_df = pd.read_csv(paths["recommendations"])
    tx_df = pd.read_csv(paths["transactions"])

    # 3) Strip column names (avoid hidden spaces/BOM)
    for df in (clusters_df, anomalies_df, recs_df, tx_df):
        df.columns = df.columns.astype(str).str.strip()

    # 4) Normalize key columns types early
    if "TransactionID" in anomalies_df.columns:
        anomalies_df["TransactionID"] = anomalies_df["TransactionID"].astype(str).str.strip()
    if "TransactionID" in tx_df.columns:
        tx_df["TransactionID"] = tx_df["TransactionID"].astype(str).str.strip()

    if "AccountID" in clusters_df.columns:
        clusters_df["AccountID"] = clusters_df["AccountID"].astype(str).str.strip()
    if "AccountID" in recs_df.columns:
        recs_df["AccountID"] = recs_df["AccountID"].astype(str).str.strip()
    if "AccountID" in tx_df.columns:
        tx_df["AccountID"] = tx_df["AccountID"].astype(str).str.strip()
    if "AccountID" in anomalies_df.columns:
        anomalies_df["AccountID"] = anomalies_df["AccountID"].astype(str).str.strip()

    # 5) Parse TransactionDate in transactions (source of truth)
    if "TransactionDate" in tx_df.columns:
        tx_df["TransactionDate"] = pd.to_datetime(tx_df["TransactionDate"], errors="coerce")

    # 6) IsAnomaly_bool
    if "IsAnomaly" in anomalies_df.columns:
        anomalies_df["IsAnomaly_bool"] = to_bool_series(anomalies_df["IsAnomaly"])
    else:
        anomalies_df["IsAnomaly_bool"] = False

    # 7) Patch AccountID if missing in anomalies.csv (merge from transactions)
    if "AccountID" not in anomalies_df.columns:
        if {"TransactionID", "AccountID"}.issubset(tx_df.columns) and "TransactionID" in anomalies_df.columns:
            anomalies_df = anomalies_df.merge(
                tx_df[["TransactionID", "AccountID"]].drop_duplicates("TransactionID"),
                on="TransactionID",
                how="left",
            )

    # 8) Merge context columns from transactions (prefer tx values)
    #    This prevents TransactionDate being string and fixes str vs Timestamp comparisons.
    context_cols = ["AccountID", "TransactionDate", "TransactionAmount", "TransactionType", "AccountBalance", "TransactionDuration", "LoginAttempts"]
    tx_keep = ["TransactionID"] + [c for c in context_cols if c in tx_df.columns]

    if "TransactionID" in anomalies_df.columns and "TransactionID" in tx_df.columns:
        merged = anomalies_df.merge(
            tx_df[tx_keep].drop_duplicates("TransactionID"),
            on="TransactionID",
            how="left",
            suffixes=("", "_tx"),
        )

        # Prefer transaction columns if present (_tx)
        for col in context_cols:
            tx_col = f"{col}_tx"
            if tx_col in merged.columns:
                # Use combine_first to fill missing values in original col with values from tx_col
                if col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[tx_col])
                else:
                    merged[col] = merged[tx_col]
                merged.drop(columns=[tx_col], inplace=True)

        anomalies_merged = merged
    else:
        anomalies_merged = anomalies_df.copy()

    # 9) Force correct dtypes in anomalies_merged
    # TransactionDate must be datetime
    if "TransactionDate" in anomalies_merged.columns:
        anomalies_merged["TransactionDate"] = pd.to_datetime(anomalies_merged["TransactionDate"], errors="coerce")
    else:
        anomalies_merged["TransactionDate"] = pd.NaT

    # numeric conversions
    for c in ["TransactionAmount", "AccountBalance", "AnomalyScore", "TransactionDuration", "LoginAttempts"]:
        if c in anomalies_merged.columns:
            anomalies_merged[c] = pd.to_numeric(anomalies_merged[c], errors="coerce")

    # Ensure AccountID exists
    if "AccountID" in anomalies_merged.columns:
        anomalies_merged["AccountID"] = anomalies_merged["AccountID"].astype(str).str.strip()

    return base_path, clusters_df, anomalies_merged, recs_df, tx_df


base_path, clusters_df, anomalies_df, recommendations_df, transactions_df = load_data()


# =========================
# Schema checks (stability)
# =========================
if not require_columns(clusters_df, ["AccountID"], "outputs/clusters.csv"):
    st.stop()

# For anomalies: require minimal fields; AccountID may be patched by merge above
if not require_columns(anomalies_df, ["TransactionID", "IsAnomaly_bool", "AnomalyScore"], "outputs/anomalies.csv"):
    st.stop()

if "AccountID" not in anomalies_df.columns:
    st.error(
        "❌ `outputs/anomalies.csv` thiếu `AccountID` và không thể merge từ `bank_transactions_data_2.csv`.\n"
        "➡️ Cần đảm bảo transactions có `AccountID` và anomalies có `TransactionID`."
    )
    st.stop()

if "RiskLevel" not in anomalies_df.columns:
    st.warning("⚠️ anomalies.csv chưa có cột RiskLevel. Bạn vẫn dùng được app, nhưng filter RiskLevel sẽ trống.")


# =========================
# Title
# =========================
st.title("🏦 Phân tích Hệ thống Tài chính ML")
st.caption(
    "Dashboard phân tích giao dịch ngân hàng: Phân khúc khách hàng (K-Means), "
    "Phát hiện bất thường (Isolation Forest), Gợi ý tiết kiệm (Rule-based)."
)


# =========================
# Sidebar filters
# =========================
st.sidebar.markdown("## 🎛️ Bộ lọc")

# Cluster filter
cluster_values = sorted(clusters_df["ClusterID"].dropna().unique().tolist()) if "ClusterID" in clusters_df.columns else []
cluster_filter = st.sidebar.multiselect("Nhóm (Cluster)", options=cluster_values, default=cluster_values)

overview_anom_only = st.sidebar.toggle(
    "Overview: chỉ tính IsAnomaly=True",
    value=True,
    help="Bật để KPI/trend/top risky chỉ tính giao dịch bất thường thật.",
)

# RiskLevel filter
risk_values = []
if "RiskLevel" in anomalies_df.columns:
    risk_values = sorted(anomalies_df["RiskLevel"].dropna().unique().tolist(), key=risk_level_order)
risk_filter = st.sidebar.multiselect("Mức rủi ro (RiskLevel)", options=risk_values, default=risk_values)

# TransactionType filter
type_values = []
if "TransactionType" in anomalies_df.columns:
    type_values = sorted(anomalies_df["TransactionType"].dropna().unique().tolist())
type_filter = st.sidebar.multiselect("Loại giao dịch (TransactionType)", options=type_values, default=type_values)

# Amount range filter
amount_range = None
if "TransactionAmount" in anomalies_df.columns:
    amt = pd.to_numeric(anomalies_df["TransactionAmount"], errors="coerce").dropna()
    if not amt.empty:
        amount_range = st.sidebar.slider(
            "Khoảng tiền giao dịch",
            min_value=float(amt.min()),
            max_value=float(amt.max()),
            value=(float(amt.min()), float(amt.max())),
            step=max((float(amt.max()) - float(amt.min())) / 100.0, 1.0),
        )

# Date range
min_dt, max_dt = safe_datetime_minmax(anomalies_df["TransactionDate"]) if "TransactionDate" in anomalies_df.columns else (None, None)
date_range = None
if min_dt is not None and max_dt is not None:
    date_range = st.sidebar.date_input(
        "Khoảng thời gian",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )

account_search = st.sidebar.text_input("Tìm AccountID", value="")


# =========================
# Apply filters
# =========================
filtered_clusters_df = clusters_df.copy()
if cluster_filter and "ClusterID" in filtered_clusters_df.columns:
    filtered_clusters_df = filtered_clusters_df[filtered_clusters_df["ClusterID"].isin(cluster_filter)]

filtered_anoms_df = anomalies_df.copy()

# apply cluster->account constraint
allowed_accounts = set(filtered_clusters_df["AccountID"].astype(str).unique().tolist())
filtered_anoms_df["AccountID"] = filtered_anoms_df["AccountID"].astype(str)
filtered_anoms_df = filtered_anoms_df[filtered_anoms_df["AccountID"].isin(allowed_accounts)]

# risk/type/amount/date filters
if risk_filter and "RiskLevel" in filtered_anoms_df.columns:
    filtered_anoms_df = filtered_anoms_df[filtered_anoms_df["RiskLevel"].isin(risk_filter)]

if type_filter and "TransactionType" in filtered_anoms_df.columns:
    filtered_anoms_df = filtered_anoms_df[filtered_anoms_df["TransactionType"].isin(type_filter)]

if amount_range and "TransactionAmount" in filtered_anoms_df.columns:
    amt = pd.to_numeric(filtered_anoms_df["TransactionAmount"], errors="coerce")
    filtered_anoms_df = filtered_anoms_df[amt.between(amount_range[0], amount_range[1], inclusive="both")]

if date_range and len(date_range) == 2 and "TransactionDate" in filtered_anoms_df.columns:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered_anoms_df = filtered_anoms_df[
        filtered_anoms_df["TransactionDate"].notna()
        & (filtered_anoms_df["TransactionDate"] >= start)
        & (filtered_anoms_df["TransactionDate"] <= end)
    ]

filtered_true_anoms_df = filtered_anoms_df[filtered_anoms_df["IsAnomaly_bool"]].copy()

# account list
account_ids = sorted(filtered_clusters_df["AccountID"].dropna().astype(str).unique().tolist())
if account_search.strip():
    key = account_search.strip().upper()
    account_ids = [a for a in account_ids if key in str(a).upper()]


# =========================
# Tabs
# =========================
tab_overview, tab_customer, tab_goal = st.tabs(["📊 Tổng quan", "👤 Khách hàng", "🎯 Mục tiêu tiết kiệm"])


# ==========================================================
# TAB OVERVIEW
# ==========================================================
with tab_overview:
    st.subheader("📊 Tổng quan hệ thống")
    overview_df = filtered_true_anoms_df if overview_anom_only else filtered_anoms_df

    c1, c2, c3, c4 = st.columns(4)
    total_customers = int(filtered_clusters_df["AccountID"].nunique()) if "AccountID" in filtered_clusters_df.columns else 0
    total_tx = int(transactions_df.shape[0]) if transactions_df is not None else 0
    total_rows = int(overview_df.shape[0]) if overview_df is not None else 0
    high_cnt = int((overview_df["RiskLevel"] == "High").sum()) if (not overview_df.empty and "RiskLevel" in overview_df.columns) else 0

    c1.metric("Tổng khách hàng", f"{total_customers}")
    c2.metric("Tổng giao dịch", f"{total_tx}")
    c3.metric("Số bản ghi (theo bộ lọc)", f"{total_rows}")
    c4.metric("Rủi ro cao", f"{high_cnt}")

    st.caption("Ghi chú: **AnomalyScore càng thấp (càng âm) ⇒ càng bất thường ⇒ rủi ro cao hơn.**")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Phân bố khách hàng theo Cluster")
        if {"ClusterID", "AccountID"}.issubset(filtered_clusters_df.columns):
            dist = (
                filtered_clusters_df.groupby("ClusterID")["AccountID"]
                .nunique().reset_index(name="CustomerCount")
                .sort_values("ClusterID")
            )
            st.bar_chart(dist.set_index("ClusterID")["CustomerCount"])
            st.dataframe(dist, use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Xu hướng bất thường theo tháng")
        if not overview_df.empty and "TransactionDate" in overview_df.columns and overview_df["TransactionDate"].notna().any():
            tmp = overview_df.copy()
            tmp["Month"] = tmp["TransactionDate"].dt.to_period("M").astype(str)
            if "RiskLevel" in tmp.columns:
                ts = tmp.groupby(["Month", "RiskLevel"]).size().reset_index(name="Count")
                pivot = ts.pivot(index="Month", columns="RiskLevel", values="Count").fillna(0).sort_index()
                st.line_chart(pivot)
            else:
                ts = tmp.groupby("Month").size().reset_index(name="Count").sort_values("Month")
                st.line_chart(ts.set_index("Month")["Count"])
        else:
            st.info("Không có TransactionDate để vẽ trend.")

    st.divider()

    st.markdown("### Top khách hàng rủi ro nhất (điểm TB thấp nhất)")
    if not overview_df.empty and {"AccountID", "AnomalyScore"}.issubset(overview_df.columns):
        agg = overview_df.groupby("AccountID").agg(
            AvgAnomalyScore=("AnomalyScore", "mean"),
            Count=("AnomalyScore", "count"),
        ).reset_index().sort_values("AvgAnomalyScore", ascending=True).head(10)

        if "RiskLevel" in overview_df.columns:
            high = overview_df[overview_df["RiskLevel"] == "High"].groupby("AccountID").size().reset_index(name="HighCount")
            agg = agg.merge(high, on="AccountID", how="left").fillna({"HighCount": 0})

        st.dataframe(agg, use_container_width=True, hide_index=True)
    else:
        st.info("Thiếu cột AccountID/AnomalyScore để tính top risky.")


# ==========================================================
# TAB CUSTOMER
# ==========================================================
with tab_customer:
    st.subheader("👤 Chi tiết khách hàng")

    if not account_ids:
        st.warning("Không có AccountID phù hợp với bộ lọc hiện tại.")
        st.stop()

    selected_account = st.selectbox("Chọn AccountID", options=account_ids, key="cust_acc")

    # Safe fetch
    customer_cluster = safe_get_row(clusters_df.astype({"AccountID": str}), "AccountID", str(selected_account))
    customer_rec = safe_get_row(recommendations_df.astype({"AccountID": str}), "AccountID", str(selected_account))

    cust_all = filtered_anoms_df[filtered_anoms_df["AccountID"] == str(selected_account)].copy()
    cust_true = cust_all[cust_all["IsAnomaly_bool"]].copy()

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cluster", f"{int(customer_cluster['ClusterID'])}" if (customer_cluster is not None and "ClusterID" in customer_cluster) else "N/A")
    k2.metric("Số dư TB", f"${float(customer_cluster['average_account_balance']):.2f}" if (customer_cluster is not None and "average_account_balance" in customer_cluster) else "N/A")
    k3.metric("Tần suất GD", f"{int(float(customer_cluster['transaction_frequency']))}" if (customer_cluster is not None and "transaction_frequency" in customer_cluster) else "N/A")
    k4.metric("Số GD nghi vấn", f"{len(cust_true)}")

    st.divider()

    # --- BALANCE TREND ---
    trend = compute_balance_trend(transactions_df, selected_account)
    st.markdown("### 📉 Xu hướng số dư (Linear Regression)")
    if trend is None:
        st.info("Không đủ dữ liệu số dư theo thời gian để fit trend (cần ít nhất 2 ngày có balance).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số dư gần nhất", f"${trend['last_balance']:,.2f}")
        c2.metric("Slope / tháng", f"{trend['slope_per_month']:+.2f}")
        c3.metric("Dự báo +30 ngày", f"${trend['forecast_30d']:,.2f}")
        c4.metric("R² (Độ tin cậy)", f"{trend['r2']:.2f}")

        if trend["slope_per_month"] < 0:
            st.warning("Xu hướng số dư đang GIẢM. Có dấu hiệu tiêu lạm vào vốn nếu không điều chỉnh.")
        else:
            st.success("Xu hướng số dư đang TĂNG hoặc ổn định. Tình hình tài chính tích cực.")

        s = trend["daily_series"].set_index("Date")["Balance"]
        st.line_chart(s)
    st.divider()


    # --- CHARTS ---
    st.markdown("### 📈 Biểu đồ (Khách hàng)")
    colA, colB = st.columns([1.4, 1])

    with colA:
        st.markdown("**Timeline giao dịch (Amount theo thời gian) + highlight anomalies**")
        if {"TransactionDate", "TransactionAmount"}.issubset(cust_all.columns) and cust_all["TransactionDate"].notna().any():
            tmp = cust_all.copy()
            tmp["TransactionAmount"] = pd.to_numeric(tmp["TransactionAmount"], errors="coerce")
            tmp = tmp.dropna(subset=["TransactionDate", "TransactionAmount"])

            if px is not None and not tmp.empty:
                fig = px.scatter(
                    tmp.sort_values("TransactionDate"),
                    x="TransactionDate",
                    y="TransactionAmount",
                    color="IsAnomaly_bool",
                    hover_data=[c for c in ["TransactionID", "RiskLevel", "TransactionType", "AnomalyScore"] if c in tmp.columns],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                if not tmp.empty:
                    st.line_chart(tmp.set_index("TransactionDate")["TransactionAmount"])
                st.caption("Plotly không khả dụng -> dùng chart cơ bản.")
        else:
            st.info("Thiếu TransactionDate/TransactionAmount để vẽ timeline.")

    with colB:
        st.markdown("**Phân phối AnomalyScore: Khách vs Toàn hệ thống**")
        if "AnomalyScore" in filtered_anoms_df.columns:
            global_scores = filtered_anoms_df["AnomalyScore"].dropna()
            cust_scores = cust_all["AnomalyScore"].dropna()

            if px is not None and not global_scores.empty and not cust_scores.empty:
                df_plot = pd.DataFrame({
                    "AnomalyScore": pd.concat([cust_scores, global_scores], ignore_index=True),
                    "Group": (["Customer"] * len(cust_scores)) + (["Global"] * len(global_scores))
                })
                fig2 = px.histogram(df_plot, x="AnomalyScore", color="Group", nbins=40, barmode="overlay")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write("Customer score count:", len(cust_scores), " | Global score count:", len(global_scores))
        else:
            st.info("Thiếu AnomalyScore để vẽ phân phối.")

    st.divider()

    st.markdown("### 📊 So sánh Khách vs Trung bình Cluster")
    if customer_cluster is not None and "ClusterID" in customer_cluster and "ClusterID" in clusters_df.columns:
        cid = int(customer_cluster["ClusterID"])
        peers = clusters_df[clusters_df["ClusterID"] == cid]

        metrics = [
            ("average_account_balance", "Avg Balance"),
            ("transaction_frequency", "Txn Frequency"),
            ("average_transaction_amount", "Avg Txn Amount"),
        ]
        rows = []
        for k, label in metrics:
            if k in clusters_df.columns and k in customer_cluster and not peers.empty:
                cust_val = pd.to_numeric(customer_cluster.get(k), errors='coerce')
                peer_mean = pd.to_numeric(peers[k], errors='coerce').mean()
                if not pd.isna(cust_val) and not pd.isna(peer_mean):
                    rows.append({"Metric": label, "Customer": cust_val, "ClusterMean": peer_mean})

        if rows:
            df_cmp = pd.DataFrame(rows)
            if px is not None:
                fig3 = px.bar(
                    df_cmp.melt(id_vars="Metric", var_name="Group", value_name="Value"),
                    x="Metric", y="Value", color="Group", barmode="group"
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)
        else:
            st.info("Thiếu feature để so sánh customer vs cluster.")
    else:
        st.info("Không đủ dữ liệu cluster để so sánh.")

    st.divider()

    st.markdown("### 🧾 Danh sách giao dịch (Drill-down)")
    show_only_anomalies = st.toggle("Chỉ hiển thị IsAnomaly=True", value=True, key="cust_anom_toggle")
    table_df = cust_true if show_only_anomalies else cust_all

    base_cols = ["TransactionID", "TransactionDate", "TransactionAmount", "TransactionType", "RiskLevel", "AnomalyScore", "IsAnomaly_bool"]
    cols_exist = [c for c in base_cols if c in table_df.columns]

    if show_only_anomalies and "AnomalyScore" in table_df.columns:
        table_df = table_df.sort_values("AnomalyScore", ascending=True)

    st.dataframe(table_df[cols_exist].head(200), use_container_width=True, hide_index=True)
    st.caption("Hiển thị tối đa 200 dòng. Dùng bộ lọc ở sidebar để thu hẹp dữ liệu.")
    st.divider()

    # --- DEVIATION EXPLANATION ---
    st.markdown("### 🔎 Giải thích vì sao bị cảnh báo (Deviation Explanation)")
    if cust_true.empty:
        st.info("Không có anomaly để giải thích.")
    else:
        pick_txid = st.selectbox("Chọn TransactionID để xem giải thích", options=cust_true["TransactionID"].astype(str).tolist())
        row = cust_true[cust_true["TransactionID"].astype(str) == str(pick_txid)].iloc[0]

        hist = transactions_df[transactions_df["AccountID"] == str(selected_account)].copy()
        expl = deviation_explanation(row, hist, top_k=3)

        if not expl:
            st.info("Không đủ dữ liệu để giải thích theo deviation.")
        else:
            st.caption("So sánh giao dịch này với hành vi lịch sử của chính khách hàng đó:")
            for e in expl:
                st.write(f"- **{e['feature']}** = `{e['value']:.2f}` → {e['reason']} (z-score = {e['z']:.2f})")
    st.divider()


    # Downloads
    st.markdown("### 📥 Tải dữ liệu")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇️ Tải anomalies của khách (CSV)",
            data=csv_bytes(cust_true if not cust_true.empty else cust_all),
            file_name=f"anomalies_{selected_account}.csv",
            mime="text/csv"
        )
    with d2:
        prof = pd.DataFrame([customer_cluster]) if customer_cluster is not None else pd.DataFrame()
        st.download_button(
            "⬇️ Tải profile (CSV)",
            data=csv_bytes(prof) if not prof.empty else b"",
            file_name=f"profile_{selected_account}.csv",
            mime="text/csv",
            disabled=prof.empty
        )
    with d3:
        rec = pd.DataFrame([customer_rec]) if customer_rec is not None else pd.DataFrame()
        st.download_button(
            "⬇️ Tải recommendation (CSV)",
            data=csv_bytes(rec) if not rec.empty else b"",
            file_name=f"recommendation_{selected_account}.csv",
            mime="text/csv",
            disabled=rec.empty
        )

    st.divider()

    # Feedback loop
    with st.expander("🧠 Feedback (Human-in-the-loop)", expanded=False):
        st.caption("Tick các giao dịch bất thường nhưng bạn cho rằng **không phải gian lận** để lưu phản hồi.")
        if cust_true.empty:
            st.info("Khách này không có giao dịch bất thường để feedback.")
        else:
            fb_cols = ["TransactionID", "TransactionDate", "TransactionAmount", "TransactionType", "RiskLevel", "AnomalyScore"]
            fb_cols = [c for c in fb_cols if c in cust_true.columns]
            fb_view = cust_true[fb_cols].copy().head(50)
            fb_view["NotFraud"] = False

            edited = st.data_editor(
                fb_view,
                use_container_width=True,
                hide_index=True,
                column_config={"NotFraud": st.column_config.CheckboxColumn("Không phải gian lận", default=False)},
                disabled=[c for c in fb_view.columns if c != "NotFraud"]
            )

            if st.button("💾 Lưu phản hồi"):
                picked = edited[edited["NotFraud"] == True]
                if picked.empty:
                    st.warning("Bạn chưa tick giao dịch nào.")
                else:
                    rows = [{
                        "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "AccountID": str(selected_account),
                        "TransactionID": str(r.get("TransactionID", "")),
                        "Reason": "User marked as Not Fraud"
                    } for _, r in picked.iterrows()]

                    append_feedback(base_path, rows)
                    st.success(f"Đã lưu {len(rows)} phản hồi vào outputs/feedback.csv")
    st.divider()

    # Recommendation
    st.markdown("### 💰 Gợi ý tiết kiệm (Rule-based)")
    if customer_rec is None:
        st.info("Không có recommendation cho khách hàng này.")
    else:
        prio = str(customer_rec.get("PriorityLevel", "MEDIUM")).upper()
        msg = str(customer_rec.get("RecommendationMessage", "")).strip()
        reason = str(customer_rec.get("RecommendationReason", "")).strip()
        cat = str(customer_rec.get("RecommendationCategory", "")).strip()
        try:
            savings = float(customer_rec.get("SavingsPotential", 0.0))
        except Exception:
            savings = 0.0

        if prio == "HIGH":
            st.error(f"**Ưu tiên: CAO** — {msg}")
        elif prio == "MEDIUM":
            st.warning(f"**Ưu tiên: TRUNG BÌNH** — {msg}")
        else:
            st.success(f"**Ưu tiên: THẤP** — {msg}")

        if cat: st.markdown(f"**Danh mục:** {cat}")
        if reason: st.markdown(f"**Lý do:** {reason}")
        if savings > 0:
            st.markdown("**Điểm tiềm năng tiết kiệm:**")
            st.progress(min(max(savings / 100.0, 0.0), 1.0))
            st.caption(f"Tiềm năng tiết kiệm: {savings:.2f}/100")


# ==========================================================
# TAB GOAL-BASED SAVINGS
# ==========================================================
with tab_goal:
    st.subheader("🎯 Tiết kiệm theo mục tiêu (Goal-based)")

    if not account_ids:
        st.warning("Không có AccountID phù hợp với bộ lọc sidebar.")
        st.stop()

    acc = st.selectbox("Chọn AccountID", options=account_ids, key="goal_acc")
    trend = compute_balance_trend(transactions_df, acc)

    c1, c2 = st.columns(2)
    with c1:
        goal_name = st.text_input("Tên mục tiêu", value="Mua Laptop Gaming")
        target_amount = st.number_input("Số tiền mục tiêu (VND)", min_value=0.0, value=40_000_000.0, step=1_000_000.0)
        deadline = st.date_input("Hạn chót", value=(datetime.now() + pd.Timedelta(days=365)).date())
    with c2:
        default_current = float(trend["last_balance"]) if trend else 0.0
        current_amount = st.number_input(
            "Số tiền đã có (mặc định lấy số dư gần nhất)",
            min_value=0.0,
            value=default_current,
            step=1_000_000.0
        )
        st.caption(f"Số dư gần nhất của tài khoản `{acc}` là `{default_current:,.0f}`.")

    st.divider()

    days_left = (pd.to_datetime(deadline) - pd.to_datetime(datetime.now().date())).days
    if days_left <= 0:
        st.error("Hạn chót phải ở trong tương lai.")
        st.stop()

    months_left = max(int(np.ceil(days_left / 30.0)), 1)
    gap = max(target_amount - current_amount, 0.0)
    need_per_month = gap / months_left

    st.markdown("#### Kế hoạch tiết kiệm")
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Số tiền còn thiếu", f"{gap:,.0f} VND")
    gc2.metric("Số tháng còn lại", f"{months_left}")
    gc3.metric("Cần tiết kiệm / tháng", f"{need_per_month:,.0f} VND")

    st.divider()
    st.markdown("#### Đánh giá tính khả thi")
    if trend:
        expected_monthly_change = trend["slope_per_month"]
        st.caption(f"Dựa trên phân tích, số dư của bạn đang thay đổi trung bình ≈ **{expected_monthly_change:+,.0f} VND/tháng** (R²={trend['r2']:.2f}).")

        if expected_monthly_change < need_per_month:
            st.warning(f"**KHÓ KHẢ THI.** Xu hướng hiện tại của bạn ({expected_monthly_change:+,.0f}) thấp hơn mức cần tiết kiệm ({need_per_month:,.0f}).")
            st.markdown("Gợi ý: Tăng thu nhập, giảm chi tiêu, hoặc kéo dài hạn chót.")
        else:
            st.success(f"**KHẢ THI.** Xu hướng hiện tại của bạn ({expected_monthly_change:+,.0f}) cao hơn mức cần tiết kiệm ({need_per_month:,.0f}).")
            st.balloons()
    else:
        st.info("Không có dữ liệu xu hướng số dư để đánh giá tính khả thi.")

    st.divider()
    st.markdown("#### Gợi ý hành động")
    st.info(
        "💡 **Mẹo:** Cắt giảm các khoản chi bất thường (anomalies) và các khoản chi lớn không cần thiết. "
        "Thiết lập lệnh chuyển tiền tự động vào tài khoản tiết kiệm vào đầu mỗi tháng."
    )


st.markdown(
    """
---
**Gợi ý nếu thiếu output:**
- Regenerate bằng pipeline/notebook:
  - `notebooks/03_customer_clustering.ipynb` (clusters)
  - `notebooks/04_fraud_detection_fixed.ipynb` (anomalies)
  - `notebooks/05_saving_recommendation.ipynb` (recommendations)
"""
)
