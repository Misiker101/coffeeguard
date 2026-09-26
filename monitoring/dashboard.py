"""
Streamlit dashboard for CoffeeGuard prediction monitoring.

Run locally:
    streamlit run monitoring/dashboard.py
"""

import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = "monitoring/predictions.db"

st.set_page_config(page_title="CoffeeGuard Monitoring", layout="wide")
st.title("CoffeeGuard — Production Monitoring")

try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY ts DESC", conn)
    conn.close()
except Exception:
    st.warning("No prediction log found yet. Serve some traffic to the API first.")
    st.stop()

if df.empty:
    st.info("No predictions logged yet.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Total predictions", len(df))
col2.metric("Avg confidence", f"{df['confidence'].mean():.2%}")
col3.metric("Avg latency (ms)", f"{df['latency_ms'].mean():.1f}")

st.subheader("Prediction volume by class")
st.bar_chart(df["predicted_class"].value_counts())

st.subheader("Confidence over time")
df["ts_dt"] = pd.to_datetime(df["ts"], unit="s")
st.line_chart(df.set_index("ts_dt")["confidence"])

st.subheader("Latency over time")
st.line_chart(df.set_index("ts_dt")["latency_ms"])

st.subheader("Raw log (latest 200)")
st.dataframe(df.head(200))
