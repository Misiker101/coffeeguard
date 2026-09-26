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


