import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Demand Forecasting App", layout="centered")

# --- Custom CSS for a clean look ---
st.markdown("""
    <style>
    .main {background-color: #f8fafc;}
    .block-container {padding-top: 2rem;}
    .stButton>button {background-color: #2563eb; color: white; border-radius: 6px;}
    .stDownloadButton>button {background-color: #059669; color: white; border-radius: 6px;}
    .stSelectbox>div>div>div>div {color: #2563eb;}
    .stTextInput>div>div>input {border-radius: 6px;}
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Demand Forecasting App")

st.markdown("""
Welcome!  
Upload your demand data, select a product, and get a 6-period forecast using several proven methods.  
**Simple, accurate, and ready for business.**

---
""")

with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    1. **Upload** your Excel or CSV file with columns:  
       - `Date` (any format)
       - `Product Code` (SKU)
       - `Demand` (or Qty, Sales, Volume)
    2. **Select** your product and aggregation level (weekly/monthly).
    3. The app runs several forecasting methods, compares their accuracy, and recommends the best one.
    4. **Download** your forecast and see clear, actionable results.
    """)

file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

def try_parse_date(series, user_fmt=None):
    if user_fmt:
        return pd.to_datetime(series, format=user_fmt, errors="coerce")
    fmts = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d-%m-%y", "%d/%m/%y", "%Y.%m.%d", "%d.%m.%Y"]
    for fmt in fmts:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0.8 * len(series):
            return parsed
    return pd.to_datetime(series, errors="coerce")

if file:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            xls = pd.ExcelFile(file)
            sheet_name = next((s for s in xls.sheet_names if "demandhistory" in s.replace(" ", "").lower()), xls.sheet_names[0])
            df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    # Column detection
    if "Date" not in df.columns or "Product Code" not in df.columns:
        st.error("❌ File must have 'Date' and 'Product Code' columns.")
        st.stop()
    demand_candidates = [c for c in df.columns if c.strip().lower() in ["demand", "qty", "sales", "volume"]]
    if demand_candidates:
        demand_col = demand_candidates[0]
    else:
        demand_col = st.selectbox("Select the demand column", df.columns)
    # Date parsing
    date_sample = df["Date"].dropna().astype(str).iloc[0]
    parsed_dates = try_parse_date(df["Date"])
    if parsed_dates.isna().sum() > 0.2 * len(parsed_dates):
        st.warning(f"⚠️ Could not auto-detect date format. Example: `{date_sample}`")
        user_fmt = st.text_input("Enter the date format (e.g. `%d-%m-%Y`):", value="%d-%m-%Y")
        parsed_dates = try_parse_date(df["Date"], user_fmt)
    df["__parsed_date"] = parsed_dates
    if df["__parsed_date"].isna().any():
        st.error("❌ Some dates could not be parsed. Please check your date format.")
        st.stop()
    # Demand column
    try:
        df["__demand"] = pd.to_numeric(df[demand_col], errors="raise")
    except Exception as e:
        st.error(f"❌ Demand column parsing failed: {e}")
        st.stop()
    # Clean and sort
    df = df.dropna(subset=["__parsed_date", "Product Code", "__demand"])
    df = df.sort_values(["Product Code", "__parsed_date"])
    # Granularity detection
    date_diffs = df.groupby("Product Code")["__parsed_date"].diff().dropna().dt.days
    most_common_diff = date_diffs.mode().iloc[0]
