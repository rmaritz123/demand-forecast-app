import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Demand Forecasting Data Uploader", layout="centered")

st.title("📈 Demand Forecasting Data Uploader & Validator")

st.markdown("""
**Instructions:**
- Upload your Excel or CSV file containing demand history.
- Your file should have columns:  
  - **Date** (any format, e.g. `dd-mm-yyyy`, `yyyy-mm-dd`, etc.)
  - **Product Code** (SKU)
  - **Demand** (can also be called Qty, Sales, or Volume)
- Minimum **12 time points per SKU** required.
- You can select the aggregation level (monthly or weekly) if your data allows.
- The app will preview your data and highlight any issues before you proceed.
""")

# --- File Upload ---
file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

if file:
    # --- Read file ---
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            # Try to find the first sheet with 'DemandHistory' in the name, else use first sheet
            xls = pd.ExcelFile(file)
            sheet_name = next((s for s in xls.sheet_names if "demandhistory" in s.replace(" ", "").lower()), xls.sheet_names[0])
            df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    # --- Column Detection ---
    col_map = {}
    # Date column (must be 'Date')
    if "Date" in df.columns:
        col_map["date"] = "Date"
    else:
        st.error("❌ Could not find a 'Date' column. Please ensure your file has a column named 'Date'.")
        st.stop()
    # SKU/Product Code column (must be 'Product Code')
    if "Product Code" in df.columns:
        col_map["sku"] = "Product Code"
    else:
        st.error("❌ Could not find a 'Product Code' column. Please ensure your file has a column named 'Product Code'.")
        st.stop()
    # Demand column (try to auto-detect, else let user pick)
    demand_candidates = [c for c in df.columns if c.strip().lower() in ["demand", "qty", "sales", "volume"]]
    if demand_candidates:
        col_map["demand"] = demand_candidates[0]
    else:
        st.warning("⚠️ Could not auto-detect the demand column. Please select it below.")
        demand_col = st.selectbox("Select the demand column", df.columns)
        col_map["demand"] = demand_col

    # --- Date Parsing ---
    date_sample = df[col_map["date"]].dropna().astype(str).iloc[0]
    date_formats = [
        "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%y", "%d/%m/%y", "%Y.%m.%d", "%d.%m.%Y"
    ]
    parsed = False
    for fmt in date_formats:
        try:
            pd.to_datetime(date_sample, format=fmt)
            date_format = fmt
            parsed = True
            break
        except Exception:
            continue
    if not parsed:
        st.warning(f"⚠️ Could not auto-detect date format. Example value: `{date_sample}`")
        date_format = st.text_input("Enter the date format (e.g. `%d-%m-%Y`):", value="%d-%m-%Y")
    # Try to parse all dates
    try:
        df["__parsed_date"] = pd.to_datetime(df[col_map["date"]], format=date_format, errors="raise")
    except Exception as e:
        st.error(f"❌ Date parsing failed: {e}. Please check your date format and data.")
        st.stop()

    # --- Demand Column Validation ---
    try:
        df["__demand"] = pd.to_numeric(df[col_map["demand"]], errors="raise")
    except Exception as e:
        st.error(f"❌ Demand column parsing failed: {e}. Please check your demand data.")
        st.stop()

    # --- Data Granularity Detection ---
    df = df.dropna(subset=["__parsed_date", col_map["sku"], "__demand"])
    df = df.sort_values(["Product Code", "__parsed_date"])
    date_diffs = df.groupby(col_map["sku"])["__parsed_date"].diff().dropna().dt.days
    most_common_diff = date_diffs.mode().iloc[0] if not date_diffs.empty else None
    if most_common_diff is not None:
        if 25 <= most_common_diff <= 35:
            granularity = "monthly"
        elif 6 <= most_common_diff <= 8:
            granularity = "weekly"
        elif 0 < most_common_diff <= 2:
            granularity = "daily"
        else:
            granularity = "unknown"
    else:
        granularity = "unknown"

    st.info(f"Detected data granularity: **{granularity.capitalize()}** (most common interval: {most_common_diff} days)" if most_common_diff else "Could not detect data granularity.")

    # --- Aggregation Option ---
    agg_level = "monthly"
    if granularity in ["daily", "weekly"]:
        agg_level = st.selectbox("Choose forecast aggregation level", ["monthly", "weekly"])
    elif granularity == "monthly":
        agg_level = "monthly"
    else:
        agg_level = st.selectbox("Choose forecast aggregation level", ["monthly", "weekly"])

    # --- Aggregate Data ---
    df["__month"] = df["__parsed_date"].dt.to_period("M").dt.to_timestamp()
    df["__week"] = df["__parsed_date"].dt.to_period("W").dt.start_time
    if agg_level == "monthly":
        group_col = "__month"
    else:
        group_col = "__week"

    # --- Preview and Validation ---
    st.subheader("Data Preview")
    st.dataframe(df[[col_map["sku"], col_map["date"], col_map["demand"], "__parsed_date"]].head(20))

    # Validation: missing values, negative/zero demand, min 12 points per SKU
    errors = []
    if df.isnull().any().any():
        errors.append("Some values are missing.")
    if (df["__demand"] <= 0).any():
        errors.append("Some demand values are zero or negative.")
    sku_counts = df.groupby(col_map["sku"])[group_col].nunique()
    skus_too_short = sku_counts[sku_counts < 12].index.tolist()
    if skus_too_short:
        errors.append(f"These SKUs have less than 12 data points: {', '.join(map(str, skus_too_short))}")

    if errors:
        st.error("❌ Data validation failed:\n- " + "\n- ".join(errors))
        st.stop()
    else:
        st.success("✅ Data validation passed! You can proceed to forecasting in the next step.")

    # --- SKU Selection with Search ---
    st.subheader("SKU Selection")
    sku_list = sorted(df[col_map["sku"]].unique())
    search = st.text_input("Search for a Product Code (SKU):")
    filtered_skus = [sku for sku in sku_list if search.lower() in str(sku).lower()]
    selected_sku = st.selectbox("Select SKU", filtered_skus if filtered_skus else sku_list)

    # --- Show Aggregated Data Preview for Selected SKU ---
    st.subheader(f"Aggregated Data Preview for {selected_sku}")
    sku_df = df[df[col_map["sku"]] == selected_sku]
    agg_df = sku_df.groupby(group_col)["__demand"].sum().reset_index()
    st.dataframe(agg_df.head(20))

    st.info("Ready for forecasting! (This is just the data handling/validation prototype.)")
