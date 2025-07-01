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
    agg_level = "monthly"
    if granularity in ["daily", "weekly"]:
        agg_level = st.selectbox("Choose forecast aggregation level", ["monthly", "weekly"])
    elif granularity == "monthly":
        agg_level = "monthly"
    else:
        agg_level = st.selectbox("Choose forecast aggregation level", ["monthly", "weekly"])
    # Aggregate
    df["__month"] = df["__parsed_date"].dt.to_period("M").dt.to_timestamp()
    df["__week"] = df["__parsed_date"].dt.to_period("W").dt.start_time
    group_col = "__month" if agg_level == "monthly" else "__week"
    # Validation
    errors = []
    if df.isnull().any().any():
        errors.append("Some values are missing.")
    if (df["__demand"] <= 0).any():
        errors.append("Some demand values are zero or negative.")
    sku_counts = df.groupby("Product Code")[group_col].nunique()
    skus_too_short = sku_counts[sku_counts < 12].index.tolist()
    if skus_too_short:
        errors.append(f"These SKUs have less than 12 data points: {', '.join(map(str, skus_too_short))}")
    if errors:
        st.error("❌ Data validation failed:\n- " + "\n- ".join(errors))
        st.stop()
    else:
        st.success("✅ Data validation passed! Proceed to forecasting below.")

    # SKU selection
    sku_list = sorted(df["Product Code"].unique())
    search = st.text_input("Search for a Product Code (SKU):")
    filtered_skus = [sku for sku in sku_list if search.lower() in str(sku).lower()]
    selected_sku = st.selectbox("Select SKU", filtered_skus if filtered_skus else sku_list)
    sku_df = df[df["Product Code"] == selected_sku]
    agg_df = sku_df.groupby(group_col)["__demand"].sum().reset_index()
    agg_df = agg_df.rename(columns={group_col: "Date", "__demand": "Demand"})
    st.subheader(f"Aggregated Data Preview for {selected_sku}")
    st.dataframe(agg_df.head(20))

    # Forecasting
    st.subheader("Forecasting Results")
    data = agg_df.copy()
    data = data.sort_values("Date")
    data = data.reset_index(drop=True)
    # Use last 3 periods as test, rest as train
    if len(data) < 15:
        st.warning("Not enough data for robust backtesting. Forecasts will use all available data.")
        train, test = data, pd.DataFrame()
    else:
        train, test = data.iloc[:-3], data.iloc[-3:]
    horizon = 6

    def simple_average(train, horizon):
        avg = train["Demand"].mean()
        return [avg] * horizon

    def moving_average(train, horizon, window=3):
        arr = list(train["Demand"])
        forecast = []
        for _ in range(horizon):
            avg = np.mean(arr[-window:])
            forecast.append(avg)
            arr.append(avg)
        return forecast

    def exp_smoothing(train, horizon, alpha=0.3):
        forecast = train["Demand"].iloc[0]
        for val in train["Demand"].iloc[1:]:
            forecast = alpha * val + (1 - alpha) * forecast
        return [forecast] * horizon

    def linear_trend(train, horizon):
        n = len(train)
        x = np.arange(1, n+1)
        y = np.array(train["Demand"])
        A = np.vstack([x, np.ones(n)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return [m * (n + i + 1) + c for i in range(horizon)]

    def seasonal_naive(train, horizon, season_length=12):
        if len(train) < season_length:
            season_length = max(1, len(train))
        last_season = train["Demand"].values[-season_length:]
        return [last_season[i % season_length] for i in range(horizon)]

    def holt_winters(train, horizon):
        if len(train) < 2:
            return [train["Demand"].mean()] * horizon
        try:
            model = ExponentialSmoothing(train["Demand"], trend="add", seasonal="add", seasonal_periods=12 if len(train) >= 24 else None)
            fit = model.fit()
            return fit.forecast(horizon)
        except Exception:
            return [train["Demand"].mean()] * horizon

    def prophet_forecast(train, horizon):
        dfp = train.rename(columns={"Date": "ds", "Demand": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=horizon, freq="M" if agg_level == "monthly" else "W")
        forecast = m.predict(future)
        return forecast["yhat"].iloc[-horizon:].values

    def calc_kpis(actual, forecast):
        n = min(len(actual), len(forecast))
        actual, forecast = np.array(actual[:n]), np.array(forecast[:n])
        mae = np.mean(np.abs(actual - forecast))
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100 if np.all(actual != 0) else np.nan
        return mae, rmse, mape

    methods = {
        "Simple Average": lambda tr, h: simple_average(tr, h),
        "Moving Average": lambda tr, h: moving_average(tr, h),
        "Exp. Smoothing": lambda tr, h: exp_smoothing(tr, h),
        "Linear Trend": lambda tr, h: linear_trend(tr, h),
        "Seasonal Naive": lambda tr, h: seasonal_naive(tr, h, season_length=12 if agg_level == "monthly" else 52),
        "Holt-Winters": lambda tr, h: holt_winters(tr, h),
        "Prophet": lambda tr, h: prophet_forecast(tr, h)
    }

    results = {}
    kpis = {}
    for name, func in methods.items():
        try:
            forecast = func(train, horizon)
            results[name] = forecast
            if not test.empty:
                test_forecast = func(train, len(test))
                kpis[name] = calc_kpis(test["Demand"], test_forecast)
            else:
                kpis[name] = (np.nan, np.nan, np.nan)
        except Exception as e:
            results[name] = [np.nan] * horizon
            kpis[name] = (np.nan, np.nan, np.nan)

    # Recommendation
    best_method = min(kpis, key=lambda m: kpis[m][2] if not np.isnan(kpis[m][2]) else np.inf)
    st.success(f"**Recommended method:** {best_method} (lowest MAPE: {kpis[best_method][2]:.2f}%)")

    # --- Conclusion & Recommendation ---
    st.markdown(f"""
    ---
    ### 📢 Conclusion & Recommendation
    For SKU **{selected_sku}**, the recommended forecasting method is **{best_method}** because it had the lowest error on your recent data.
    The forecast for the next 6 periods is shown below.  
    Use this forecast to guide your planning and inventory decisions.
    ---
    """)

    # Show table of forecasts
    st.markdown("### 6-Period Forecasts")
    forecast_df = pd.DataFrame({m: results[m] for m in methods})
    forecast_df.index = [f"Period {i+1}" for i in range(horizon)]
    st.dataframe(forecast_df)

    # Show KPIs
    st.markdown("### Model KPIs (on last 3 periods)")
    kpi_df = pd.DataFrame(kpis, index=["MAE", "RMSE", "MAPE"]).T
    st.dataframe(kpi_df.style.format("{:.2f}"))

    # --- Plot: Actual vs Forecast (Recommended Method) ---
    st.markdown("### 📊 Forecast Plot")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["Date"], data["Demand"], label="Actual", color="#2563eb", marker="o")
    last_date = data["Date"].iloc[-1]
    freq = "M" if agg_level == "monthly" else "W"
    future_dates = pd.date_range(last_date, periods=horizon+1, freq=freq)[1:]
    ax.plot(future_dates, results[best_method], label="Forecast", color="#f59e42", marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # --- Histogram of Demand ---
    st.markdown("### 📈 Demand Distribution (Histogram)")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.histplot(data["Demand"], bins=10, kde=True, color="#2563eb", ax=ax2)
    ax2.axvline(data["Demand"].mean(), color="#f59e42", linestyle="--", label="Mean")
    ax2.set_xlabel("Demand")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    st.pyplot(fig2)

    # --- Download ---
    csv = forecast_df.to_csv(index=True).encode()
    st.download_button("Download Forecasts as CSV", csv, "forecast_results.csv")

    st.info("Forecasts are for the next 6 periods (months or weeks, as selected). KPIs are calculated on the last 3 periods of your data.")

    # --- Explanations ---
    st.markdown("---")
    st.markdown("## ℹ️ Method & KPI Explanations")
    with st.expander("Forecasting Methods Explained"):
        st.markdown("""
        - **Simple Average:** Uses the average of all past values as the forecast.
        - **Moving Average:** Uses the average of the last 3 periods.
        - **Exponential Smoothing:** Gives more weight to recent data.
        - **Linear Trend:** Fits a straight line to the data.
        - **Seasonal Naive:** Uses the value from the same period last year/season.
        - **Holt-Winters:** Captures trend and seasonality in the data.
        - **Prophet:** Advanced model by Meta (Facebook) for trend and seasonality.
        """)
    with st.expander("KPI Explanations"):
        st.markdown("""
        - **MAE (Mean Absolute Error):** Average of absolute errors between forecast and actual.
        - **RMSE (Root Mean Squared Error):** Square root of the average squared errors.
        - **MAPE (Mean Absolute Percentage Error):** Average of absolute percentage errors.
        - **Lower values mean better accuracy.**
        """)
