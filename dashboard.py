# air_quality_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta

# -------------------
# Page Setup
# -------------------
st.set_page_config(page_title="🌍 Air Quality Forecasting Dashboard", layout="wide")
st.title("🌍 Air Quality Forecasting Dashboard ")

# -------------------
# Sidebar Controls
# -------------------
data_source = st.sidebar.radio("📊 Choose Data Source", ["Excel", "MySQL"])

# -------------------
# Pollutants List
# -------------------
ALL_POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

# -------------------
# Load Dataset
# -------------------
if data_source == "Excel":
    DATASET_FOLDER = "data.csv"
    all_datasets = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
    dataset_choice = st.sidebar.selectbox("📂 Select Dataset", all_datasets)
    
    @st.cache_data
    def load_dataset(filename):
        df = pd.read_csv(os.path.join(DATASET_FOLDER, filename), parse_dates=["Datetime"])
        df = df.set_index("Datetime").sort_index()
        return df
    
    df = load_dataset(dataset_choice)

elif data_source == "MySQL":
    st.sidebar.info("⚠️ MySQL connection demo - replace with your DB credentials")
    # Simulated MySQL data with multiple cities + pollutants
    dates = pd.date_range("2025-09-01", periods=72, freq="H")
    cities = ["Chennai", "Mumbai", "Kolkata", "Delhi", "Bangalore"]
    data = []
    for city in cities:
        for d in dates:
            row = {
                "Datetime": d,
                "City": city,
                "PM2.5": np.random.randint(20, 100),
                "PM10": np.random.randint(30, 120),
                "NO": np.random.randint(5, 30),
                "NO2": np.random.randint(10, 60),
                "NOx": np.random.randint(15, 70),
                "NH3": np.random.randint(1, 10),
                "CO": np.random.uniform(0.5, 2.5),
                "SO2": np.random.randint(5, 25),
                "O3": np.random.randint(15, 50),
                "Benzene": np.random.uniform(1, 5),
                "Toluene": np.random.uniform(2, 8),
                "Xylene": np.random.uniform(0.5, 3),
                "AQI": np.random.randint(50, 300),
                "AQI_Bucket": np.random.choice(["Good","Moderate","Poor","Very Poor","Severe"])
            }
            data.append(row)
    df = pd.DataFrame(data).set_index("Datetime")

# -------------------
# Filters: City + Date Range
# -------------------
if "City" in df.columns:
    cities = sorted(df["City"].dropna().unique())
    city_choice = st.sidebar.selectbox("🏙️ Select City", cities)
    df = df[df["City"] == city_choice]

date_range = st.sidebar.date_input("📅 Select Date Range", [df.index.min().date(), df.index.max().date()])
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]

time_range = st.sidebar.selectbox("⏳ Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All"])
forecast_horizon = st.sidebar.selectbox("🔮 Forecast Horizon", ["24 Hours", "3 Days", "7 Days"])

# -------------------
# Pollutant Selection
# -------------------
available_pollutants = [p for p in ALL_POLLUTANTS if p in df.columns]
pollutant_choices = st.sidebar.multiselect("☁️ Select Pollutants", available_pollutants, default=available_pollutants[:2])
admin_mode = st.sidebar.toggle("⚙️ Admin Mode")

if st.sidebar.button("🔄 Update Dashboard"):
    st.rerun()

# -------------------
# Show Filtered Data
# -------------------
st.subheader("📋 Filtered Data")
st.dataframe(df.head())

# -------------------
# Layout (2x2 grid)
# -------------------
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# --- Gauge: Current AQI ---
with col1:
    st.subheader("Current Air Quality")
    if "AQI" in df.columns:
        latest_aqi = df["AQI"].iloc[-1]
        def categorize_aqi(value):
            if value <= 50: return "Good", "green"
            elif value <= 100: return "Moderate", "yellow"
            elif value <= 150: return "Unhealthy (Sensitive)", "orange"
            elif value <= 200: return "Unhealthy", "red"
            elif value <= 300: return "Very Unhealthy", "purple"
            else: return "Hazardous", "maroon"
        category, color = categorize_aqi(latest_aqi)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_aqi,
            title={'text': f"AQI - {category}"},
            gauge={'axis': {'range': [0, 500]},
                   'bar': {'color': color}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- Forecast ---
with col2:
    st.subheader("Pollutant Forecast")
    fig = go.Figure()
    for pollutant in pollutant_choices:
        hist = df[pollutant].dropna().tail(24)
        forecast = hist + np.random.normal(0, 2, len(hist))  # dummy forecast
        fig.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines+markers",
                                 name=f"{pollutant} - Historical"))
        fig.add_trace(go.Scatter(x=hist.index, y=forecast, mode="lines+markers",
                                 name=f"{pollutant} - Forecast", line=dict(dash="dot")))
    st.plotly_chart(fig, use_container_width=True)

# --- Pollutant Trends ---
with col3:
    st.subheader(f"Pollution Trends - {city_choice if 'city_choice' in locals() else ''}")
    fig, ax = plt.subplots(figsize=(8,4))
    for p in pollutant_choices:
        df[p].dropna().resample("D").mean().plot(ax=ax, label=p)
    ax.legend()
    ax.set_ylabel("µg/m³ / ppm")
    st.pyplot(fig)

# --- Alerts ---
with col4:
    st.subheader("Alert Notifications")
    latest_aqi = df["AQI"].iloc[-1] if "AQI" in df.columns else 0
    if latest_aqi <= 50:
        st.success("✅ Good air quality today")
    elif latest_aqi <= 100:
        st.info("🟡 Moderate air quality expected")
    elif latest_aqi <= 200:
        st.warning("⚠️ Unhealthy air quality, take precautions")
    else:
        st.error("❌ Hazardous air quality, avoid outdoor activity")
    st.info("📢 Model update completed (demo).")

# -------------------
# Admin Panel
# -------------------
if admin_mode:
    st.subheader("⚙️ Admin Panel")
    uploaded_file = st.file_uploader("Upload new dataset", type=["csv"])
    if uploaded_file:
        st.write("Preview:", pd.read_csv(uploaded_file, nrows=5))
    st.button("🔄 Retrain Models")

# -------------------
# Additional Features
# -------------------
st.subheader("📊 Additional Insights")

# --- AQI Distribution ---
if "AQI_Bucket" in df.columns:
    st.write("### AQI Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="AQI_Bucket", order=["Good","Moderate","Poor","Very Poor","Severe"], palette="viridis", ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("AQI Category")
    st.pyplot(fig)

# --- Summary Statistics ---
st.write("### Summary Statistics")
if pollutant_choices:
    summary = df[["AQI"] + pollutant_choices].describe().T[["mean","min","max"]]
    st.dataframe(summary)

# --- Top Pollutant Contributor ---
if "AQI" in df.columns and pollutant_choices:
    latest_row = df.iloc[-1]
    dominant_pollutant = latest_row[pollutant_choices].idxmax()
    st.info(f"🌫️ Dominant Pollutant Right Now: **{dominant_pollutant}**")
