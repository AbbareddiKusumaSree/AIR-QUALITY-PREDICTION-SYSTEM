import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------
# Streamlit Settings
# -------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌍 Air Quality Explorer")

# -------------------
# Dataset Handling
# -------------------
dataset_folder = "data.csv"  # rename this folder to "data" ideally
all_datasets = [
    "city_day.csv",
    "city_hour.csv",
    "station_day.csv",
    "station_hour.csv"
]



all_dataset_paths = [os.path.join(dataset_folder, f) for f in all_datasets]


available_datasets = [f for f in all_dataset_paths if os.path.exists(f)]

if not available_datasets:
    st.error(f"No datasets found in '{dataset_folder}' folder!")
    st.stop()
    
    

dataset_choice = st.sidebar.selectbox("Dataset", available_datasets, index=0)


st.sidebar.markdown(f"**Selected Dataset:** {os.path.basename(dataset_choice)}")



@st.cache_data
def load_dataset(dataset):
    try:
        df = pd.read_csv(dataset, parse_dates=["Datetime"])
    except Exception as e:
        st.error(f"Error loading {dataset}: {e}")
        return pd.DataFrame(), []

    pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO',
                  'SO2','O3','Benzene','Toluene','Xylene','AQI']
    for p in pollutants:
        if p in df.columns:


            df[p] = pd.to_numeric(df[p], errors="coerce")
 
 
    available_pollutants = [p for p in pollutants if p in df.columns]
    return df, available_pollutants

df, pollutants = load_dataset(dataset_choice)

if df.empty:
    st.warning("Dataset is empty or could not be loaded.")
    st.stop()



location_col = "City" if os.path.basename(dataset_choice).startswith("city") else "Station"
locations = sorted(df[location_col].dropna().unique())

if not locations:
    st.warning("No locations found in dataset.")
    st.stop()

selected_location = st.sidebar.selectbox("🌍 Location", locations)



time_options = {"Last 24 Hours":24, "Last 7 Days":24*7, "Last 30 Days":24*30, "All":None}
time_choice = st.sidebar.selectbox("⏳ Time Range", list(time_options.keys()))


selected_pollutants = st.sidebar.multiselect(
    "☁️ Pollutants",
    pollutants,
    default=[pollutants[0]] if pollutants else []
)

# -------------------
# Data Filtering
# -------------------
df = df[df[location_col] == selected_location].copy()
df = df.set_index("Datetime").sort_index()

if time_choice != "All":
    hours = time_options[time_choice]
    cutoff = df.index.max() - pd.Timedelta(hours=hours)
    df = df[df.index >= cutoff]

# -------------------
# Visualization
# -------------------
col1, col2 = st.columns([2,1])

with col1:
    if selected_pollutants:
        st.subheader(f"📈 {', '.join(selected_pollutants)} Time Series")
        fig, ax = plt.subplots(figsize=(8,3))
        for p in selected_pollutants:
            if p in df:
                df[p].plot(ax=ax, label=p, alpha=0.8)
        ax.set_ylabel("Concentration (µg/m³)")
        ax.legend()
        st.pyplot(fig)

with col2:
    st.subheader("🔗 Pollutant Correlations")
    if len(selected_pollutants) > 1:
        corr = df[selected_pollutants].corr()
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select 2+ pollutants for correlation heatmap")

# -------------------
# Statistics
# -------------------
if selected_pollutants:
    pollutant = selected_pollutants[0]

    st.subheader("📊 Statistical Summary")
    stats = df[pollutant].describe()[["mean","50%","max","min","std","count"]]
    st.write(stats.rename({"50%":"median"}))



    st.subheader("📦 Distribution (Histogram)")
    fig, ax = plt.subplots(figsize=(6,4))
    df[pollutant].plot(kind="hist", bins=20, color="green", alpha=0.7, ax=ax)
    ax.set_xlabel(f"{pollutant} (µg/m³)")
    st.pyplot(fig)



    st.subheader("🛡️ Data Quality")
    completeness = 100 * df[pollutant].notna().mean()
    validity = 100 * (df[pollutant] >= 0).mean()
    st.write(f"✅ Completeness: {completeness:.1f}%")
    st.write(f"✅ Validity: {validity:.1f}%")

if __name__ == "__main__":
    # This block ensures Streamlit runs correctly
    st.write("✅ App initialized successfully")
