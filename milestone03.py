import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
dataset_folder = "data.csv"  # Correct folder name
all_datasets = ["cleaned_air_quality.csv"]

# Build dataset paths
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
    return df.set_index("Datetime").sort_index(), available_pollutants

df, pollutants = load_dataset(dataset_choice)

if df.empty:
    st.warning("Dataset is empty or could not be loaded.")
    st.stop()

# -----------------------------
# Define Model Functions
# -----------------------------
def train_arima(series):
    """Train ARIMA model and forecast"""
    model = ARIMA(series, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return model_fit, forecast

def train_prophet(series):
    """Train Prophet model and forecast"""
    df_prophet = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast[["ds", "yhat"]]

def train_lstm(series):
    """Train LSTM model and forecast"""
    values = series.values.reshape(-1, 1)
    generator = TimeseriesGenerator(values, values, length=10, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(10, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(generator, epochs=5, verbose=0)

    # Forecast next 30 steps
    predictions = []
    current_batch = values[-10:].reshape((1, 10, 1))
    for _ in range(30):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

    return model, np.array(predictions).flatten()

# -----------------------------
# Train & Evaluate Models
# -----------------------------
results = []  # ✅ FIX: define before using

for pollutant in pollutants:
    st.write(f"### Training models for {pollutant}...")

    if pollutant not in df.columns:
        st.warning(f"Skipping {pollutant}: not found in dataset.")
        continue

    series = df[pollutant].dropna()
    if len(series) < 50:
        st.warning(f"Skipping {pollutant}: insufficient data.")
        continue

    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    # --- ARIMA ---
    arima_model, arima_forecast = train_arima(train)
    arima_rmse = np.sqrt(mean_squared_error(test[:30], arima_forecast))
    arima_mae = mean_absolute_error(test[:30], arima_forecast)

    # --- Prophet ---
    prophet_model, prophet_forecast = train_prophet(train)
    prophet_pred = prophet_forecast.tail(30)["yhat"].values
    prophet_rmse = np.sqrt(mean_squared_error(test[:30], prophet_pred))
    prophet_mae = mean_absolute_error(test[:30], prophet_pred)

    # --- LSTM ---
    lstm_model, lstm_forecast = train_lstm(train)
    lstm_rmse = np.sqrt(mean_squared_error(test[:30], lstm_forecast))
    lstm_mae = mean_absolute_error(test[:30], lstm_forecast)

    # Collect results
    results.append({
        "Pollutant": pollutant,
        "ARIMA_RMSE": arima_rmse, "ARIMA_MAE": arima_mae,
        "Prophet_RMSE": prophet_rmse, "Prophet_MAE": prophet_mae,
        "LSTM_RMSE": lstm_rmse, "LSTM_MAE": lstm_mae
    })

    # Save best model
    best = min(
        [(arima_rmse, "ARIMA"), (prophet_rmse, "Prophet"), (lstm_rmse, "LSTM")],
        key=lambda x: x[0]
    )
    st.success(f"Best model for {pollutant}: {best[1]} (RMSE={best[0]:.2f})")

    if best[1] == "ARIMA":
        joblib.dump(arima_model, f"{pollutant}_best_model.pkl")
    elif best[1] == "Prophet":
        joblib.dump(prophet_model, f"{pollutant}_best_model.pkl")
    else:
        lstm_model.save(f"{pollutant}_best_model.h5")

# -----------------------------
# Save Evaluation Results
# -----------------------------
results_df = pd.DataFrame(results)
st.write("## Model Comparison Results")
st.dataframe(results_df)

results_df.to_csv("model_evaluation_results.csv", index=False)
st.success("✅ Results saved to model_evaluation_results.csv")
