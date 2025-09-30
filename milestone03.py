# milestone2.py
# Train & Evaluate Models (ARIMA, Prophet, LSTM)
# Import Libraries

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib

# -----------------------------
# Load Preprocessed Dataset
# -----------------------------
df = pd.read_csv("cleaned_air_quality.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3"]
results = []

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
for pollutant in pollutants:
    print(f"\nTraining models for {pollutant}...")

    if pollutant not in df.columns:
        print(f"Skipping {pollutant}: not found in dataset.")
        continue

    series = df[pollutant].dropna()
    if len(series) < 50:
        print(f"Skipping {pollutant}: insufficient data.")
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
    print(f"Best model for {pollutant}: {best[1]} (RMSE={best[0]:.2f})")

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
print("\nModel Comparison Results:")
print(results_df)

results_df.to_csv("model_evaluation_results.csv", index=False)
print("Results saved to model_evaluation_results.csv")
