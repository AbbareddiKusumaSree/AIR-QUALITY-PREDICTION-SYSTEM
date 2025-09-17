# milestone1.py
# Milestone 1: Data Preprocessing & EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. Load All Datasets
# -----------------------------
folder = "data.csv"   # folder containing your datasets
all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

df_list = []
for file in all_files:
    path = os.path.join(folder, file)
    print(f"Loading {file} ...")
    
    # Peek at columns
    temp_df = pd.read_csv(path, nrows=5)  # only first few rows
    print("Columns found:", temp_df.columns.tolist())
    
    # Only process files that have a Datetime column
    if "Datetime" not in temp_df.columns:
        print(f"⏭ Skipping {file} (no Datetime column)")
        continue
    
    # Load with Datetime parsing
    temp_df = pd.read_csv(path, parse_dates=["Datetime"])
    df_list.append(temp_df)

# Merge all datasets
if not df_list:
    raise ValueError("No valid CSVs with Datetime column were found in the folder!")

df = pd.concat(df_list, ignore_index=True)
print("Combined dataset shape:", df.shape)

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
df = df.fillna(method="ffill").fillna(method="bfill")

# -----------------------------
# 3. Resample Data
# -----------------------------
df.set_index("Datetime", inplace=True)
df_resampled = df.resample("D").mean(numeric_only=True)
print("Resampled dataset shape:", df_resampled.shape)

# -----------------------------
# 4. Feature Engineering
# -----------------------------
df_resampled["day"] = df_resampled.index.day
df_resampled["month"] = df_resampled.index.month
df_resampled["year"] = df_resampled.index.year
df_resampled["season"] = pd.cut(
    df_resampled["month"],
    bins=[0, 2, 5, 8, 11],
    labels=["Winter", "Spring", "Summer", "Autumn"],
    include_lowest=True
)

# -----------------------------
# 5. Exploratory Data Analysis
# -----------------------------
pollutants = [col for col in ["PM2.5", "PM10", "NO2", "SO2", "O3"] if col in df_resampled.columns]

if pollutants:
    # Pollutant trends
    df_resampled[pollutants].plot(figsize=(12,6))
    plt.title("Pollutant Trends Over Time")
    plt.ylabel("Concentration (µg/m³)")
    plt.xlabel("Date")
    plt.legend()
    plt.savefig("eda_pollutant_trends.png")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df_resampled[pollutants].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Between Pollutants")
    plt.savefig("eda_correlation_heatmap.png")
    plt.show()

    # Seasonal variation (only if PM2.5 exists)
    if "PM2.5" in df_resampled.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(x="season", y="PM2.5", data=df_resampled)
        plt.title("Seasonal Variation in PM2.5")
        plt.savefig("eda_seasonal_pm25.png")
        plt.show()

# -----------------------------
# 6. Save Cleaned Dataset
# -----------------------------
df_resampled.to_csv("cleaned_air_quality.csv")
print("✅ Cleaned dataset saved as cleaned_air_quality.csv")
