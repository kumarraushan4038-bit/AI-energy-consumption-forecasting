# ==========================================
# AI-Powered Energy Consumption Forecasting
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------
# 0. Create output folders
# -----------------------------
os.makedirs("outputs/charts", exist_ok=True)

# -----------------------------
# 1. Generate Synthetic Dataset
# -----------------------------
np.random.seed(42)

# Create hourly timestamps for 180 days
date_range = pd.date_range(start='2024-01-01', periods=180*24, freq='H')

df = pd.DataFrame()
df['datetime'] = date_range

# Extract time-based features
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Generate synthetic weather data
df['temperature'] = 25 + 10 * np.sin(2 * np.pi * df['hour'] / 24) + np.random.normal(0, 2, len(df))
df['humidity'] = 60 + 20 * np.cos(2 * np.pi * df['hour'] / 24) + np.random.normal(0, 5, len(df))

# Generate synthetic energy consumption
base_load = 100
hourly_pattern = 20 * np.sin(2 * np.pi * (df['hour'] - 8) / 24)
weekend_effect = -10 * df['is_weekend']
temperature_effect = 2.5 * np.abs(df['temperature'] - 22)
random_noise = np.random.normal(0, 5, len(df))

df['energy_consumption'] = (
    base_load + hourly_pattern + weekend_effect + temperature_effect + random_noise
)

# -----------------------------
# 2. Introduce Missing Values
# -----------------------------
for col in ['temperature', 'humidity']:
    missing_idx = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_idx, col] = np.nan

# -----------------------------
# 3. Data Cleaning
# -----------------------------
print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing values
df.fillna(method='ffill', inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -----------------------------
# 4. Feature Engineering
# -----------------------------
# Previous hour energy usage
df['lag_1'] = df['energy_consumption'].shift(1)

# Average of previous 3 hours
df['rolling_mean_3'] = df['energy_consumption'].shift(1).rolling(window=3).mean()

# Remove NaN rows created by lag/rolling
df.dropna(inplace=True)

# -----------------------------
# 5. Exploratory Data Analysis
# -----------------------------
print("\nDataset Preview:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

# Plot 1: Energy Consumption Over Time
plt.figure(figsize=(14, 5))
plt.plot(df['datetime'], df['energy_consumption'])
plt.title('Energy Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.tight_layout()
plt.savefig("outputs/charts/energy_consumption_over_time.png")
plt.show()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(columns=['datetime']).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("outputs/charts/correlation_heatmap.png")
plt.show()

# -----------------------------
# 6. Prepare Features and Target
# -----------------------------
features = [
    'hour',
    'day',
    'month',
    'day_of_week',
    'is_weekend',
    'temperature',
    'humidity',
    'lag_1',
    'rolling_mean_3'
]

target = 'energy_consumption'

X = df[features]
y = df[target]

# Time-based split (80% train, 20% test)
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df['datetime'].iloc[split_index:]

# -----------------------------
# 7. Build and Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 8. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 9. Model Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== Model Performance =====")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R² Score : {r2:.4f}")

# -----------------------------
# 10. Feature Importance
# -----------------------------
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig("outputs/charts/feature_importance.png")
plt.show()

# -----------------------------
# 11. Actual vs Predicted Plot
# -----------------------------
results = pd.DataFrame({
    'datetime': test_dates.values,
    'Actual': y_test.values,
    'Predicted': y_pred
})

plt.figure(figsize=(15, 6))
plt.plot(results['datetime'], results['Actual'], label='Actual', linewidth=2)
plt.plot(results['datetime'], results['Predicted'], label='Predicted', linewidth=2)
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/actual_vs_predicted.png")
plt.show()

# -----------------------------
# 12. Scatter Plot
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Actual vs Predicted Scatter Plot')
plt.tight_layout()
plt.savefig("outputs/charts/scatter_plot.png")
plt.show()

# -----------------------------
# 13. Save Output Files
# -----------------------------
results.to_csv("outputs/predictions.csv", index=False)
feature_importance.to_csv("outputs/feature_importance.csv", index=False)

print("\nFiles saved successfully:")
print("- outputs/predictions.csv")
print("- outputs/feature_importance.csv")
print("- outputs/charts/")
print("\nProject completed successfully!")

