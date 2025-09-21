import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
crop_df = pd.read_csv(r"C:\Users\Ujjwal\Videos\crop yield\crop_yield.csv")
soil_df = pd.read_csv(r"C:\Users\Ujjwal\Videos\crop yield\state_soil_data.csv")
weather_df = pd.read_csv(r"C:\Users\Ujjwal\Videos\crop yield\state_weather_data_1997_2020.csv")
print("All datasets loaded successfully.")

merged_df = pd.merge(crop_df, soil_df, on='state', how='left')

final_df = pd.merge(merged_df, weather_df, on=['state', 'year'], how='left')

print("\n--- Shape of data after merging ---")
print(final_df.shape)
print("\n--- First 5 rows of the merged dataset ---")
print(final_df.head())
print("\n" + "="*50 + "\n")

print("--- Missing values before cleaning ---")
print(final_df.isnull().sum())

df_clean = final_df.dropna()

print("\n--- Shape of data after dropping missing rows ---")
print(df_clean.shape)

df_encoded = pd.get_dummies(df_clean, columns=['crop', 'season', 'state'], drop_first=True)

print("\n--- First 5 rows of the final preprocessed dataset ---")
print(df_encoded.head())
print("\n" + "="*50 + "\n")

X = df_encoded.drop('yield', axis=1)
y = df_encoded['yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("\n" + "="*50 + "\n")

model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")
print("\n" + "="*50 + "\n")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Model Performance Metrics on Real Data ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")
print("\n" + "="*50 + "\n")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

ax.set_xlabel('Actual Yields (from real data)', fontsize=12)
ax.set_ylabel('Predicted Yields', fontsize=12)
ax.set_title('Actual vs. Predicted Crop Yields on Real Data', fontsize=14, fontweight='bold')
ax.grid(True)
plt.show()

