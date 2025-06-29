import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

data = pd.read_csv('synthetic_weather_data.csv')


data['date'] = pd.to_datetime(data['Date'], format='mixed')
data['hour'] = data['Hour'].astype(int)
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['week'] = data['date'].dt.isocalendar().week
data['Year'] = data['date'].dt.year
data['dayofweek'] = data['date'].dt.dayofweek


features = ['day', 'hour' , 'month', 'Year']
X = data[features]
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

y_temperature =(data['Temperature'])
y_wind_speed = data['Wind Speed']
y_humidity = data['Humidity']
y_rain_shower = data['rain/shower'] 


X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temperature, test_size=0.2, random_state=42)
_, _, y_wind_train, y_wind_test = train_test_split(X, y_wind_speed, test_size=0.2, random_state=42)
_, _, y_hum_train, y_hum_test = train_test_split(X, y_humidity, test_size=0.2, random_state=42)
_, _, y_rain_train, y_rain_test = train_test_split(X, y_rain_shower, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def grid_search_rf(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=2, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
    param_grid = {
    'n_estimators': [10, 5],
    'max_depth': [6, None ],  # Prevent overfitting
    'min_samples_leaf': [5, 2],  # Prevents small leaves
    'min_samples_split': [2, 5]
} 
    grid_search = GridSearchCV(rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for the model: {grid_search.best_params_}")
    return grid_search.best_estimator_


print("Models saved successfully.")
temp_model = grid_search_rf(X_train_scaled, y_temp_train)
print("Models saved successfully.")
wind_model = grid_search_rf(X_train_scaled, y_wind_train)
print("Models saved successfully.")
hum_model = grid_search_rf(X_train_scaled, y_hum_train)
print("Models saved successfully.")
rain_model = grid_search_rf(X_train_scaled, y_rain_train)
print("Models saved successfully.")


y_temp_pred = temp_model.predict(X_test_scaled)
y_wind_pred = wind_model.predict(X_test_scaled)
y_hum_pred = hum_model.predict(X_test_scaled)
y_rain_pred = rain_model.predict(X_test_scaled)


print("Temperature Model - Mean Squared Error:", mean_squared_error(y_temp_test, y_temp_pred))
print("Wind Speed Model - Mean Squared Error:", mean_squared_error(y_wind_test, y_wind_pred))
print("Humidity Model - Mean Squared Error:", mean_squared_error(y_hum_test, y_hum_pred))
print("Rain/Shower Model - Mean Squared Error:", mean_squared_error(y_rain_test, y_rain_pred))


joblib.dump(temp_model, 'temperature_model.pkl')
joblib.dump(wind_model, 'wind_speed_model.pkl')
joblib.dump(hum_model, 'humidity_model.pkl')
joblib.dump(rain_model, 'rain_shower_model.pkl')
joblib.dump(scaler, 'scaler_wea.pkl')

print("Models saved successfully.")

def evaluate_model(y_test, y_pred, model_name):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    return r2, rmse


print("\nModel Performance Metrics:")
evaluate_model(y_temp_test, y_temp_pred, "Temperature Model")
evaluate_model(y_wind_test, y_wind_pred, "Wind Speed Model")
evaluate_model(y_hum_test, y_hum_pred, "Humidity Model")
evaluate_model(y_rain_test, y_rain_pred, "Rain/Shower Model")


def plot_predictions_vs_actual(y_test, y_pred, model_name, ax):
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(f"{model_name} Predictions vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_predictions_vs_actual(y_temp_test, y_temp_pred, "Temperature Model", axes[0, 0])
plot_predictions_vs_actual(y_wind_test, y_wind_pred, "Wind Speed Model", axes[0, 1])
plot_predictions_vs_actual(y_hum_test, y_hum_pred, "Humidity Model", axes[1, 0])
plot_predictions_vs_actual(y_rain_test, y_rain_pred, "Rain/Shower Model", axes[1, 1])

plt.tight_layout()
plt.show()

print("Training data shape:", X_train_scaled.shape)

input("Press any key to exit...")
