import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
import joblib
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.impute import KNNImputer

data_path = 'synthetic_weather_data.csv'
traindata = pd.read_csv(data_path)

print(traindata.head())
print(traindata.info())

traindata['Date'] = pd.to_datetime(traindata['Date'])
traindata['Day'] = traindata['Date'].dt.day
traindata['Weekday'] = traindata['Date'].dt.weekday
traindata['Month'] = traindata['Date'].dt.month
traindata['Year'] = traindata['Date'].dt.year

features = ['Hour', 'Day','Weekday' ,'Month', 'Year' , 'Temperature', 'Humidity', 'Wind Speed', 'rain/shower']
X = traindata[features]
y = traindata['Demand']
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100, 1000, 2000]}  
grid_search_ridge = GridSearchCV(ridge, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search_ridge.fit(X_train_poly, y_train)

best_ridge = grid_search_ridge.best_estimator_





y_pred_ridge = best_ridge.predict(X_test_poly)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mape_ridge = np.mean(np.abs((y_test - y_pred_ridge) / y_test)) * 100
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

joblib.dump(best_ridge, 'ridge_poly_model.pkl')
joblib.dump(poly, 'poly_transform.pkl')

print(f"Ridge - MAPE: {mape_ridge:.2f}%")
print(f"Ridge - Best alpha: {grid_search_ridge.best_params_['alpha']}")
print(f"Ridge - R²: {r2_ridge:.4f}")
print(f"Ridge - RMSE: {rmse_ridge:.4f}")
print(f"Ridge - MAE: {mae_ridge:.4f}")


xgb_model = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.1, max_depth=8, subsample=0.7, colsample_bytree=1)
xgb_param_grid = {
    'gamma': [0.3],
    'reg_alpha': [5, 10],
    'reg_lambda': [0.1]
}

xgb_grid_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=2, cv=5, scoring='r2', verbose=1, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)

best_xgb = xgb_grid_search.best_estimator_

y_pred_xgb = best_xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

joblib.dump(best_xgb, 'xgb_model.pkl')

print(f"XGBoost - MAPE: {mape_xgb:.2f}%")
print(f"XGBoost - Best Parameters: {xgb_grid_search.best_params_}")
print(f"XGBoost - R²: {r2_xgb:.4f}")
print(f"XGBoost - RMSE: {rmse_xgb:.4f}")
print(f"XGBoost - MAE: {mae_xgb:.4f}")

##rf_model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=2, min_samples_leaf=2, random_state=42)
##rf_param_grid = {
##    'min_samples_leaf': [1, 2]
##}

##rf_grid_search = RandomizedSearchCV(rf_model, rf_param_grid, n_iter=2, cv=5, scoring='r2', verbose=1, n_jobs=-1)
##rf_grid_search.fit(X_train, y_train)
##
##best_rf = rf_grid_search.best_estimator_
##
##y_pred_rf = best_rf.predict(X_test)
##r2_rf = r2_score(y_test, y_pred_rf)
##rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
##mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
##mae_rf = mean_absolute_error(y_test, y_pred_rf)
##
##joblib.dump(best_rf, 'rf_model.pkl')
##
##print(f"Random Forest - MAPE: {mape_rf:.2f}%")
##print(f"Random Forest - Best Parameters: {rf_grid_search.best_params_}")
##print(f"Random Forest - R²: {r2_rf:.4f}")
##print(f"Random Forest - RMSE: {rmse_rf:.4f}")
##print(f"Random Forest - MAE: {mae_rf:.4f}")


lgb_model = lgb.LGBMRegressor(n_estimators=1000)
lgb_param_grid = {
    'objective': ['regression'],
    'metric': ['rmse'],
    'boosting_type': ['gbdt'],
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],  
    'max_depth': [6, 8, 10],  
    'num_leaves': [15, 31, 40],  
    'min_data_in_leaf': [5, 10, 20],  
    'min_split_gain': [0.0, 0.001, 0.01],  
    'lambda_l1': [0.0, 0.1, 0.5],  
    'lambda_l2': [0.0, 0.1, 0.5],  
    'subsample': [0.7, 0.8, 0.9],  
    'colsample_bytree': [0.7, 0.8, 0.9],  
    'verbosity': [-1]  
} 

lgb_grid_search = RandomizedSearchCV(lgb_model, lgb_param_grid, n_iter=3, cv=5, scoring='r2', verbose=1, n_jobs=-1)
lgb_grid_search.fit(X_train_poly, y_train)

best_lgb = lgb_grid_search.best_estimator_

y_pred_lgb = best_lgb.predict(X_test_poly)
r2_lgb = r2_score(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
mape_lgb = np.mean(np.abs((y_test - y_pred_lgb) / y_test)) * 100
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)

joblib.dump(best_lgb, 'lgb_model.pkl')

print(f"LightGBM - MAPE: {mape_lgb:.2f}%")
print(f"LightGBM - Best Parameters: {lgb_grid_search.best_params_}")
print(f"LightGBM - R²: {r2_lgb:.4f}")
print(f"LightGBM - RMSE: {rmse_lgb:.4f}")
print(f"LightGBM - MAE: {mae_lgb:.4f}")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ridge, alpha=0.5)
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lgb, alpha=0.5)
plt.title("LGB vs Actual")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()

print("Models trained and saved successfully.")

