import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define dataset
df = pd.DataFrame({
    'MedInc': [5.0, 4.0, 8.0, 6.0],
    'HouseAge': [10, 20, 5, 30],
    'AveRooms': [6.0, 5.5, 7.0, 6.5],
    'AveOccup': [2.0, 2.5, 1.0, 3.0],
    'Latitude': [34.0, 36.0, 38.0, 33.0],
    'Price': [300, 280, 350, 320]
})

# Features and target
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude']]
y = df['Price']

# Create and fit XGBoost Regressor
model = XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=2, alpha=5, reg_lambda=1, verbosity=0)
model.fit(X, y)

# Predict
df['Prediction'] = model.predict(X)

# Evaluate
mse = mean_squared_error(y, df['Prediction'])
r2 = r2_score(y, df['Prediction'])

df_result = df.copy()
df_result['Residual'] = df_result['Price'] - df_result['Prediction']

df_result, mse, r2
