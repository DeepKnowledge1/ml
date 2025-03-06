from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Actual and predicted values
y_true = np.array([120, 150, 160, 130, 170])
y_pred = np.array([125, 155, 165, 135, 175])

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)

# MAE
mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)

# R² Score
r2 = r2_score(y_true, y_pred)
print("R² Score:", r2)
