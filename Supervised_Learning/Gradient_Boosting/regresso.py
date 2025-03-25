import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# Data
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
Y = np.array([5, 7, 9, 11])

# Train Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1)
model.fit(X, Y)

# Predictions
preds = model.predict(X)

# Print results
for i in range(len(X)):
    print(f"X = {X[i][0]}, True Y = {Y[i]}, Predicted Y = {preds[i]:.2f}")

# Plot
plt.scatter(X, Y, color='red', label='Actual')
plt.plot(X, preds, color='blue', linestyle='dashed', marker='o', label='Predicted')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gradient Boosting Regression Example")
plt.show()
