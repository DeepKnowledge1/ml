import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# Data
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
Y = np.array([0, 0, 1, 1])  # Pass/Fail (Binary classification)

# Train Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=3, learning_rate=0.5, max_depth=1)
model.fit(X, Y)

# Predictions
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]  # Probability of passing

# Print results
for i in range(len(X)):
    print(f"Study Hours = {X[i][0]}, True Y = {Y[i]}, Predicted Y = {preds[i]}, Probability = {probs[i]:.2f}")

# Plot decision boundary
plt.scatter(X, Y, color='red', label='Actual')
plt.plot(X, probs, color='blue', linestyle='dashed', marker='o', label="Probability of Passing")
plt.axhline(y=0.5, color='green', linestyle='--', label="Decision Boundary (0.5)")
plt.legend()
plt.xlabel("Study Hours")
plt.ylabel("Pass Probability")
plt.title("Gradient Boosting Classification Example")
plt.show()
