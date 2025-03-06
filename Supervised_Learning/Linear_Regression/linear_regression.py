import numpy as np

# Dataset
X = np.array([
    [1, 65, 20],  # Person 1
    [1, 70, 25],  # Person 2
    [1, 72, 30],  # Person 3
    [1, 68, 22],  # Person 4
    [1, 75, 35]   # Person 5
])  # Features (with intercept term)

y = np.array([120, 150, 160, 130, 170])  # Target (weight in lbs)

# Initialize parameters
beta = np.array([0, 0, 0])  # [beta0, beta1, beta2]

# Hyperparameters
learning_rate = 0.01
iterations = 50

# Gradient Descent
for i in range(iterations):
    print(f"Iteration {i+1}:")

    # Compute predictions
    y_pred = np.dot(X, beta)
    print("Predictions:", y_pred)

    # Compute residuals (y - y_pred)
    residuals = y - y_pred
    print("Residuals:", residuals)

    # Compute gradients
    gradients = -np.dot(X.T, residuals) / len(y)
    print("Gradients:", gradients)

    # Update parameters
    beta = beta - learning_rate * gradients
    print("Updated beta:", beta)
    print("-" * 50)

# Final coefficients
print("Final coefficients (beta):", beta)

# Make a prediction
def predict_weight(height, age):
    return beta[0] + beta[1] * height + beta[2] * age

# Example prediction
height = 70
age = 25
predicted_weight = predict_weight(height, age)
print(f"Predicted weight for height={height} inches and age={age} years: {predicted_weight:.2f} lbs")