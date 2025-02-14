import numpy as np

# Dataset
X = np.array([65, 70, 72, 68, 75])  # Height
X_squared = X ** 2                  # Height squared
y = np.array([120, 150, 160, 130, 170])  # Weight

# Initialize parameters
beta = np.array([0.0, 0.0, 0.0])  # [beta0, beta1, beta2]

# Hyperparameters
learning_rate = 0.0001
iterations = 2

# Gradient Descent
for i in range(iterations):
    print(f"Iteration {i+1}:")

    # Compute predictions
    y_pred = beta[0] + beta[1] * X + beta[2] * X_squared
    print("Predictions:", y_pred)

    # Compute residuals
    residuals = y - y_pred
    print("Residuals:", residuals)

    # Compute gradients
    grad_beta0 = -np.mean(residuals)
    grad_beta1 = -np.mean(residuals * X)
    grad_beta2 = -np.mean(residuals * X_squared)
    gradients = np.array([grad_beta0, grad_beta1, grad_beta2])
    print("Gradients:", gradients)

    # Update parameters
    beta = beta - learning_rate * gradients
    print("Updated beta:", beta)
    print("-" * 50)

# Final coefficients
print("Final coefficients (beta):", beta)

# Make a prediction
def predict_weight(height):
    return beta[0] + beta[1] * height + beta[2] * (height ** 2)

# Example prediction
height = 70
predicted_weight = predict_weight(height)
print(f"Predicted weight for height={height} inches: {predicted_weight:.2f} lbs")