import numpy as np

# Dataset
X = np.array([
    [1, 65, 20],  # [Intercept, Height, Age]
    [1, 70, 25],
    [1, 72, 30],
    [1, 68, 22],
    [1, 75, 35]
])
y = np.array([120, 150, 160, 130, 170])  # Target variable (Weight)

# Initialize parameters
beta = np.array([0.0, 0.0, 0.0])  # [beta0, beta1, beta2]
lambda_ = 1  # Regularization parameter (lambda)
alpha = 0.01  # Learning rate
m = len(y)  # Number of training examples

# Gradient Descent with L2 Regularization
def gradient_descent(X, y, beta, alpha, lambda_, iterations):
    for iteration in range(iterations):
        # Compute predictions
        y_pred = X.dot(beta)  # y_pred = beta0 + beta1 * x1 + beta2 * x2

        # Compute gradients
        error = y_pred - y
        grad_beta0 = -(1 / m) * np.sum(error)  # Gradient for beta0 (no regularization)
        grad_beta1 = -(1 / m) * np.sum(error * X[:, 1]) + lambda_ * beta[1]  # Gradient for beta1
        grad_beta2 = -(1 / m) * np.sum(error * X[:, 2]) + lambda_ * beta[2]  # Gradient for beta2

        # Update parameters
        beta[0] -= alpha * grad_beta0
        beta[1] -= alpha * grad_beta1
        beta[2] -= alpha * grad_beta2

        # Print results for each iteration
        print(f"Iteration {iteration + 1}:")
        print(f"Beta: {beta}")
        print(f"Predictions: {y_pred}")
        print(f"Gradients: [beta0: {grad_beta0}, beta1: {grad_beta1}, beta2: {grad_beta2}]")
        print("-" * 50)

    return beta

# Run gradient descent for 2 iterations
beta = gradient_descent(X, y, beta, alpha, lambda_, iterations=2)

# Final results
print("Final Coefficients:")
print(f"beta0 (Intercept): {beta[0]}")
print(f"beta1 (Height): {beta[1]}")
print(f"beta2 (Age): {beta[2]}")



##########  sklearn.linear_model  #########################################################


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([
    [65, 20],  # [Height, Age]
    [70, 25],
    [72, 30],
    [68, 22],
    [75, 35]
])
y = np.array([120, 150, 160, 130, 170])  # Target variable (Weight)

# Initialize Ridge Regression model
lambda_ = 1  # Regularization parameter (alpha in scikit-learn)
ridge = Ridge(alpha=lambda_, fit_intercept=True)  # fit_intercept=True adds beta0

# Fit the model
ridge.fit(X, y)

# Get the coefficients
beta0 = ridge.intercept_  # Intercept (beta0)
beta1, beta2 = ridge.coef_  # Coefficients for Height (beta1) and Age (beta2)

# Make predictions
y_pred = ridge.predict(X)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Print results
print("Final Coefficients:")
print(f"beta0 (Intercept): {beta0}")
print(f"beta1 (Height): {beta1}")
print(f"beta2 (Age): {beta2}")
print("\nPredictions:", y_pred)
print("Mean Squared Error (MSE):", mse)