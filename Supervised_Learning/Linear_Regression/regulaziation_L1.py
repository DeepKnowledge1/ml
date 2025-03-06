

import numpy as np

# Dataset
X = np.array([
    [1, 65, 20],
    [1, 70, 25],
    [1, 72, 30],
    [1, 68, 22],
    [1, 75, 35]
])  # Features (with intercept term)
y = np.array([120, 150, 160, 130, 170])  # Target variable

# Initialize parameters
beta = np.array([0.0, 0.0, 0.0])  # [beta0, beta1, beta2]
lambda_ = 1  # Regularization parameter (lambda)
alpha = 0.01  # Learning rate
m = len(y)  # Number of training examples

# Gradient Descent with L1 Regularization
def gradient_descent(X, y, beta, alpha, lambda_, iterations):
    for iteration in range(iterations):
        # Compute predictions
        y_pred = X.dot(beta)

        # Compute gradients
        error = y_pred - y
        grad_beta0 = (1 / m) * np.sum(error)  # Gradient for beta0 (no regularization)
        grad_beta1 = (1 / m) * np.sum(error * X[:, 1]) + lambda_ * np.sign(beta[1])  # Gradient for beta1
        grad_beta2 = (1 / m) * np.sum(error * X[:, 2]) + lambda_ * np.sign(beta[2])  # Gradient for beta2

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




# from sklearn.linear_model import Lasso
# import numpy as np

# # Example data
# X = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([1, 2, 3])

# # Lasso Regression
# lasso = Lasso(alpha=0.1)  # alpha is the regularization parameter
# lasso.fit(X, y)

# print("Coefficients:", lasso.coef_)
