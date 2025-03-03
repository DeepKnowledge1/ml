import numpy as np

# Dataset
X = np.array([[2, 60], [3, 75], [4, 85], [1, 50]])  # Features: Hours Studied, Previous Exam Score
y = np.array([0, 1, 1, 0])  # Labels: Pass (1) or Fail (0)

# Initialize parameters
w = np.zeros(X.shape[1])  # Weights for features
b = 0  # Bias term
learning_rate = 0.1
num_iterations = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    m = len(y)  # Number of training examples

    for iteration in range(num_iterations):
        # Compute predictions
        z = np.dot(X, w) + b
        h = sigmoid(z)

        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (h - y))  # Gradient for weights
        db = (1 / m) * np.sum(h - y)  # Gradient for bias

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Print loss every 100 iterations
        if iteration % 100 == 0:
            loss = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
            print(f"Iteration {iteration}, Loss: {loss}")

    return w, b

# Train the model
w, b = gradient_descent(X, y, w, b, learning_rate, num_iterations)

# Final weights and bias
print("\nFinal weights (w):", w)
print("Final bias (b):", b)

# Make predictions
def predict(X, w, b):
    z = np.dot(X, w) + b
    h = sigmoid(z)
    return (h >= 0.5).astype(int)

# Predict on training data
predictions = predict(X, w, b)
print("\nPredictions:", predictions)

# Test on new data
new_student = np.array([[5, 80]])  # Hours Studied: 5, Previous Exam Score: 80
prediction = predict(new_student, w, b)
print("\nPrediction for new student (Pass=1, Fail=0):", prediction[0])