import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, C=1.0, n_epochs=100):
        """
        Initialize the SVM model.
        :param learning_rate: Learning rate for gradient descent (default: 0.01)
        :param C: Regularization parameter (default: 1.0)
        :param n_epochs: Number of training epochs (default: 100)
        """
        self.learning_rate = learning_rate
        self.C = C
        self.n_epochs = n_epochs
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        """
        Train the SVM model using Stochastic Gradient Descent (SGD).
        :param X: Input features (numpy array of shape [n_samples, n_features])
        :param y: Target labels (numpy array of shape [n_samples])
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # Compute prediction
                y_pred = np.dot(self.w, X[i]) + self.b

                # Compute hinge loss
                if 1 - y[i] * y_pred > 0:
                    # Update gradients
                    dw = self.w - self.C * y[i] * X[i]
                    db = -self.C * y[i]
                else:
                    # No contribution to gradients
                    dw = self.w
                    db = 0

                # Update weights and bias
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            # Print progress (optional)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Weights: {self.w}, Bias: {self.b}")

    def predict(self, X):
        """
        Predict the class labels for input features.
        :param X: Input features (numpy array of shape [n_samples, n_features])
        :return: Predicted labels (numpy array of shape [n_samples])
        """
        y_pred = np.dot(X, self.w) + self.b
        return np.sign(y_pred)  # Return +1 or -1 based on the sign

    def score(self, X, y):
        """
        Compute the accuracy of the model.
        :param X: Input features (numpy array of shape [n_samples, n_features])
        :param y: True labels (numpy array of shape [n_samples])
        :return: Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  # Accuracy




# Training data
X = np.array([
    [3, 2],  # Email 1
    [1, 5],  # Email 2
    [4, 3],  # Email 3
    [2, 1]   # Email 4
])
y = np.array([1, -1, 1, -1])  # Labels (+1 for Spam, -1 for Not Spam)

# Initialize and train the SVM model
svm = SVM(learning_rate=0.01, C=1.0, n_epochs=100)
svm.fit(X, y)

# Make predictions
predictions = svm.predict(X)
print("Predictions:", predictions)

# Evaluate accuracy
accuracy = svm.score(X, y)
print("Accuracy:", accuracy)