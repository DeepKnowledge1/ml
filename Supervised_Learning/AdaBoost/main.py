# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (Iris dataset in this example)
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a weak classifier (Decision Tree with max_depth=1)
weak_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost classifier
adaboost = AdaBoostClassifier(
    estimator=weak_classifier,
    n_estimators=50,  # Number of weak classifiers
    learning_rate=1.0,  # Learning rate
    random_state=42
)

# Train the AdaBoost model
adaboost.fit(X_train, y_train)

# Make predictions on the test set
y_pred = adaboost.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")