import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class IrisDecisionTree:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.labels = []
        self.model = DecisionTreeClassifier()

    def load_data(self):
        """Loads data from a CSV file."""
        with open(self.file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert numerical values to float and append to data
                self.data.append([
                    float(row['sepal length']),
                    float(row['sepal width']),
                    float(row['petal length']),
                    float(row['petal width'])
                ])
                # Append label to labels list
                self.labels.append(row['label'])

    def preprocess_data(self):
        """Converts labels to numerical values if necessary."""
        # In this case, we can directly use labels as strings since scikit-learn handles them.
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def train_model(self):
        """Trains the decision tree classifier."""
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

    def predict(self, sample):
        """Predicts the label for a given sample."""
        return self.model.predict([sample])

# Example usage
if __name__ == "__main__":
    # Assuming the CSV file is named 'iris_data.csv'
    file_path = './data/iris_data.csv'  # Update this path to where your CSV file is located
    
    # Initialize the Decision Tree class
    decision_tree = IrisDecisionTree(file_path)
    
    # Load and preprocess data
    decision_tree.load_data()
    decision_tree.preprocess_data()
    
    # Train the model
    decision_tree.train_model()
    
    # Predict a sample
    sample = [5.2, 3.8, 1.5, 0.3]  # Example sepal and petal measurements
    prediction = decision_tree.predict(sample)
    print(f"Predicted Label: {prediction[0]}")