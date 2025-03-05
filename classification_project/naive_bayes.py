import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class IrisNaiveBayes:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.labels = []
        self.model = GaussianNB()

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
        """Preprocesses the data by encoding labels."""
        self.data = np.array(self.data)
        
        # Encode labels into numerical format
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

    def train_model(self):
        """Trains the Naive Bayes classifier."""
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
        prediction = self.model.predict([sample])
        return prediction[0]

# Example usage
if __name__ == "__main__":
    # Assuming the CSV file is named 'iris.csv'
    file_path = './data/iris_data.csv'  # Update this path to where your CSV file is located
    
    # Initialize the Naive Bayes class
    naive_bayes_model = IrisNaiveBayes(file_path)
    
    # Load and preprocess data
    naive_bayes_model.load_data()
    naive_bayes_model.preprocess_data()
    
    # Train the model
    naive_bayes_model.train_model()
    
    # Predict a sample
    sample = [5.2, 3.8, 1.5, 0.3]  # Example sepal and petal measurements
    prediction = naive_bayes_model.predict(sample)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy') if hasattr(label_encoder, 'classes_') else np.unique(naive_bayes_model.labels)
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    print(f"Predicted Label: {predicted_label}")