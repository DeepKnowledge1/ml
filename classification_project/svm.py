import csv
import codecs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


CLASSIFIERS = {
    'decision_tree': DecisionTreeClassifier,
    'logistic_regression': LogisticRegression,
    'naive_bayes': GaussianNB,
    'svm': SVC
}


class RaisinClassifier:
    """Flexible Raisin Classifier with multiple algorithm support CLASSIFIERS"""
    # Dataset: https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification

    def __init__(self, classifier_type='logistic_regression', **kwargs):
        """
        Initialize classifier
        
        :param classifier_type: Type of classifier to use
        :param kwargs: Additional parameters for the classifier
        """
        if classifier_type not in CLASSIFIERS:
            raise ValueError(f"Unsupported classifier: {classifier_type}")
        
        # Default parameters
        default_params = {
            'decision_tree': {'random_state': 42},
            'logistic_regression': {'max_iter': 1000, 'random_state': 42},
            'naive_bayes': {},
            'svm': {'kernel': 'rbf', 'random_state': 42}
        }
        
        # Merge default and user-provided parameters
        params = {**default_params[classifier_type], **kwargs}
        
        # Create classifier
        self.model = CLASSIFIERS[classifier_type](**params)
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        
        # Store classifier type for tracking
        self.classifier_type = classifier_type

    def load_data(self, file_path):
        """
        Load and preprocess Raisin dataset
        
        :param file_path: Path to CSV file
        :return: Tuple of features and labels
        """
        features, labels = [], []
        
        # Open file with UTF-8-BOM encoding
        with codecs.open(file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract numerical features
                feature_row = [
                    float(row['Area']),
                    float(row['MajorAxisLength']),
                    float(row['MinorAxisLength']),
                    float(row['Eccentricity']),
                    float(row['ConvexArea']),
                    float(row['Extent']),
                    float(row['Perimeter'])
                ]
                features.append(feature_row)
                labels.append(row['Class'])
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        return X, y

    def train(self, X, y, test_size=0.2):
        """
        Train the model and evaluate performance
        
        :param X: Feature matrix
        :param y: Label vector
        :param test_size: Proportion of test set
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Predict
        self.predictions = self.model.predict(self.X_test)
        
        # Print evaluation metrics
        self.evaluate()
        
        return self

    def evaluate(self):
        """
        Print evaluation metrics
        """
        print(f"\n--- {self.classifier_type.replace('_', ' ').title()} Classifier ---")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, self.predictions))
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))

    def predict(self, sample):
        """
        Predict label for a single sample
        
        :param sample: Feature vector
        :return: Predicted label
        """
        # Scale the sample
        sample_scaled = self.scaler.transform([sample])
        
        # Predict
        prediction = self.model.predict(sample_scaled)[0]
        return prediction

def main():
    file_path = './data/Raisin_Dataset.csv'
    
    # Example classifiers to test
    classifiers = [
        'decision_tree',
        'logistic_regression',
        'naive_bayes',
        ('svm', {'kernel': 'rbf'})
    ]
    
    for clf_info in classifiers:
        # Handle different input formats
        if isinstance(clf_info, tuple):
            clf_type, params = clf_info
            classifier = RaisinClassifier(clf_type, **params)
        else:
            classifier = RaisinClassifier(clf_info)
        
        # Load and train
        X, y = classifier.load_data(file_path)
        classifier.train(X, y)
        
        # Example prediction (using first feature vector)
        sample = X[0]
        print(f"\nSample Prediction: {classifier.predict(sample)}")

if __name__ == "__main__":
    main()
