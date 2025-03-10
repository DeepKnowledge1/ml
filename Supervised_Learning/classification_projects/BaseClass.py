from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class BaseClassifier:
    """Base Classifier for both Spam and Raisin Classification"""

    def __init__(self, classifier_type, **kwargs):
        self.classifier_type = classifier_type
        # self.model_kwargs = kwargs  # Store kwargs for later use


    def train(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        self.evaluate()

    def evaluate(self):
        print(f"\n--- {self.classifier_type.replace('_', ' ').title()} Classifier ---")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, self.predictions))
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))

    def predict(self, sample):
        return self.model.predict(sample)[0]

    def predict_proba(self, sample):
        return self.model.predict_proba(sample)[0][1]