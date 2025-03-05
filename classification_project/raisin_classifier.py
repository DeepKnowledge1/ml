import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


class BaseRaisinClassifier:
    """Base Raisin Classifier"""

    def __init__(self, classifier_type, **kwargs):
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        X = df[
            [
                "Area",
                "MajorAxisLength",
                "MinorAxisLength",
                "Eccentricity",
                "ConvexArea",
                "Extent",
                "Perimeter",
            ]
        ].values
        y = df["Class"].values
        return X, y

    def train(self, X, y, test_size=0.2):
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
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
        return self.model.predict(self.scaler.transform([sample]))[0]