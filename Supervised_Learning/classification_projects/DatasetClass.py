from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
class SpamDataHandler:
    """Handles data loading and preprocessing for Spam Classification"""

    def __init__(self):
        self.vectorizer = CountVectorizer()

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        X = df["text"].values
        y = df["spam"].values
        y = ["spam" if x == 1 else "ham" for x in y]
        X = self.preprocess_data(X)
        return X, y

    def preprocess_data(self, X):
        return self.vectorizer.fit_transform(X)

class RaisinDataHandler:
    """Handles data loading and preprocessing for Raisin Classification"""

    def __init__(self):
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
        X= self.preprocess_data(X=X)
        return X, y

    def preprocess_data(self, X):
        return self.scaler.fit_transform(X)