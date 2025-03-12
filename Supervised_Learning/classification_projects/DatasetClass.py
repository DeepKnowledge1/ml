from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

class BaseDataHandler(ABC):
    """Base class for data loading and preprocessing."""

    def load_data(self, file_path: str) -> Tuple[any, any]:
        """
        Load data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            Tuple[any, any]: A tuple containing the feature matrix (X) and target labels (y).
        """
        df = pd.read_csv(file_path)
        X, y = self._extract_features_and_labels(df)
        X = self.preprocess_data(X)
        return X, y

    @abstractmethod
    def _extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[any, any]:
        """
        Extract features and labels from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[any, any]: A tuple containing the feature matrix (X) and target labels (y).
        """
        pass

    @abstractmethod
    def preprocess_data(self, X: any) -> any:
        """
        Preprocess the feature matrix.

        Args:
            X (any): The input feature matrix.

        Returns:
            any: The transformed feature matrix.
        """
        pass

class SpamDataHandler(BaseDataHandler):
    """Handles data loading and preprocessing for Spam Classification."""

    def _extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[any, any]:
        """
        Extract text features and labels for spam classification.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[any, any]: A tuple containing the text features (X) and labels (y).
        """
        X = df["text"].values
        y = df["spam"].astype(str).replace({"1": "spam", "0": "ham"}).values
        return X, y

    def preprocess_data(self, X: any) -> any:
        """
        Preprocess the text data using CountVectorizer.

        Args:
            X (any): The input text data.

        Returns:
            any: The transformed feature matrix.
        """
        return CountVectorizer().fit_transform(X)

class RaisinDataHandler(BaseDataHandler):
    """Handles data loading and preprocessing for Raisin Classification."""

    def _extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[any, any]:
        """
        Extract numerical features and labels for raisin classification.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[any, any]: A tuple containing the numerical features (X) and labels (y).
        """
        features = [
            "Area", "MajorAxisLength", "MinorAxisLength",
            "Eccentricity", "ConvexArea", "Extent", "Perimeter"
        ]
        X = df[features].values
        y = df["Class"].values
        return X, y

    def preprocess_data(self, X: any) -> any:
        """
        Preprocess the numerical data using StandardScaler.

        Args:
            X (any): The input numerical data.

        Returns:
            any: The transformed feature matrix.
        """
        return StandardScaler().fit_transform(X)


class PredictiveMaintenanceDataHandler(BaseDataHandler):
    """Handles data loading and preprocessing for PredictiveMaintenanceDataHandler Classification."""

    def _extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[any, any]:
        """
        Extract numerical features and labels for PredictiveMaintenanceDataHandler classification.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[any, any]: A tuple containing the numerical features (X) and labels (y).
        """
        df = self.encode(df)
        X = df.drop('Failure Type', axis = 1)

        y = df["Failure Type"].values
        return X, y

    def encode(self,df):
        df = df.drop('UDI', axis = 1)
       
        categorical_cols = ["Type", "Product ID"]
        scaler = LabelEncoder()
        for cat in categorical_cols:        
            df[cat] = scaler.fit_transform(df[cat])            
        
            
        
        return df

            
            
        
    def preprocess_data(self, X: any) -> any:
        """
        Preprocess the numerical data using StandardScaler.

        Args:
            X (any): The input numerical data.

        Returns:
            any: The transformed feature matrix.
        """
        return StandardScaler().fit_transform(X)


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# class SpamDataHandler:
#     """Handles data loading and preprocessing for Spam Classification"""

#     def __init__(self):
#         self.vectorizer = CountVectorizer()

#     def load_data(self, file_path):
#         df = pd.read_csv(file_path)
#         X = df["text"].values
#         y = df["spam"].values
#         y = ["spam" if x == 1 else "ham" for x in y]
#         X = self.preprocess_data(X)
#         return X, y

#     def preprocess_data(self, X):
#         return self.vectorizer.fit_transform(X)

# class RaisinDataHandler:
#     """Handles data loading and preprocessing for Raisin Classification"""

#     def __init__(self):
#         self.scaler = StandardScaler()

#     def load_data(self, file_path):
#         df = pd.read_csv(file_path)
#         X = df[
#             [
#                 "Area",
#                 "MajorAxisLength",
#                 "MinorAxisLength",
#                 "Eccentricity",
#                 "ConvexArea",
#                 "Extent",
#                 "Perimeter",
#             ]
#         ].values
#         y = df["Class"].values
#         X= self.preprocess_data(X=X)
#         return X, y

#     def preprocess_data(self, X):
#         return self.scaler.fit_transform(X)