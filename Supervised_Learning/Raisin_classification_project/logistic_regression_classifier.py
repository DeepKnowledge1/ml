from sklearn.linear_model import LogisticRegression
from raisin_classifier import BaseRaisinClassifier


class LogisticRegressionRaisinClassifier(BaseRaisinClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="logistic_regression", **kwargs)
        self.model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)