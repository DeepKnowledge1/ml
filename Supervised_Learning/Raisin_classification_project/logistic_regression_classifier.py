from sklearn.linear_model import LogisticRegression
from email_spam_classifer import BaseSpamClassifier


class LogisticRegressionRaisinClassifier(BaseSpamClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="logistic_regression", **kwargs)
        self.model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)