from sklearn.naive_bayes import GaussianNB
from raisin_classifier import BaseRaisinClassifier


class NaiveBayesRaisinClassifier(BaseRaisinClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="naive_bayes", **kwargs)
        self.model = GaussianNB(**kwargs)