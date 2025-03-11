from sklearn.naive_bayes import GaussianNB
from BaseClassifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="naive_bayes", **kwargs)
        self.model = GaussianNB(**kwargs)