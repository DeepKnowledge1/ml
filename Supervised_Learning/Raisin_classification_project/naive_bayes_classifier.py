from sklearn.naive_bayes import GaussianNB
from email_spam_classifer import BaseSpamClassifier


class NaiveBayesRaisinClassifier(BaseSpamClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="naive_bayes", **kwargs)
        self.model = GaussianNB(**kwargs)