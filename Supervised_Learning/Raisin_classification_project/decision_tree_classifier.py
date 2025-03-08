from sklearn.tree import DecisionTreeClassifier
from email_spam_classifer import BaseSpamClassifier


class DecisionTreeRaisinClassifier(BaseSpamClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="decision_tree", **kwargs)
        self.model = DecisionTreeClassifier(random_state=42, **kwargs)