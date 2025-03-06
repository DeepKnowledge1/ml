from sklearn.tree import DecisionTreeClassifier
from raisin_classifier import BaseRaisinClassifier


class DecisionTreeRaisinClassifier(BaseRaisinClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="decision_tree", **kwargs)
        self.model = DecisionTreeClassifier(random_state=42, **kwargs)