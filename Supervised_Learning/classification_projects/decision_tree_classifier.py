from sklearn.tree import DecisionTreeClassifier
from BaseClassifier import BaseClassifier


class DecisionTree(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="decision_tree", **kwargs)
        
        self.model = DecisionTreeClassifier(random_state=42, **kwargs)        