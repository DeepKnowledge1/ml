from BaseClassifier import BaseClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier


class AdaBoost(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="AdaBoost", **kwargs)
        weak_classifier = DecisionTreeClassifier(max_depth=1)
        
        self.model = AdaBoostClassifier(estimator = weak_classifier,**kwargs)        