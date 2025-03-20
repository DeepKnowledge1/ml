from BaseClassifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier


class RandomForest(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="RandomForest", **kwargs)
        
        self.model = RandomForestClassifier(**kwargs)        