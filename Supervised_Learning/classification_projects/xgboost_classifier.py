from xgboost import XGBClassifier
from BaseClassifier import BaseClassifier

class XGBoostClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="xgboost", **kwargs)
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
