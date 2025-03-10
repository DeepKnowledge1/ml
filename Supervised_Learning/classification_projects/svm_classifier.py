from sklearn.svm import SVC
from BaseClass import BaseClassifier


class SVCClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(classifier_type="svm", **kwargs)
        
        # Default parameters for SVC
        default_params = {"kernel": "rbf", "random_state": 42}
        
        # Merge default parameters with user-provided parameters
        params = {**default_params, **kwargs}
        
        # Ensure 'kernel' is not duplicated
        self.model = SVC(**params)