from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
}

# Evaluate models using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: Mean Accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
