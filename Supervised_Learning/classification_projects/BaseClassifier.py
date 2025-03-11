
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt


class BaseClassifier:
    """Base Classifier for both Spam and Raisin Classification"""

    def __init__(self, classifier_type, **kwargs):
        self.classifier_type = classifier_type
        # self.model_kwargs = kwargs  # Store kwargs for later use


    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self,y_test,predictions,y_scores):
        print(f"\n--- {self.classifier_type.replace('_', ' ').title()} Classifier ---")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label="spam")
        roc_auc = auc(fpr, tpr)

        # # Plot ROC Curve
        # plt.figure()
        # RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        # plt.title(f"ROC Curve for {self.classifier_type.replace('_', ' ').title()}")
        # plt.show()

        print(f"\nAUC: {roc_auc:.2f}")

    def predict(self, sample):
        return self.model.predict(sample)

    def predict_proba(self, sample):        
    
        # Calculate ROC Curve and AUC
        if hasattr(self.model, "predict_proba"):
            y_scores = self.model.predict_proba(sample)[:, 1]  # Probability of the positive class
        else:
            y_scores = self.model.decision_function(sample)  # Use decision function for SVM
        
        return y_scores
    