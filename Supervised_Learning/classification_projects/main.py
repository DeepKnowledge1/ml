from decision_tree_classifier import DecisionTree
from logistic_regression_classifier import LogisticRegressionClassifier
from naive_bayes_classifier import NaiveBayesClassifier
from svm_classifier import SVCClassifier
from DatasetClass import SpamDataHandler, RaisinDataHandler,PredictiveMaintenanceDataHandler
from sklearn.model_selection import train_test_split

from collections import Counter
def main():
    spamDataHandler = PredictiveMaintenanceDataHandler()
    file_path = "./data/predictive_maintenance.csv"
    X, y =spamDataHandler.load_data(file_path=file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    classes = list(Counter(y_test).keys())

    
    classifiers = [
        ("decision_tree", {}),
        ("logistic_regression", {}),
        # ("naive_bayes", {}),
        ("svm", {"kernel": "linear"}),
    ]

    for clf_type, params in classifiers:
        if clf_type == "decision_tree":
            classifier = DecisionTree(**params)
        elif clf_type == "logistic_regression":
            classifier = LogisticRegressionClassifier(**params)
        elif clf_type == "naive_bayes":
            classifier = NaiveBayesClassifier(**params)
        elif clf_type == "svm":
            classifier = SVCClassifier(**params)

        # X, y = classifier.load_data(file_path)
        classifier.train(X_train, y_train)

        predictions = classifier.predict(X_test)
        y_scores = classifier.predict_proba(X_test) # Probability of the positive class
        
        
        classifier.evaluate(y_test,predictions,y_scores=y_scores,pos_label=classes[1])
    stop = 1
        


if __name__ == "__main__":
    main()