from decision_tree_classifier import DecisionTree
from logistic_regression_classifier import LogisticRegressionClassifier
from naive_bayes_classifier import NaiveBayesClassifier
from svm_classifier import SVCClassifier
from random_forest import RandomForest
from adaboost import AdaBoost
from xgboost_classifier import XGBoostClassifier
from DatasetClass import SpamDataHandler, RaisinDataHandler,PredictiveMaintenanceDataHandler
from sklearn.model_selection import train_test_split
from label_transformer  import LabelTransformer
from collections import Counter
def main():
    spamDataHandler = SpamDataHandler()
    file_path = "./data/emails.csv"
    X, y =spamDataHandler.load_data(file_path=file_path)

    label_transformer = LabelTransformer()
    y_encoded = label_transformer.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    
    classes = list(Counter(y_test).keys())

    
    classifiers = [
        ("decision_tree", {}),
        ("logistic_regression", {}),
        # ("naive_bayes", {}),
        ("svm", {"kernel": "linear"}),
        ("rf", {"n_estimators": 3, "random_state": 42}),       
        ("adaboost", {"n_estimators": 50,"learning_rate":1.0, "random_state": 42}),
        ("xgboost", {"max_depth": 1, "learning_rate": 0.1, "n_estimators": 1}) ,
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
        elif clf_type == "rf":
            classifier = RandomForest(**params)
        elif clf_type == "adaboost":
            classifier = AdaBoost(**params)        
        elif clf_type == "xgboost":
            classifier = XGBoostClassifier(**params)
            
        # X, y = classifier.load_data(file_path)
        classifier.train(X_train, y_train)

        predictions = classifier.predict(X_test)
        y_test_decoded = label_transformer.inverse_transform(y_test)
        predictions = label_transformer.inverse_transform(predictions)
        y_scores = classifier.predict_proba(X_test) # Probability of the positive class
        
        
        classifier.evaluate(y_test_decoded,predictions,y_scores=y_scores,pos_label=classes[1])
    stop = 1
        


if __name__ == "__main__":
    main()