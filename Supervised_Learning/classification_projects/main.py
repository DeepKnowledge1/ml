from decision_tree_classifier import DecisionTree
from logistic_regression_classifier import LogisticRegressionClassifier
from naive_bayes_classifier import NaiveBayesClassifier
from svm_classifier import SVCClassifier
from DatasetClass import SpamDataHandler
from sklearn.model_selection import train_test_split


def main():
    spamDataHandler = SpamDataHandler()
    file_path = "./data/emails.csv"
    X, y =spamDataHandler.load_data(file_path=file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
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
        
        classifier.evaluate(y_test,predictions,y_scores=y_scores)
        


if __name__ == "__main__":
    main()