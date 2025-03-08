from decision_tree_classifier import DecisionTreeRaisinClassifier
from logistic_regression_classifier import LogisticRegressionRaisinClassifier
from naive_bayes_classifier import NaiveBayesRaisinClassifier
from svm_classifier import SVCRaisinClassifier


def main():
    file_path = "./data/emails.csv"
    classifiers = [
        ("decision_tree", {}),
        ("logistic_regression", {}),
        # ("naive_bayes", {}),
        ("svm", {"kernel": "linear"}),
    ]

    for clf_type, params in classifiers:
        if clf_type == "decision_tree":
            classifier = DecisionTreeRaisinClassifier(**params)
        elif clf_type == "logistic_regression":
            classifier = LogisticRegressionRaisinClassifier(**params)
        elif clf_type == "naive_bayes":
            classifier = NaiveBayesRaisinClassifier(**params)
        elif clf_type == "svm":
            classifier = SVCRaisinClassifier(**params)

        X, y = classifier.load_data(file_path)
        classifier.train(X, y)


if __name__ == "__main__":
    main()