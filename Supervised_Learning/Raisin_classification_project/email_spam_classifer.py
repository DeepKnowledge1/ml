import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import csv


class BaseSpamClassifier:
    """Base Spam Email Classifier"""

    def __init__(self, classifier_type, **kwargs):
        self.classifier_type = classifier_type
        self.vectorizer = CountVectorizer()

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        X = df["text"].values
        y = df["spam"].values
        y = ["spam" if x == 1 else "ham" for x in y]

        return X, y

    def train(self, X, y, test_size=0.2):
        # Transform text data using CountVectorizer
        X_vectorized = self.vectorizer.fit_transform(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_vectorized, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        self.evaluate()
        
        # Print class probabilities
        # spam_probability = np.mean(y)
        spam_probability = sum(1 for x in y if x == 'spam') / len(y)

        ham_probability = 1 - spam_probability
        print(f'Spam Probability: {spam_probability:.4f}')
        print(f'Ham Probability: {ham_probability:.4f}')

    def evaluate(self):
        print(f"\n--- {self.classifier_type.replace('_', ' ').title()} Classifier ---")
        print("\nConfusion Matrix:")
        for item1, item2 in zip(self.y_test, self.predictions):
            print(f"{item1}\t{item2}")
            
        # Create DataFrame
        df = pd.DataFrame({
            'Actual Lable': self.y_test,
            'Prediction Label': self.predictions
        })

        # Save to CSV
        df.to_csv('combined_lists.csv', index=False)
                    
        # e = [self.y_test, self.predictions]
        # print(e)
        print(confusion_matrix(self.y_test, self.predictions))
        print("\nClassification Report:")
        
        print(classification_report(self.y_test, self.predictions))
        # converted_labelstest = ["spam" if x == 1 else "ham" for x in self.y_test]
        # converted_labelspre = ["spam" if x == 1 else "ham" for x in self.predictions]
        # print(classification_report(converted_labelstest, converted_labelspre))


    def predict(self, sample_text):
        # Transform the text sample
        sample_vectorized = self.vectorizer.transform([sample_text])
        return self.model.predict(sample_vectorized)[0]
    
    def predict_proba(self, sample_text):
        # Return probability of spam for a given text
        sample_vectorized = self.vectorizer.transform([sample_text])
        return self.model.predict_proba(sample_vectorized)[0][1]

