import math
from collections import Counter

# Sample dataset
data = [
    {"Weather": "Sun", "Time": ">30", "Hungry": "Yes", "Decision": "Bus"},
    {"Weather": "Sun", "Time": "<30", "Hungry": "No", "Decision": "Walk"},
    {"Weather": "Cloud", "Time": ">30", "Hungry": "Yes", "Decision": "Bus"},
    {"Weather": "Cloud", "Time": "<30", "Hungry": "No", "Decision": "Walk"},
    {"Weather": "Rain", "Time": ">30", "Hungry": "Yes", "Decision": "Bus"},
    {"Weather": "Rain", "Time": "<30", "Hungry": "No", "Decision": "Bus"},
]

# Function to calculate entropy
def entropy(labels):
    total_count = len(labels)
    label_counts = Counter(labels)
    entropy_value = 0.0
    for count in label_counts.values():
        probability = count / total_count
        entropy_value -= probability * math.log2(probability)
    return entropy_value

# Function to calculate information gain
def information_gain(data, feature, target):
    total_entropy = entropy([item[target] for item in data])
    total_count = len(data)
    
    # Split the data based on the feature
    feature_values = set(item[feature] for item in data)
    weighted_entropy = 0.0
    
    for value in feature_values:
        subset = [item for item in data if item[feature] == value]
        subset_entropy = entropy([item[target] for item in subset])
        subset_weight = len(subset) / total_count
        weighted_entropy += subset_weight * subset_entropy
    
    return total_entropy - weighted_entropy

# Function to find the best feature to split on
def find_best_feature(data, features, target):
    best_feature = None
    best_gain = -1
    
    for feature in features:
        gain = information_gain(data, feature, target)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    
    return best_feature

# Function to build the decision tree
def build_tree(data, features, target):
    labels = [item[target] for item in data]
    
    # If all labels are the same, return the label
    if len(set(labels)) == 1:
        return labels[0]
    
    # If no features left, return the majority label
    if not features:
        return Counter(labels).most_common(1)[0][0]
    
    # Find the best feature to split on
    best_feature = find_best_feature(data, features, target)
    tree = {best_feature: {}}
    
    # Remove the best feature from the list of features
    remaining_features = [f for f in features if f != best_feature]
    
    # Split the data and recursively build the tree
    feature_values = set(item[best_feature] for item in data])
    for value in feature_values:
        subset = [item for item in data if item[best_feature] == value]
        tree[best_feature][value] = build_tree(subset, remaining_features, target)
    
    return tree

# Function to classify a new instance using the decision tree
def classify(tree, instance):
    if not isinstance(tree, dict):
        return tree
    
    feature = next(iter(tree))
    value = instance[feature]
    
    if value not in tree[feature]:
        return None  # Handle unseen values
    
    return classify(tree[feature][value], instance)

# Define features and target
features = ["Weather", "Time", "Hungry"]
target = "Decision"

# Build the decision tree
decision_tree = build_tree(data, features, target)
print("Decision Tree:", decision_tree)

# Classify a new instance
new_instance = {"Weather": "Sun", "Time": "<30", "Hungry": "No"}
prediction = classify(decision_tree, new_instance)
print("Prediction for new instance:", prediction)



####################################################################

#############     sklearn.Decision tree               ################

####################################################################



# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define the dataset
data = {
    "Weather": ["Sun", "Sun", "Cloud", "Cloud", "Rain", "Rain"],
    "Time": [">30", "<30", ">30", "<30", ">30", "<30"],
    "Hungry": ["Yes", "No", "Yes", "No", "Yes", "No"],
    "Decision": ["Bus", "Walk", "Bus", "Walk", "Bus", "Bus"]
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Convert categorical variables into numerical values using one-hot encoding
df_encoded = pd.get_dummies(df, columns=["Weather", "Time", "Hungry"], drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop("Decision", axis=1)  # Features
y = df_encoded["Decision"]  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Bus", "Walk"])
plt.title("Decision Tree")
plt.show()