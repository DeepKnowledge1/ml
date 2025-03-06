# Step 1: Define the dataset
data = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play Tennis": "No"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "Play Tennis": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play Tennis": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play Tennis": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play Tennis": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play Tennis": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play Tennis": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play Tennis": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play Tennis": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play Tennis": "No"}
]

# Step 2: Calculate prior probabilities
def calculate_prior_probabilities(data, target_column):
    classes = {}
    for row in data:
        target_value = row[target_column]
        if target_value not in classes:
            classes[target_value] = 0
        classes[target_value] += 1
    total = len(data)
    for key in classes:
        classes[key] /= total
    return classes

# Step 3: Calculate likelihoods
def calculate_likelihoods(data, feature, feature_value, target_column, target_value):
    count_feature_and_target = 0
    count_target = 0
    for row in data:
        if row[target_column] == target_value:
            count_target += 1
            if row[feature] == feature_value:
                count_feature_and_target += 1
    return count_feature_and_target / count_target

# Step 4: Predict using Naïve Bayes
def predict(data, new_instance, target_column):
    # Calculate prior probabilities
    priors = calculate_prior_probabilities(data, target_column)
    
    # Initialize posterior probabilities
    posteriors = {}
    
    # For each class, calculate the posterior probability
    for target_value in priors:
        likelihood = 1
        for feature in new_instance:
            likelihood *= calculate_likelihoods(data, feature, new_instance[feature], target_column, target_value)
        posteriors[target_value] = likelihood * priors[target_value]
    
    # Return the class with the highest posterior probability
    return max(posteriors, key=posteriors.get)

# Step 5: Test the model
new_instance = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
prediction = predict(data, new_instance, "Play Tennis")
print(f"Prediction: {prediction}")


####################################################################

#############     sklearn.naive_bayes               ################

####################################################################


from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Define the dataset


# Step 2: Prepare Label Encoders for each feature
label_encoders = {}
features = ["Outlook", "Temperature", "Humidity", "Wind", "Play Tennis"]

# Create a separate encoder for each feature
for feature in features:
    label_encoders[feature] = LabelEncoder()
    # Fit the encoder on all values for this feature
    label_encoders[feature].fit([row[feature] for row in data])

# Step 3: Convert categorical data to numerical data
encoded_data = []
for row in data:
    encoded_row = {}
    for feature in features:
        encoded_row[feature] = label_encoders[feature].transform([row[feature]])[0]
    encoded_data.append(encoded_row)

# Step 4: Prepare the dataset
X = np.array([
    [row["Outlook"], row["Temperature"], row["Humidity"], row["Wind"]] 
    for row in encoded_data
])
y = np.array([row["Play Tennis"] for row in encoded_data])

# Step 5: Train the Naïve Bayes model
model = CategoricalNB()
model.fit(X, y)

# Step 6: Prepare the new instance for prediction
new_instance = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}

# Encode the new instance
encoded_new_instance = np.array([
    label_encoders["Outlook"].transform([new_instance["Outlook"]])[0],
    label_encoders["Temperature"].transform([new_instance["Temperature"]])[0],
    label_encoders["Humidity"].transform([new_instance["Humidity"]])[0],
    label_encoders["Wind"].transform([new_instance["Wind"]])[0]
]).reshape(1, -1)

# Step 7: Make a prediction
prediction = model.predict(encoded_new_instance)
predicted_class = label_encoders["Play Tennis"].inverse_transform(prediction)[0]

print(f"Prediction: {predicted_class}")
