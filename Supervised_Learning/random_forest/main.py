# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the dataset
data = {
    'Student': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Hours_Studied': [5, 2, 6, 3, 4, 1, 7, 2, 4, 3],
    'Attendance': [90, 60, 85, 70, 80, 50, 95, 65, 75, 80],
    'Previous_Exam_Score': [80, 50, 75, 60, 70, 40, 90, 55, 65, 60],
    'Pass_Fail': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail']
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Step 2: Prepare the data
# Features (X) and Target (y)
X = df[['Hours_Studied', 'Attendance', 'Previous_Exam_Score']]
y = df['Pass_Fail']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create and train the Random Forest model
# We'll use 3 trees (n_estimators=3) for simplicity
rf_model = RandomForestClassifier(n_estimators=3, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
print("Test Set Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict for a new student
new_student = [[4, 85, 75]]  # Hours Studied = 4, Attendance = 85%, Previous Exam Score = 75
prediction = rf_model.predict(new_student)
print("\nPrediction for New Student:", prediction[0])