from sklearn.metrics import classification_report, confusion_matrix

# True labels
y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 spam emails
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 10 non-spam emails

# Predicted labels
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0,  # 9 correct spam predictions, 1 incorrect
          1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 5 incorrect spam predictions, 5 correct non-spam

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Not Spam", "Spam"]))