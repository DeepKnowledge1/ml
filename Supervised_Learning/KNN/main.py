import math

# Step 1: Define the dataset
# Each data point has two features: Weight (g) and Sweetness (scale of 1 to 10)
dataset = [
    {"features": [150, 7], "label": "Apple"},
    {"features": [170, 8], "label": "Apple"},
    {"features": [160, 6], "label": "Apple"},
    {"features": [180, 9], "label": "Orange"},
    {"features": [190, 8], "label": "Orange"},
]

# Query point: [Weight, Sweetness]
query_point = [165, 7]

# Number of neighbors (K)
K = 3

# Step 2: Define a function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# Step 3: Calculate distances from the query point to all points in the dataset
distances = []
for data_point in dataset:
    distance = euclidean_distance(query_point, data_point["features"])
    distances.append({"distance": distance, "label": data_point["label"]})

# Step 4: Sort the distances in ascending order
distances.sort(key=lambda x: x["distance"])

# Step 5: Select the K nearest neighbors
k_nearest_neighbors = distances[:K]

# Step 6: Perform majority voting to determine the class
votes = {}
for neighbor in k_nearest_neighbors:
    label = neighbor["label"]
    if label in votes:
        votes[label] += 1
    else:
        votes[label] = 1

# Find the class with the most votes
predicted_class = max(votes, key=votes.get)

# Step 7: Output the result
print("K-Nearest Neighbors:")
for neighbor in k_nearest_neighbors:
    print(f"Distance: {neighbor['distance']:.2f}, Class: {neighbor['label']}")

print("\nVotes:", votes)
print("Pure Python: Predicted Class:", predicted_class)



####################################################################

#############     sklearn               ################

####################################################################




from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Dataset
X = np.array([[150, 7], [170, 8], [160, 6], [180, 9], [190, 8]])
y = np.array(['Apple', 'Apple', 'Apple', 'Orange', 'Orange'])

# Query Point
query_point = np.array([[165, 7]])

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Prediction
prediction = knn.predict(query_point)
print("Sklearn: Predicted Class:", prediction[0])