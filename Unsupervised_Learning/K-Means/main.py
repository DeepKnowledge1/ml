import random
import math
random.seed(0)  # Set a fixed random seed for reproducibility

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# Initialize K-Means with given dataset and K value
def kmeans(data, K, max_iterations=100):
    # Step 1: Initialize centroids randomly by selecting K random points
    centroids = random.sample(data, K)
    
    for _ in range(max_iterations):
        # Step 2: Assign points to the nearest centroid
        clusters = {i: [] for i in range(K)}
        
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid_index = distances.index(min(distances))
            clusters[closest_centroid_index].append(point)
        
        # Step 3: Recalculate centroids
        new_centroids = []
        for i in range(K):
            cluster_points = clusters[i]
            if cluster_points:  # Avoid empty cluster division by zero
                new_centroid = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
            else:  # If cluster is empty, reinitialize randomly
                new_centroid = random.choice(data)
            new_centroids.append(new_centroid)
        
        # Step 4: Check for convergence (if centroids do not change)
        if new_centroids == centroids:
            break
        
        centroids = new_centroids  # Update centroids

    return centroids, clusters

# Example dataset: (Annual Income in $1000, Spending Score)
data = [
    (10, 30), (15, 35), (30, 60), (40, 70), (50, 80), (55, 85)
]

# Run K-Means with K=2
K = 2
final_centroids, final_clusters = kmeans(data, K)

# Output the results

print("Pure Python Output")

print("Final Centroids:", final_centroids)
for cluster_idx, points in final_clusters.items():
    print(f"Cluster {cluster_idx + 1}: {points}")


####################################################################

#############     sklearn               ################

####################################################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample Data (Annual Income, Spending Score)
data = np.array([[10, 30], [15, 35], [30, 60], [40, 70], [50, 80], [55, 85]])

# Applying K-Means with K=2
kmeans = KMeans(n_clusters=2, random_state=0, max_iter=100)
kmeans.fit(data)

# Getting the cluster labels
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# Output the results
print("Python with SKlearn Output")
for cluster_idx in range(len(centroids)):
    print(f"Cluster {cluster_idx + 1}: {centroids[cluster_idx]}")
    
# Plotting
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label="Customers")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.xlabel("Annual Income ($1000)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.show()
