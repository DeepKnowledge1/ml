from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Sample dataset
data = np.array([
    [1, 2],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11]
])

# Perform K-Means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(data)

# Compute Silhouette Score
score = silhouette_score(data, labels)
print(f"Silhouette Score: {score:.3f}")