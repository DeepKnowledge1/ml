import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Step 1: Define the dataset
# Each row represents a point (x, y) in 2D space
data = np.array([
    [1, 1],    # Point A
    [1.5, 1.5], # Point B
    [5, 5],    # Point C
    [3, 4],    # Point D
    [4, 4]     # Point E
])

# Step 2: Perform hierarchical clustering using the 'single' linkage method
# The 'linkage' function computes the hierarchical clustering
Z = linkage(data, method='single', metric='euclidean')

# Step 3: Visualize the dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z, labels=['A', 'B', 'C', 'D', 'E'])
plt.title("Dendrogram of Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Optional: Print the linkage matrix Z
print("Linkage Matrix:")
print(Z)

# Step 4: Extract clusters at a specific distance threshold
# Let's extract clusters with a distance threshold of 2.0
clusters = fcluster(Z, t=2.0, criterion='distance')
print("Cluster assignments:", clusters)