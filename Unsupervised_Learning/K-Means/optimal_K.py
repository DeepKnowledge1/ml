import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample Data
data = np.array([[10, 30], [15, 35], [30, 60], [40, 70], [50, 80], [55, 85]])

# Try different values of K
wcss = []
K_range = range(1, 10)  # Testing K from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)  # Inertia is WCSS

# Plot the Elbow Curve
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.show()
