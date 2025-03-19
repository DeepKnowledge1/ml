
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

n_components = 2

# Define batch size
batch_size = 10

# Initialize IncrementalPCA without using the batch_size parameter
ipca = IncrementalPCA(n_components=n_components)

# Manually divide data into batches and fit incrementally
n_samples = X.shape[0]
n_batches = int(np.ceil(n_samples / batch_size))

# Fit in batches
for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, n_samples)
    batch = X[start_idx:end_idx]
    
    ipca.partial_fit(batch)
        
# Transform the entire dataset at once after fitting
X_ipca = ipca.transform(X)

# Standard PCA for comparison
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Visualization
colors = ["navy", "turquoise", "darkorange"]

for X_transformed, title in [(X_ipca, "Incremental PCA (Manual Batching)"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(
            X_transformed[y == i, 0],
            X_transformed[y == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()