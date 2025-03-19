import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================
# 1️⃣ Create Sample Data
# ============================
X = np.array([
    [2, 4],
    [4, 6],
    [6, 8],
    [8, 10],
    [10, 12]
])

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=['X1', 'X2'])
print("Original Data:\n", df)

# ============================
# 2️⃣ Standardize the Data
# ============================
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Convert to DataFrame for better visualization
df_std = pd.DataFrame(X_std, columns=['X1_std', 'X2_std'])
print("\nStandardized Data:\n", df_std)

# ============================
# 3️⃣ Compute PCA Manually (Using NumPy)
# ============================
# Step 1: Compute the Covariance Matrix
cov_matrix = np.cov(X_std.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 2: Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 3: Sort Eigenvalues and Eigenvectors
sorted_indices = np.argsort(-eigenvalues)  # Sort in descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 4: Compute Principal Components (Project Data)
PCs = X_std.dot(eigenvectors)

# Convert to DataFrame for better visualization
df_pca_manual = pd.DataFrame(PCs, columns=['PC1', 'PC2'])
print("\nPCA Result (Manual Calculation):\n", df_pca_manual)

# ============================
# 4️⃣ Compute PCA Using Scikit-Learn
# ============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Convert to DataFrame for better visualization
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
print("\nPCA Result (Scikit-Learn):\n", df_pca)

# Explained Variance
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# ============================
# 5️⃣ Visualizing PCA Results
# ============================
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', edgecolors='black', alpha=0.7)
plt.axhline(0, color='gray', linestyle='dashed')
plt.axvline(0, color='gray', linestyle='dashed')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Transformation")
plt.grid()
plt.show()



##########################################



# Incremental PCA



##########################################