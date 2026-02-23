import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Create synthetic traffic data
np.random.seed(42)

low = np.random.normal([20, 30], [5, 5], (200, 2))
medium = np.random.normal([50, 60], [7, 7], (200, 2))
high = np.random.normal([80, 90], [6, 6], (200, 2))

data = np.vstack((low, medium, high))

# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply GMM
# Model Selection using BIC
bic_scores = []

for k in range(1, 6):
    gmm_test = GaussianMixture(n_components=k, random_state=42)
    gmm_test.fit(data_scaled)
    bic_scores.append(gmm_test.bic(data_scaled))

print("BIC Scores for 1–5 components:", bic_scores)

# Choose best K (lowest BIC)
best_k = bic_scores.index(min(bic_scores)) + 1
print("Optimal number of components:", best_k)

# Fit final GMM with best K
gmm = GaussianMixture(n_components=best_k, random_state=42)
gmm.fit(data_scaled)
labels = gmm.predict(data_scaled)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# KMeans Clustering
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Silhouette Scores
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
gmm_silhouette = silhouette_score(data_scaled, labels)

print("KMeans Silhouette Score:", kmeans_silhouette)
print("GMM Silhouette Score:", gmm_silhouette)

if gmm_silhouette > kmeans_silhouette:
    print("GMM provides better cluster separation than KMeans.")
else:
    print("KMeans performs better for this dataset.")
# Plot clusters
plt.figure()
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels)
plt.title("5G Traffic Clustering using GMM")
plt.savefig("gmm_output.png")
plt.show()
# Simulated energy consumption per cluster (hypothetical units)
energy_map = {
    0: 50,   # low traffic
    1: 80,   # medium traffic
    2: 120   # high traffic
}

total_energy = sum([energy_map[label] for label in labels])

print("Estimated Total Energy Consumption:", total_energy)