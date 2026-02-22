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
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(data_scaled)
labels = gmm.predict(data_scaled)

# Plot clusters
plt.figure()
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels)
plt.title("5G Traffic Clustering using GMM")
plt.show()