# Task 8: K-Means Clustering
# Author: Rucha
# Tools: Scikit-learn, Pandas, Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
# Example dataset: Mall Customer Segmentation
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv"
df = pd.read_csv(url)

print("First 5 rows of data:")
print(df.head())

# 2. Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.savefig('screenshots/elbow_plot.png')
plt.close()

# 4. Fit K-Means with chosen k (example: k=5)
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = labels

# 5. Evaluate Clustering
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for k={k_optimal}: {sil_score:.3f}")

# 6. Visualize Clusters
plt.figure(figsize=(6,4))
for cluster in range(k_optimal):
    cluster_points = X_scaled[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='black', marker='X', label='Centroids')
plt.title(f'K-Means Clustering (k={k_optimal})')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.savefig('screenshots/clusters.png')
plt.close()
