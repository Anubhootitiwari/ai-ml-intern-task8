# Task 8: K-Means Clustering

## Objective
Perform unsupervised learning using **K-Means clustering** to group customers based on income and spending score.

## Steps Performed
1. Loaded dataset (Mall Customer Segmentation).
2. Selected relevant features for clustering.
3. Standardized data for better clustering performance.
4. Applied **Elbow Method** to find optimal number of clusters.
5. Trained K-Means with chosen k.
6. Evaluated clusters using **Silhouette Score**.
7. Visualized clusters with centroids.

## Requirements
```bash
pip install pandas matplotlib scikit-learn
```

## Run
```bash
python task8_kmeans.py
```

## Output
- Elbow plot for optimal k
- Cluster scatter plot with centroids
- Silhouette score in terminal

## Results
![Elbow Method](screenshots/elbow_plot.png)
![Clusters](screenshots/clusters.png)
