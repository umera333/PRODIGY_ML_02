import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Create dummy data
data = pd.DataFrame({
    'Annual Income (k$)': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
})

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
data['Cluster'] = kmeans.labels_

# Add centroids
centroids = kmeans.cluster_centers_

# Plot with seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='Set2', s=100)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], 
            s=200, c='black', marker='X', label='Centroids')

plt.title('Customer Clusters with Centroids', fontsize=14)
plt.legend()
plt.show()
cluster_counts = data['Cluster'].value_counts().sort_index()
sns.heatmap(cluster_counts.to_frame().T, annot=True, cmap="YlGnBu", cbar=False)
plt.title("Customer Count per Cluster")
plt.xlabel("Cluster")
plt.show()
sns.pairplot(data, hue="Cluster", palette="Set1")
plt.suptitle("Cluster Distribution", y=1.02)
plt.show()