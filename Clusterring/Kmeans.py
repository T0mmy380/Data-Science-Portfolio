import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the movie dataset
df = pd.read_csv('movie_dataset.csv')

features = df[['budget', 'revenue', 'popularity', 'vote_average']].dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia = []  # Sum of squared distances

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig('Clusterring/kmeans_elbow_method.png')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(8,6))
sns.scatterplot(x='budget', y='revenue', hue='Cluster', data=df, palette='viridis')
plt.title('K-Means Clustering of Movies')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.savefig('Clusterring/kmeans_clusters.png')
plt.show()

cluster_summary = df.groupby('Cluster')[['budget', 'revenue', 'popularity', 'vote_average']].mean()
print(cluster_summary)
