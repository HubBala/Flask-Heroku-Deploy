import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Lung Cancer Dataset Innv.csv')
df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})

df = df.dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Elbow method 
inertia=[]
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state= 42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
# plot Elbow
'''plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k) ")
plt.ylabel("inertia")
plt.title("Elbow Method")
plt.show()'''

# k-means Clustering
optimal_k=3 
kmeans = KMeans(n_clusters = optimal_k, random_state = 42, n_init = 10 )
df['Cluster'] = kmeans.fit_predict(df_scaled)

print(df.head())

for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Characteristics:\n", df[df['Cluster'] == cluster].mean())

# plot for cluster
plt.scatter(df_scaled[:,0], df_scaled[:,1], c=df['Cluster'], cmap='viridis')
for i in df['Cluster']:
    center = kmeans.cluster_centers_
    plt.scatter(center[:, 0],center[:, 1],marker = '^',c = 'red', s=200, edgecolors = 'black', label = 'Cluster Centers')

plt.legend()
plt.title("K-Means Clustering")
plt.xlabel("AGE")
plt.ylabel("BREATHING_ISSUE")
plt.show()

