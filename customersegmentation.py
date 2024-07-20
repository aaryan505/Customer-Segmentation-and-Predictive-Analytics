from sklearn.cluster import KMeans

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(customer_data)

# Analyze customer segments
customer_segments = customer_data.groupby('cluster').mean()
print(customer_segments)
