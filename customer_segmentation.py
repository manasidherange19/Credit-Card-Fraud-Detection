import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("creditcard.csv")

# Select features for segmentation
segmentation_features = df[['Amount', 'Time']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Save clustered data
df[['Amount', 'Time', 'Cluster']].to_csv("customer_segments.csv", index=False)
print("✅ Customer segments saved to 'customer_segments.csv'.")

# Plot segmentation
plt.figure(figsize=(8,6))
plt.scatter(df['Amount'], df['Time'], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Customer Segmentation by KMeans')
plt.xlabel('Transaction Amount')
plt.ylabel('Time')
plt.grid(True)
plt.show()
