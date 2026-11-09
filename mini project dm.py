import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import datetime as dt

# -------------------------------------------------------------
# 1. DATA COLLECTION & PREPROCESSING
# -------------------------------------------------------------
# For demonstration: Generate synthetic transactional data or load Online Retail CSV
np.random.seed(42)
n_records = 5000

customer_ids = np.random.randint(10001, 10500, size=n_records)
invoice_dates = [
    dt.datetime(2025, 1, 1) + dt.timedelta(days=int(np.random.uniform(0, 365))) 
    for _ in range(n_records)
]
quantities = np.random.exponential(scale=3, size=n_records).astype(int) + 1
unit_prices = np.round(np.random.exponential(scale=15, size=n_records) + 2.0, 2)

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'InvoiceDate': invoice_dates,
    'Quantity': quantities,
    'UnitPrice': unit_prices
})
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Data Cleaning: Handle missing values and invalid rows
df.dropna(subset=['CustomerID'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# -------------------------------------------------------------
# 2. FEATURE EXTRACTION: RFM (RECENCY, FREQUENCY, MONETARY)
# -------------------------------------------------------------
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'CustomerID': 'count',                                   # Frequency
    'TotalAmount': 'sum'                                     # Monetary Value
}).rename(columns={
    'InvoiceDate': 'Recency',
    'CustomerID': 'Frequency',
    'TotalAmount': 'Monetary'
}).reset_index()

# -------------------------------------------------------------
# 3. DATA TRANSFORMATION & SCALING
# -------------------------------------------------------------
# Log transform to mitigate skewness
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

# Standardization (Z-score scaling)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# -------------------------------------------------------------
# 4. OPTIMAL CLUSTER EVALUATION (ELBOW & SILHOUETTE)
# -------------------------------------------------------------
inertia = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans_temp.fit(rfm_scaled)
    inertia.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans_temp.labels_))

# Select optimal k (e.g., k=4 based on standard customer segment tiers)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# -------------------------------------------------------------
# 5. MODEL EVALUATION METRICS
# -------------------------------------------------------------
sil_score = silhouette_score(rfm_scaled, rfm['Cluster'])
db_score = davies_bouldin_score(rfm_scaled, rfm['Cluster'])

print(f"Model Evaluation:")
print(f"Silhouette Score (Target: > 0.4): {sil_score:.4f}")
print(f"Davies-Bouldin Index (Lower is better): {db_score:.4f}\n")

# Cluster Summary
cluster_profile = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(2)
print("Segment Profiling Summary:")
print(cluster_profile)