import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Load dataset
# -----------------------------

folder = "data/CICIOT23/train"

files = []

for file in os.listdir(folder):
    if file.endswith(".csv"):
        files.append(os.path.join(folder, file))

print("Number of files found:", len(files))

dataframes = []

for file in files:
    print("Loading:", file)
    df = pd.read_csv(file)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("\nDataset loaded successfully")
print("Dataset shape:", data.shape)

# -----------------------------
# STEP 2: Remove label column
# -----------------------------

if "label" in data.columns:
    data = data.drop(columns=["label"])

# -----------------------------
# STEP 3: Clean dataset
# -----------------------------

data = data.fillna(0)
data = data.drop_duplicates()

print("Dataset after cleaning:", data.shape)

# -----------------------------
# STEP 4: Select behavior features
# -----------------------------

features = data[[
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "ack_count",
    "syn_count",
    "rst_count",
    "Tot size",
    "AVG",
    "Std",
    "IAT",
    "Number"
]]

print("Selected features shape:", features.shape)

# -----------------------------
# STEP 5: Normalize features
# -----------------------------

scaler = StandardScaler()
X = scaler.fit_transform(features)

print("Feature scaling completed")

# -----------------------------
# STEP 6: Train clustering model
# -----------------------------

model = MiniBatchKMeans(
    n_clusters=12,          # more clusters → better separation
    batch_size=10000,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X)

data["cluster"] = clusters

print("\nClustering completed")

# -----------------------------
# STEP 7: Check cluster distribution
# -----------------------------

print("\nCluster distribution:")
print(data["cluster"].value_counts())

# -----------------------------
# STEP 8: Evaluate cluster quality
# -----------------------------

sample_size = min(10000, len(X))
sample_indices = np.random.choice(len(X), sample_size, replace=False)

score = silhouette_score(X[sample_indices], clusters[sample_indices])

print("\nSilhouette Score:", score)

# -----------------------------
# STEP 9: Save trained model
# -----------------------------

joblib.dump(model, "device_cluster_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully")

# -----------------------------
# STEP 10: Print cluster centers
# -----------------------------

print("\nCluster centers learned by model:")
print(model.cluster_centers_)

# -----------------------------
# STEP 11: Visualize clusters
# -----------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    data["flow_duration"],
    data["Rate"],
    c=data["cluster"],
    s=5
)

plt.xlabel("Flow Duration")
plt.ylabel("Rate")
plt.title("Device Behavior Clusters")

plt.show()

print("\nTraining pipeline finished")