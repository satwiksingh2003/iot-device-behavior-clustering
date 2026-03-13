import joblib
import pandas as pd

# Load trained clustering model
model = joblib.load("device_cluster_model.pkl")

# Load feature scaler
scaler = joblib.load("scaler.pkl")

# Feature names (must match training)
columns = [
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
]


def predict_device_behavior(device_features):

    df = pd.DataFrame([device_features], columns=columns)

    scaled = scaler.transform(df)

    cluster = model.predict(scaled)

    return cluster[0]


# Function for teammates to use
def get_device_cluster(device_features):
    return predict_device_behavior(device_features)


# Example device behavior (test)
device_data = [
1200,
50,
6,
100,
200,
150,
120,
5,
2,
0,
500,
30,
5,
10,
3
]

cluster = predict_device_behavior(device_data)

print("Predicted Device Cluster:", cluster)