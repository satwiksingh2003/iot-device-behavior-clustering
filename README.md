# IoT Device Behavior Clustering

This project builds a machine learning model to cluster IoT device network behavior patterns using the CICIoT2023 dataset.

## Features
- Unsupervised clustering of IoT traffic
- Behavior pattern detection
- Device behavior fingerprinting

## Dataset
Dataset used: CICIoT2023

Due to GitHub file size limits, the dataset is not included.

Download it from:
https://www.unb.ca/cic/datasets/iotdataset-2023.html

After downloading, place it in:

data/CICIOT23/train
data/CICIOT23/test
data/CICIOT23/validation
## How to Run

Train the model:

python train_cluster_model.py

Predict device behavior cluster:

python predict_cluster.py
