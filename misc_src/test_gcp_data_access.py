#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3rd Dec 2025
Trying to read data from GCP and store locally and on GCP

@author: joe
"""


import pandas as pd
import numpy as np

from google.cloud import storage
from io import BytesIO
#storage_client = storage.Client()


# --- Configuration ---
# Replace with your actual project ID and bucket name
project_id =  "ds-ml-vertex-ai"
bucket_name = "ubot-sar-data"
file_name = "data/Copernicus/Train/Reports/Train_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt50m.csv"

# --- Execution ---
client = storage.Client(project=project_id)

# Get the bucket and the blob (file)
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(file_name)

# Download the file contents into memory
print(f"Downloading {file_name} in memory...")
file_bytes = blob.download_as_bytes()
data_stream = BytesIO(file_bytes)

# Read the data directly into a Pandas DataFrame
df = pd.read_csv(data_stream)
print("\n--- Data Head (Method 2) ---")
print(df.head())

