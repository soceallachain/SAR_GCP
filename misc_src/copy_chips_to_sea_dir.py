import os
import shutil
import pandas as pd
import glob

# Path to the CSV file
csv_file = "data.csv"

# Directory where matching files will be moved
destination_dir = "./Sea"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_file)

# Iterate through each row in the CSV file
for _, row in df.iterrows():
    slc_id = row['SLC_ID']
    burst_number = row['Burst_Number']

    # Construct the file pattern
    pattern = f"{slc_id}_*_burst#{burst_number}.npy"

    # Use glob to find matching files
    matching_files = glob.glob(pattern)

    # Move each matching file to the destination directory
    for file in matching_files:
        shutil.copy(file, destination_dir)
        #shutil.move(file, destination_dir)

print(f"Matching files moved to {destination_dir}.")

