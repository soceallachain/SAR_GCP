#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:45:59 2025

@author: joe
"""

import os
import pandas as pd
import glob
import multiprocessing

# File Paths
csv_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SARFish_RAW_Public_BurstChipInfo.csv"
#source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Test3"   # Change to your source directory
source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_renaming"   # Change to your source directory
#destination_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Test3/Sea"
destination_dir = source_dir
# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Load CSV data efficiently
df = pd.read_csv(csv_file, usecols=["Product_ID", "Burst_Number", "Chip_Number", "Ships_per_Chip", "Sea_Status", "H_Ships_per_Chip", "M_Ships_per_Chip", "L_Ships_per_Chip"])

# Create a dictionary mapping (Product_ID, Burst_Number, Chip_Number) -> Ships_per_chip
file_mapping = {
    (row["Product_ID"], row["Burst_Number"], row["Chip_Number"]): row["Ships_per_Chip"]
    for _, row in df.iterrows()
}

# Create a dictionary mapping (Product_ID, Burst_Number, Chip_Number) -> Sea_Status
file_mappingS = {
    (row["Product_ID"], row["Burst_Number"], row["Chip_Number"]): row["Sea_Status"]
    for _, row in df.iterrows()
}

file_mappingH = {
    (row["Product_ID"], row["Burst_Number"], row["Chip_Number"]): row["H_Ships_per_Chip"]
    for _, row in df.iterrows()
}
file_mappingM = {
    (row["Product_ID"], row["Burst_Number"], row["Chip_Number"]): row["M_Ships_per_Chip"]
    for _, row in df.iterrows()
}
file_mappingL = {
    (row["Product_ID"], row["Burst_Number"], row["Chip_Number"]): row["L_Ships_per_Chip"]
    for _, row in df.iterrows()
}


# Function to rename and move files
def process_file(file):
    filename = os.path.basename(file)

    # Extract Product_ID, middle_part, Burst_Number, and Chip_Number from the filename
    parts = filename.split("_")
    
    # Identify indices of key components in filename
    try:
        product_id = "_".join(parts[0:10])
        burst_index = next(i for i, part in enumerate(parts) if "burst#" in part)
        chip_index = next(i for i, part in enumerate(parts) if "chip#" in part and "npy" in part)

        burst_number = parts[burst_index].split("#")[1]
        #Extract chip number after '#' and before '.'
        chip_number = parts[chip_index].split('.')[0].split('#')[1]


        middle_part = "_".join(parts[10:burst_index])  # Everything between Product_ID and burst#

        # Look up Ships_per_chip
        ships_per_chip = file_mapping.get((product_id, int(burst_number), int(chip_number)), "UNKNOWN")
        sea_status = file_mappingS.get((product_id, int(burst_number), int(chip_number)), "UNKNOWN")
        h_ships_per_chip = file_mappingH.get((product_id, int(burst_number), int(chip_number)), "UNKNOWN")
        m_ships_per_chip = file_mappingM.get((product_id, int(burst_number), int(chip_number)), "UNKNOWN")
        l_ships_per_chip = file_mappingL.get((product_id, int(burst_number), int(chip_number)), "UNKNOWN")
        #Only rename if find match in csv file
        if (ships_per_chip !="UNKNOWN" and sea_status != "UNKNOWN" 
                                      and h_ships_per_chip !="UNKNOWN" 
                                      and m_ships_per_chip !="UNKNOWN" 
                                      and l_ships_per_chip !="UNKNOWN"):
            # Construct new filename
            new_filename = (f"{product_id}_{middle_part}_burst#{burst_number}_chip#{chip_number}_#Ships{ships_per_chip}"
                           f"_SeaStatus{sea_status}_H{h_ships_per_chip}M{m_ships_per_chip}L{l_ships_per_chip}.npy")
            new_path = os.path.join(destination_dir, new_filename)
            # Rename and move the file
            os.rename(file, new_path)
            print(f"Renamed: {file} -> {new_path}")

    except Exception as e:
        print(f"Skipping {filename}: {e}")

# Get all .npy files in the source directory
files = glob.glob(os.path.join(source_dir, "*.npy"))

# Use multiprocessing to speed up file renaming
if __name__ == "__main__":
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(process_file, files)

    print("File renaming completed!")
