#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:49:36 2025

@author: joe
"""

import pandas as pd
import os
import shutil
import re
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Set up paths
label_dir       = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/'
label_file      = 'SARFish_RAW_Public_BurstChipInfo_Only_Sea_HighConfidenceShips_gt_100m.csv'
label_file_path = f'{label_dir}{label_file}'
source_dir      = "/path/to/source/files"  # 👈 your folder containing the .npy files
output_dir      = "new_directory"             # <- where you want to copy files
missing_log     = "missing_files.log"        # <- file to log missing ones

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load filenames
df = pd.read_csv(label_file_path)

# Update the file_list to include full paths
file_list = [os.path.join(source_dir, fname) for fname in df["Chip_Names"].tolist()]

# Thread-safe list for logging (can also use a Queue if needed)
missing_files = []



def extract_product_burst_and_chip_information(file):
    """

    Parameters
    ----------        
    file : Function to parse filename of a chip to obtain its root
           SLC_ID and the burst it belongs to
    Returns
    -------
    product_id : SLC_PRODUCT_ID
    burst_number : 
    chip_number : 

    """
    if not os.path.exists(file):
        print(f'{file} not found ....')
    
    else:
        filename= os.path.basename(file)
        parts = filename.split("_")
        product_id = "_".join(parts[0:10])
        burst_index = next(i for i, part in enumerate(parts) if "burst#" in part)
        chip_index = next(i for i, part in enumerate(parts) if "chip#" in part)
        burst_number = parts[burst_index].split("#")[1]
        chip_number =parts[chip_index].split("#")[1].split(".")[0]
        return product_id, burst_number, chip_number
    


# Helper function for multiprocessing
def copy_and_rename(src_file):
    try:
        # Check if source file exists
        if not os.path.exists(src_file):
            # Log missing file
            return f"MISSING::{src_file}"

        # Extract and rename
        base_name = os.path.basename(src_file)
        new_name = re.sub(r'_#HShips.*(?=\.npy$)', '', base_name)
        dest_path = os.path.join(output_dir, new_name)

        shutil.copy2(src_file, dest_path)

        return f"Copied::{src_file} -> {dest_path}"
    except Exception as e:
        return f"ERROR::{src_file}:: {e}"

# Parallel processing
if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        results = pool.map(copy_and_rename, file_list)

    # Separate and print results
    with open(missing_log, "w") as log_file:
        for res in results:
            if res.startswith("MISSING::"):
                print(f"⚠️ {res}")
                log_file.write(res.replace("MISSING::", "") + "\n")
            elif res.startswith("ERROR::"):
                print(f"❌ {res}")
                log_file.write("ERROR: " + res + "\n")
            else:
                print(f"✅ {res}")

    print(f"\nDone. Missing files logged in: {missing_log}")
