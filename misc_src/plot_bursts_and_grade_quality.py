#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 17:58:08 2025

@author: joe
Extracts ROIs around ship locations contained in csv label files
Plots real and slc data of ROI and user determines whether
visibility of real interferogram is sufficient for utilisation in
training set
"""

import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define paths
#path = '/media/joe/WD_12TB_HD1/jc_work/Copernicus/Public'
path = '/home/joe/Work/Projects/RSAR/Test/SampleData/Public'
raw_burst_path = f'{path}/raw_burst_data/'
slc_burst_path = f'{path}/slc_burst_data/'
csv_path = f'{path}/Reports/'
img_path='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/t2'

# 2. Read CSV
csv_file = os.path.join(csv_path, 'Public_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m.csv')
columns = ['Product_ID', 'Burst_Number', 'ship_burst_row', 'ship_burst_col', 'ship_length', 'Number_Ships', 'Burst_Name']
df = pd.read_csv(csv_file, usecols=columns)

# 3. Initialize list to store quality labels
quality_labels = []

# 4. Loop over all rows
#for idx, row in df.head(5).iterrows():
for idx, row in df.iterrows():
    burst_name = row['Burst_Name']

    # 4.1 Generate filenames
    # Replace '_#Ships*.npy' with '.npy'

    raw_burst_filename = re.sub(r'_#Ships.*\.npy$', '.npy', burst_name)
    raw_burst_file = os.path.join(raw_burst_path, raw_burst_filename)
    slc_burst_filename = raw_burst_filename.replace('_real_', '_slc_')
    slc_burst_file = os.path.join(slc_burst_path, slc_burst_filename)
    raw_imag_burst_filename = raw_burst_filename.replace('_real_', '_imag_')
    raw_imag_burst_file = os.path.join(raw_burst_path, raw_imag_burst_filename)
    binary_mask_filename = raw_burst_filename.replace('_real_', '_binary_mask_')
    binary_mask_file = os.path.join(raw_burst_path, binary_mask_filename)

    # 4.2 Load numpy arrays
    try:
        raw_burst_data = np.load(raw_burst_file)
        slc_burst_data = np.load(slc_burst_file)
       # raw_imag_burst_data = np.load(raw_imag_burst_file)
        binary_mask_data = np.load(binary_mask_file)
    except Exception as e:
        print(f"Error loading files for Burst '{burst_name}': {e}")
        quality_labels.append('E')  # E for Error
        continue

    # 5. Get row and column
    row_center = int(row['ship_burst_row'])
    col_center = int(row['ship_burst_col'])

    # 6. Calculate ROI indices
    max_row, max_col = raw_burst_data.shape
    size = 2000
    half_size = int(size/2)

    # Row indices
    if row_center < half_size:
        row_start = 0
        row_end = min(size, max_row)
    elif row_center + half_size > max_row:
        row_end = max_row
        row_start = max(0, max_row - size)
    else:
        row_start = row_center - half_size
        row_end = row_center + half_size

    # Column indices
    if col_center < half_size:
        col_start = 0
        col_end = min(size, max_col)
    elif col_center + half_size > max_col:
        col_end = max_col
        col_start = max(0, max_col - size)
    else:
        col_start = col_center - half_size
        col_end = col_center + half_size

    # 6.1 Extract ROIs
    roi_raw = raw_burst_data[row_start:row_end, col_start:col_end]
    roi_slc = slc_burst_data[row_start:row_end, col_start:col_end]
    roi_bin_mask = binary_mask_data[row_start:row_end, col_start:col_end]
    #roi_imag = raw_imag_burst_data[row_start:row_end, col_start:col_end]

    # 7. Plotting
    fig, axs = plt.subplots(3, 1, figsize=(15, 5))
    fig.suptitle(f"{burst_name}: row={row_center}, column={col_center}", fontsize=14)

    # roi_raw - linear scale
    im0 = axs[0].imshow(roi_raw, aspect='auto', cmap='viridis')
    axs[0].set_title('Raw Burst (Linear)')
    #plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # roi_slc - log scale
    im1 = axs[1].imshow(np.log1p(np.abs(roi_slc)), aspect='auto', cmap='viridis')
    axs[1].set_title('SLC Burst (Log Scale)')
    #plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # roi_bin_mask - gray scale
    im2 = axs[2].imshow(roi_bin_mask, aspect='auto', cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('Binary Mask')
    #plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()  # Waits until the user closes the plot window
    
    plt.pause(2)
    
    valid_keys = ['Y', 'y', 'N', 'n', 'M', 'm']
    while True:
        key = input("Press 'Y', 'y', 'N', 'n', 'M', or 'm' to label this burst: ").strip()
        if key in valid_keys:
            quality_labels.append(key)
            save_name = f"{burst_name}_{row_center}_{col_center}.jpg"
            save_path = os.path.join(img_path, save_name)
            fig.savefig(save_path)
            plt.close(fig)
            break
        else:
            print("Invalid key pressed. Please try again.")
    

# 9. Add Quality_OK column to DataFrame
df['Quality_OK'] = quality_labels

# 10. Optionally, save the updated DataFrame
df.to_csv(os.path.join(img_path, 'bursts_labeled.csv'), index=False)

print("Processing complete. Labeled CSV saved as 'bursts_labeled.csv'.")
