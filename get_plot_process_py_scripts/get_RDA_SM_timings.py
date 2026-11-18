#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 07:43:34 2026

@author: joe
"""

import time

import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d


from modules.SAR_code_module import *
#from modules.chunk_processing_module import *
from modules.SAR_focusing_module import *
from modules.image_processing_module import *
from scipy.ndimage import zoom

#Set DEBUG mode if do not want to save files
DEBUG = True
# Get list of SLC_IDs

#test_dir   = f'Houston_SM_2019_20'
test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/jc_data/{test_dir}/'
#root_path  = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'
#test_dir   = f'SaoPaulo_SM_2019_20'
#root_path  = f'/mnt/SAR_bucket/data/SaoPaulo_SM/{test_dir}/'

#test_dir   = f'S1A_D_S1_S6_Samples'
root_path     = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'
#save_path     = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/'

# Vm cannot write to GCP for some reason
#save_path     = f'/mnt/SAR_bucket/data/sand_box/rc_testing/files_for_iain/'
raw_burst_dir = f'raw_burst'
slc_burst_dir = f'slc_burst'
rc_burst_dir  = f'rc_burst'

pol1       = 'hv'
pol2       = 'vh'
pol3       = 'vv'
pol4       = 'hh'
pols       = [pol1, pol2, pol3, pol4]
SLC_IDs    = os.listdir(root_path)
swaths     = ['_S2_', '_S4_','_S5_','_S6_','_S3_']
slice_size = [5000, 5000] 
plot_plots=False
#Load raw file
slc_id= SLC_IDs[500]
slc_id_path = root_path + slc_id +'/'

for pol in pols:  
   files       = get_SM_files_from_slc_id(slc_id_path, pol)
   if files != -1:
       break  # Found files
if files == -1:
    print(f'Warning: {slc_id}:  Files not found ')    


##############################################################################
## Determine RC timings for various data sizes 

save_path    = f'/mnt/SAR_bucket/data/sand_box/rc_testing/timing_results/'
#root_path    = f'/mnt/SAR_bucket/data/sand_box/rc_testing/files_for_iain/'
#filename     = f'S1B_S3_SLC__1SDH_20180214T001739_20180214T001802_009612_011540_E777_Pol_hv_Node_asc_raw_block0.npy'
#npy_file     = f'{root_path}{filename}'


# Load raw data
#raw_data = np.load(npy_file)

# get slc_aoi
chunk_info  = process_file_for_chunks(files)
timing_based_offsets = chunk_info[2]
slc_block_dims  = get_SM_slc_matched_chunk_dimensions(timing_based_offsets)
ncc_offsets = process_chunks_for_offsets(chunk_info, slice_size, plot_plots)
slc_block_indices = get_SM_slc_block_offsets(ncc_offsets,
                                            timing_based_offsets
                                            )
slc_block_dims = [[0, slc_block_indices[0][1] - slc_block_indices[0][0]],
                  [slc_block_indices[0][1] - slc_block_indices[0][0] + 1, 
                   slc_block_indices[1][1] - slc_block_indices[0][0]]
                 ]

block = 0
slc_start_row  = 0
slc_end_row    = slc_block_indices[block][1]-slc_block_indices[block][0]
slc_start_col  = 0
slc_end_col    = slc_block_indices[block][3]-slc_block_indices[block][2]
slc_block0_aoi = [slc_start_row,slc_end_row, slc_start_col, slc_end_col]


raw_block0_aoi,sb0 = get_SM_slc_matched_raw_AOIs(slc_block0_aoi, slc_block_indices,
                                slc_block_dims)

block_raw_data = chunk_info[0]
raw_block0_aoi = np.squeeze(raw_block0_aoi)
raw_block0_aoi_data = block_raw_data[block][raw_block0_aoi[0]:raw_block0_aoi[1],
                                            raw_block0_aoi[2]:raw_block0_aoi[3]
                                           ]

# Get meta and ephemerides data for block0
# Needed for full focusing
block0_metadata = chunk_info[1][block]
l0file_block   = chunk_info[4]


slc_block0_aoi_data = get_SM_slc_aoi(slc_block0_aoi,files[1])


#slc_aoi = slc_bloc0_aoi_data[]

azimuth_sizes = [1024, 2048, 4096, 8192,16392]
range_sizes   = [1024, 2048, 4096, 8192,16392]




sm_swath = 3
range_padding = 1024
azimuth_padding =1024
#az_start_row = 0
#rng_end_col = raw_data.shape[1]

# List to store benchmark results
results = []

slc_midpoint= [slc_block0_aoi[1]//2, slc_block0_aoi[3]//2]
for az_size in azimuth_sizes:
    for rng_size in range_sizes:
        
        # get mid-point of slc 
        slc_aoi_start_row = slc_midpoint[0] - az_size//2 - azimuth_padding
        slc_aoi_start_col = slc_midpoint[1] - rng_size//2 - range_padding
        
        az_start_row  = slc_block_indices[block][0] + slc_aoi_start_row
        az_end_row    = az_start_row + az_size + 2*azimuth_padding
        rng_start_col = slc_block_indices[block][2]  + slc_aoi_start_col
        rng_end_col   = rng_start_col + rng_size + 2*range_padding
        raw_roi_data  = raw_block0_aoi_data[az_start_row:az_end_row, 
                                 rng_start_col:rng_end_col
                                 ]
        # Note raw_roi is  [range_start, range_length, az_start, az_length]
        raw_roi       = [rng_start_col, az_size + 2*range_padding,
                         az_start_row, az_size + 2*azimuth_padding
                         ]

        # Get average of 10 runs
        elapsed_time = []
        for j in range(10):  # Fixed: range(10) instead of range(0:10)
            # Start timer
            start = time.perf_counter()
            rda_data = focus_SM_burst(block0_metadata,
                raw_roi_data,
                raw_roi,
                l0file_block,
                'rma'
            )
            elapsed = time.perf_counter() - start
            elapsed_time.append(elapsed)

        # Calculate statistics using NumPy
        min_time = np.min(elapsed_time)
        max_time = np.max(elapsed_time)
        mean_time = np.mean(elapsed_time)

        # Append row dict to results list
        results.append(
            {
                "Azimuth Size": az_size,
                "Range Size": rng_size,
                "Mean Time": mean_time,
                "Max Time": max_time,
                "Min Time": min_time,
            }
        )

        print(
            f"Azimuth x Range of {az_size} x {rng_size} takes {mean_time:.6f} "
            f"seconds for range compression with Range Padding of {range_padding}"
        )

# Create the pandas DataFrame
df_results = pd.DataFrame(results)

# Display the DataFrame
print(df_results)
df_results.to_csv(f'{save_path}RDA_timing_results_RngPadding_{range_padding}_'
                  f'AzPadding_{azimuth_padding}.csv', 
                  index=False,
                  float_format="%.4f"
                  )


# Plot the results and save plot
import matplotlib.pyplot as plt
df=df_results
plt.style.use("dark_background")
# 2. Setup Plot Figure
plt.figure(figsize=(10, 6))

# Define distinct markers and colors for each Azimuth Size
markers = ["o", "s", "^", "D", "v"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

azimuth_sizes = sorted(df["Azimuth Size"].unique())

# 3. Plot dashed line for each Azimuth Size
for idx, az_size in enumerate(azimuth_sizes):
    sub_df = df[df["Azimuth Size"] == az_size].sort_values("Range Size")

    plt.plot(
        sub_df["Range Size"],
        sub_df["Mean Time"],
        linestyle="--",  # Dashed line between points
        marker=markers[idx % len(markers)],  # Distinct symbol
        color=colors[idx % len(colors)],  # Distinct color
        linewidth=2,
        markersize=7,
        label=f"Azimuth Size = {az_size}",
    )

# 4. Format X-Axis ticks in 1024 (1k) units
unique_range_sizes = sorted(df["Range Size"].unique())
x_tick_labels = [f"{int(r // 1024)}k" for r in unique_range_sizes]

plt.xticks(ticks=unique_range_sizes, labels=x_tick_labels)

# 5. Customize Plot Formatting
plt.xlabel("Range Size (k)", fontsize=12)
plt.ylabel("Mean Time (seconds)", fontsize=12)
plt.title(
    "RDA Focusing Performance: Mean Time vs Range Size", fontsize=14
)
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend(title="Azimuth Size", loc="upper left")

plt.tight_layout()

plt.savefig(f'{save_path}RDA_timing_results_RngPadding_{range_padding}_'
            f'AzPadding_{azimuth_padding}.jpg',
            format="jpg", dpi=300
            )
# 6. Display the plot
plt.show()
