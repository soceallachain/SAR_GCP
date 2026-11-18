#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 07:43:34 2026

@author: joe
"""

import time
import numpy as np
import pandas as pd

#from modules.SAR_code_module import *
#from modules.chunk_processing_module import *
from modules.SAR_focusing_module import *


##############################################################################
## Determine RC timings for various data sizes 

save_path    = f'/mnt/SAR_bucket/data/sand_box/rc_testing/timing_results/'
root_path    = f'/mnt/SAR_bucket/data/sand_box/rc_testing/files_for_iain/'
filename     = f'S1B_S3_SLC__1SDH_20180214T001739_20180214T001802_009612_011540_E777_Pol_hv_Node_asc_raw_block0.npy'
npy_file     = f'{root_path}{filename}'


# Load raw data
raw_data = np.load(npy_file)


azimuth_sizes = [1024, 2048, 4096, 8192,16392]
range_sizes   = [1024, 2048, 4096, 8192,16392]




sm_swath = 3
range_padding = 1024
azimuth_padding = 0
az_start_row = 0
rng_end_col = raw_data.shape[1]

# List to store benchmark results
results = []

for az_size in azimuth_sizes:
    for rng_size in range_sizes:
        az_end_row = az_start_row + az_size
        rng_start_col = rng_end_col - rng_size
        aoi_data = raw_data[
            az_start_row:az_end_row, rng_start_col:rng_end_col
        ]

        # Get average of 10 runs
        elapsed_time = []
        for j in range(10):  # Fixed: range(10) instead of range(0:10)
            # Start timer
            start = time.perf_counter()
            rc_data = range_compress_padded_raw_data(
                aoi_data,
                "S1SM",
                sm_swath,
                range_padding,
                azimuth_padding,
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
df_results.to_csv(f'{save_path}RC_timing_results_RangePadding_{range_padding}.csv', 
                  index=False,
                  float_format="%.4f"
                  )


# Plot the results and ssave plot
import matplotlib.pyplot as plt
df=df_results

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
    "Range Compression Performance: Mean Time vs Range Size", fontsize=14
)
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend(title="Azimuth Size", loc="upper left")

plt.tight_layout()

plt.savefig(f'{save_path}RC_timing_results_RangePadding_{range_padding}.jpg',
            format="jpg", dpi=300
            )
# 6. Display the plot
plt.show()