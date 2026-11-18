
# extract_SM_slc_matched_raw_offsets.py 


# Script to determine offsets between Raw and SLC SM data
# Returns azimuth and range start offsets for each chunk/block as well as the
# number of azimuth lines and range samples in each block
# Needs only the name and path to the top level directory which must contain
#  /RAW and /SLC directories that each contain the respective .SAFE file
# The toplevel directory must have the same same as the stem of the SLC .SAFE file




# 1. Setup


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
from modules.SAR_focusing_module import *
from modules.image_processing_module import *
from modules.chunk_processing_module import *
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
root_path  = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'

pol1       = 'hv'
pol2       = 'vh'
pol        = [pol1, pol2]
SLC_IDs    = os.listdir(root_path)
swaths     = ['_S2_', '_S4_','_S5_','_S6_','_S3_']

slice_size = [5000, 5000]
plot_plots = False
n_files_per_swath = 1


#2. Get single file
slc_id = SLC_IDs[500]
print(f'Raw-SLC Offsets for: {slc_id}')



# Get offsets for single file
#chunk_info = process_file_for_chunks(slc_id, root_path, pol)
#offsets    = process_chunks_for_offsets(chunk_info,
#                                        slice_size,
#                                        plot_plots
#                                        )


#instead of this:
swath_offsets =  process_files_for_chunk_offsets(slc_id,root_path,swaths,
                                                  n_files_per_swath, pol,
                                                  slice_size, plot_plots
                                                  )





rows = []
delta_swst = []
delta_swl = []



raw_slc_offsets = chunk_info[2]

for entry in offsets:
    swath_raw = entry[0]
    swath_name = swath_raw.strip("_")
    slc_id = entry[1]
    chunk_list = entry[2]

    for chunk_idx, chunk_data in enumerate(chunk_list, start=0):
        # Extract existing offsets...
        bottom_tuple = chunk_data[0]
        bottom_az_offset = int(np.round(bottom_tuple[0]))
        bottom_range_offset = int(np.round(bottom_tuple[1]))

        top_tuple = chunk_data[1]
        top_az_offset = int(np.round(top_tuple[0]))
        top_range_offset = int(np.round(top_tuple[1]))

        parameters_tuple = chunk_data[2]
        SWST = float(parameters_tuple[0])
        SWL = float(parameters_tuple[1])
        dt = float(parameters_tuple[2])
        f = float(parameters_tuple[3])
        raw_slc_az_offset = int(parameters_tuple[4])
        raw_slc_rng_offset = int(parameters_tuple[5])

        half_chirp_filter_length = int(np.round(dt * f / 2))
        avg_ncc_az_offset = int(np.round((bottom_az_offset + top_az_offset) / 2))
        avg_ncc_rng_offset = int(
            np.round((bottom_range_offset + top_range_offset) / 2)
        )

        # Calculate SLC dimensions for current chunk
        slc_az_lines = (
            raw_slc_offsets[chunk_idx][1] - raw_slc_offsets[chunk_idx][0]
        )
        slc_range_samples = (
            raw_slc_offsets[chunk_idx][3] - raw_slc_offsets[chunk_idx][2]
        )

        rows.append({
            "Swath": swath_name,
            "SLC_ID": slc_id,
            "Chunk Number": chunk_idx,
            "Bottom Az Offset": bottom_az_offset,
            "Bottom Range Offset": bottom_range_offset,
            "Top Az Offset": top_az_offset,
            "Top Range Offset": top_range_offset,
            "SWST": SWST,
            "SWL": SWL,
            "Pulsewidth": dt,
            "Range Sampling Frequency": f,
            "Azimuth Offset": raw_slc_az_offset,
            "Range Offset": raw_slc_rng_offset,
            "Half Chirp Filter Length": half_chirp_filter_length,
            "Avg NCC Az Offset": avg_ncc_az_offset,
            "Avg NCC Range Offset": avg_ncc_rng_offset,
            "N SLC Az Lines": slc_az_lines,
            "N SLC Range Samples": slc_range_samples,
        })

df = pd.DataFrame(rows)


df['Total Az Offset']    = df['Azimuth Offset'] - df['Avg NCC Az Offset']
df['Total Range Offset'] = df['Range Offset']   - df['Avg NCC Range Offset']

df['Range Start'] = df['Total Range Offset']
df['Range End']   = df['Range Start'] + df['N SLC Range Samples']
#df['Az Start']    = df['Total Az Offset'] + 


# >>>>>> $****** figure out what to do for azimuth offsets and slc_lines per chunk
# Have offsets for each block/chunk. Now extract slc matched raw data
# Need to know number of azimuth lines and range samples in each slc block/chunk


# Number of slc lines in each chunk is:
# Chunk 0 : Number of azimuth lines in the chunk minus the Total Az Offset
# Chunk 1 : Number of total SLC lines minus number of lines in first chunk iff only 2 chunks
# Chunk 1 : else >2 chunks, number of slc lines equals number of azimuth lines in chunk1

# Number of raw samples equals number of slc raw samples





