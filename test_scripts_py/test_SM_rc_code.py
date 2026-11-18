#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:37:46 2026

@author: joe
"""

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
save_path     = f'/home/joe/SAR_GCP/tests/slc_raw_matching/'
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


# Define how big a region to search to cross correlate over for matching
slice_size = [5000, 5000] 

plot_plots = False
# Test if FFT in azimuth needed for Range COmpression

#Load raw file
slc_id= SLC_IDs[500]
slc_id_path = root_path + slc_id +'/'

for pol in pols:  
   files       = get_SM_files_from_slc_id(slc_id_path, pol)
   if files != -1:
       break  # Found files
if files == -1:
    print(f'Warning: {slc_id}:  Files not found ')      
else:
    # Get block raw data and metadata, and time-based-offsets
    scene_files = files
    chunk_info  = process_file_for_chunks(scene_files)
    timing_based_offsets = chunk_info[2]
    slc_block_dims  = get_SM_slc_matched_chunk_dimensions(timing_based_offsets)
    ncc_offsets = process_chunks_for_offsets(chunk_info, slice_size, plot_plots)
    slc_block_indices = get_SM_slc_block_offsets(ncc_offsets,
                                                timing_based_offsets
                                                )

#Load Slc file
slc_filename = files[1]
slc_annotation_file = files[2]

# Choose AOI

# Slc AOI in gps co-ordinates  [[lon_SW,lat_SW],[lon_NE, lat_NE]]
slc_gps_aoi = [[-94.870,29.128], [-94.727, 29.358]]
slc_cartesian_aoi = get_SM_cartesian_aoi(slc_gps_aoi, slc_annotation_file)

slc_aoi_data = get_SM_slc_aoi(slc_cartesian_aoi,slc_filename)

## Plot slc file
# Get title
parts = slc_id.split('_')
title = f'{parts[1]}_{parts[9]}_SLC_AOI_{slc_gps_aoi[0]}->{slc_gps_aoi[1]}'

fig_ax = plot_2d_array(intensity(slc_aoi_data),title,'log_normal',
                                 'Azimuth','Range'
                                 )

## Plot raw
# Get slc matched raw aoi
raw_aoi_info = get_SM_slc_matched_raw_AOIs(slc_cartesian_aoi, 
                   slc_block_indices,
                   slc_block_dims
                   )
slc_matched_raw_aoi = raw_aoi_info[0]

# Get raw data  
chunk_raw_data = chunk_info[0]
raw_aoi_data_list=[]
for i in range(len(slc_matched_raw_aoi)):
    raw_aoi_data_list.append(chunk_raw_data[i][
                     slc_matched_raw_aoi[i][0]:slc_matched_raw_aoi[i][1],
                     slc_matched_raw_aoi[i][2]:slc_matched_raw_aoi[i][3]
                                               ]
                              )
raw_aoi_data =np.concatenate( np.array(raw_aoi_data_list), axis=0) 



plot_2d_array(np.real(raw_aoi_data),f'Raw {title}', 'log_normal','Azimuth', 'Range')
# Do Range Compression with existing code
sm_swath = int(slc_id.split('_')[1][-1])
# Setup timer
import time
start = time.perf_counter()

# Range compress


#rc_data = range_compress_burst(raw_aoi_data,'S1SM',sm_swath )
rc_data = range_compress_padded_raw_data(raw_aoi_data,'S1SM',sm_swath,1024,512 )
# End timer
elapsed = time.perf_counter() - start
print(f"Execution time: {elapsed:.6f} seconds")
#  Plot Range compressed data

# Get title
parts = slc_id.split('_')
title = f'{parts[0]}_{parts[1]}_{parts[9]}'

plot_2d_array(intensity(rc_data),f'{title} RC', 'log_normal','Range','Azimuth')


# Do Range Compression with revised code


# Plot range compressed data


# Compare pre and post

#Do full focusing:

# What block/chunk?
start_block = raw_aoi_info[1]
# How may blocks spanned?
n_spanned_blocks = len(raw_aoi_info[0])
for i in range(n_spanned_blocks):
    # focus data from each block in turn
    # get meta data for start block
    block = start_block + i
    block_meta_data = chunk_info[1][block]
    l0file = chunk_info[4]
    # Get block_offsets 
    offsets    = raw_aoi_info[0][block]
    n_aoi_cols = raw_aoi_data.shape[1]
    n_aoi_rows = raw_aoi_data.shape[0]
    block_offsets =[offsets[2],n_aoi_cols, offsets[0],n_aoi_rows]
    focused_raw_aoi = focus_SM_burst(block_meta_data,
                                 raw_aoi_data,
                                 block_offsets,
                                 l0file,
                                 'r'
                                 )

plot_2d_array(intensity(focused_raw_aoi),f'{title} RC','log_normal','Azimuth','Range')
                                 
