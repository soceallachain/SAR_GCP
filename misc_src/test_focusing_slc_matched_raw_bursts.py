#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:59:54 2025

@author: joe
"""

import numpy as np
#import matplotlib as plt
from skimage.measure import find_contours, label
import itertools
from SAR_code_module import *
import platform
import psutil
import os

from SAR_focusing_module import *
from pypapi import events, papi_high
import time
import pickle



def place_roi(A: np.ndarray, roi_shape: tuple, p: tuple):
    """
    Place a region of interest (ROI) of given shape centrally at location p in 2D numpy array A.
    If ROI extends beyond array edges, position it to include the edge fully.

    Args:
        A (np.ndarray): 2D numpy array
        roi_shape (tuple): (num_cols, num_rows) size of region of interest
        p (tuple): (row, col) coordinate where ROI is to be centered

    Returns:
        starting_row (int): starting row index of ROI within A
        starting_col (int): starting column index of ROI within A
    """

    num_rows, num_cols = roi_shape  # note: roi_shape is (num_cols, num_rows) as per your problem
    total_rows, total_cols = A.shape

    row_c, col_c = p  # The location p is (row, col)

    # Calculate half sizes (use integer division)
    half_rows = num_rows // 2
    half_cols = num_cols // 2

    # Starting row and col with +1 as in your example
    starting_row = row_c - half_rows + 1
    starting_col = col_c - half_cols + 1

    # Correct start if ROI extends beyond top/left edges
    if starting_row < 0:
        starting_row = 0
    if starting_col < 0:
        starting_col = 0

    # Correct start if ROI extends beyond bottom/right edges
    if starting_row + num_rows > total_rows:
        starting_row = total_rows - num_rows
    if starting_col + num_cols > total_cols:
        starting_col = total_cols - num_cols

    # Ensure no negative indices after correction (in case ROI bigger than array)
    starting_row = max(0, starting_row)
    starting_col = max(0, starting_col)

    return int(starting_row), int(starting_col)


raw_path =f'/home/joe/Work/RSAR/Test/sampleData/Public/raw_data'
slc_path =f'/home/joe/Work/RSAR/Test/sampleData/Public/slc_data'
slc_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/Public/slc_burst_data'

#burst_id = f'S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794'
burst_id = f'S1A_IW_SLC__1SDV_20200221T051410_20200221T051437_031346_039B7B_EB4B'
burst_number = 6
crc_burst = f"{burst_id.split('_')[9]}_burst#{burst_number}"
slc_burst_file = f'{burst_id}_RAW_IW2_slc_burst#{burst_number}.npy'
slc_burst =np.load(f'{slc_burst_dir}/{slc_burst_file}')

#Get ship locations in SLC file
path = '/home/joe/Work/Projects/RSAR/Test/SampleData/Public'
csv_path = f'{path}/Reports/'
img_path='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results'

#  Read CSV
csv_file = os.path.join(csv_path, 'Public_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships.csv')
burst_info = get_slc_id_burst_and_chip_numbers(slc_burst_file)
burst_ship_locations = get_ship_locations_in_bursts(burst_info, csv_file)
ship_locations =burst_ship_locations[['ship_burst_row',
                                      'ship_burst_col']
                                     ].to_numpy()


data = get_raw_burst_with_offset_info(burst_id, burst_number)

#An slc-matched-raw-burst is one that has dimensions that match with the 
#equivalent slc file ensuring de-burstment and offsets aligned

slc_matched_raw_burst = data[0]
raw_burst    = data[1]
az_offset    = data[2]
az_end       = data[3]
range_offset = data[4]
range_end    = data[5]
meta_data    = data[6]
p_l0file     = data[7]

range_length   = range_end-range_offset
azimuth_length = az_end-az_offset
roi = np.array([range_offset, range_length, az_offset, azimuth_length])

#Plot de-bursted SLC burst from 
img_name = f'SLC_{crc_burst}'
f=plot_2d_array(np.square(np.abs(slc_burst)), img_name)
f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
# Plot RAW burst
img_name = f'SLC_matched_Raw_Burst_Real_{crc_burst}'
f=plot_2d_array(np.real(slc_matched_raw_burst),img_name,'linear')
f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)

# Want to focus raw burst that matched SLC timings
rd = focus_SM_burst(meta_data, slc_matched_raw_burst, roi, p_l0file, 'rma' )
img_name = f'Full_focus_{crc_burst}'
f=plot_2d_array(np.square(np.abs(rd)),img_name)
f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)


#range compression
rd = focus_SM_burst(meta_data, slc_matched_raw_burst, roi, p_l0file, 'r' )
img_name = f'Range Compression_only_{crc_burst}'
f=plot_2d_array(np.abs(rd),img_name)
f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)

#Plot areas around ships.
#Place ship in centre and then offset by fraction of crop size
crop_size = [512, 512]
padded_size = 4096

for i in range(0,len(ship_locations)):
#for i in range(0,1):
    ship_location= ship_locations[i]
    #6974 ship1 @ [280,10300];ship2@ [275,16250]
    # Position crop around ship location

    r,c = place_roi(slc_matched_raw_burst, crop_size, ship_location)    
    #Note: Need range and azimuth offsets as need to match with timings that 
    #      are all with respect to original raw burst
    new_roi = np.array([range_offset + c, crop_size[1], 
                        az_offset    + r, crop_size[0]
                        ])
    cropped_raw_burst= slc_matched_raw_burst[r:r + crop_size[0],
                                             c:c + crop_size[1]]
    rdd = focus_SM_burst(meta_data, cropped_raw_burst, new_roi, p_l0file, 'rma' )

    img_name = f"Raw_Burst_Real_Cropped_around_ship_{ship_location}_{crc_burst}"
    f=plot_2d_array(np.real(cropped_raw_burst),img_name,'linear')
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    img_name =f'Full_focus_Cropped_around_ship{ship_location}__{crc_burst}'
    f=plot_2d_array(np.square(np.abs(rdd)),img_name)
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    
    

#Very crop sizes
ship_location=ship_locations[4]
crop_size=[1024,1024]
#crop_size = [512, 1024]
padded_range_size = 4096
padded_az_size    = 1024

#for i in range(0,len(ship_locations)):
for i in range(0,1):
    ship_location= ship_locations[0]
    #6974 ship1 @ [280,10300];ship2@ [275,16250]
    # Position crop around ship location
    ship_offset =np.array([0,0])
    ship_location = ship_location + ship_offset
    r,c = place_roi(slc_matched_raw_burst, crop_size, ship_location)    
    #Note: Need range and azimuth offsets as need to match with timings that 
    #      are all with respect to original raw burst
    new_roi = np.array([range_offset + c, crop_size[1], 
                        az_offset    + r, crop_size[0]
                        ])
    cropped_raw_burst= slc_matched_raw_burst[r:r + crop_size[0],
                                             c:c + crop_size[1]]
    rdd = focus_SM_burst(meta_data, cropped_raw_burst, new_roi, p_l0file, 'rma' )

    img_name = f"Raw_Burst_Real_Cropped_around_ship_{ship_location}__{crop_size}_{crc_burst}"
    f=plot_2d_array(np.real(cropped_raw_burst),img_name,'linear')
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    img_name =f'Full_focus_Cropped_around_ship{ship_location}__crop_size_{crop_size}_{crc_burst}'
    f=plot_2d_array(np.square(np.abs(rdd)),img_name)
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    
 
