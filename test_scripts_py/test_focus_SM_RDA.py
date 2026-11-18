#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:57:13 2026

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

test_dir   = f'Houston_SM_No_OCN'

root_path     = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'
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
   
   
save_path    = f'/mnt/SAR_bucket/data/sand_box/rc_testing/timing_results/'


# Function to accept an AOI in an SLC file in [row,col]
# co-ordinates and focus the equivalent raw data and plot
# the original raw and slc data and the focused raw data 
# and returns the focused raw data
#
# Inputs : slc_aoi  -- [row_start,row_end,col_start,col_end]
#          files    -- list of raw, slc and annotation file
#
# Outputs: rda_aoi  -- RDA focused aoi data
#
def focus_RDA_SM_aoi(slc_aoi, files, pad_with ='Zeroes',
                     plot_plots=False,
                     azimuth_padding=1024, range_padding=1024
                     ):
    # Get offsets between SLC and raw data
    chunk_info  = process_file_for_chunks(files)
    timing_based_offsets = chunk_info[2]
    slc_block_dims  = get_SM_slc_matched_chunk_dimensions(timing_based_offsets)
    ncc_offsets     = process_chunks_for_offsets(chunk_info, slice_size, False)
    slc_block_indices = get_SM_slc_block_offsets(ncc_offsets,
                                            timing_based_offsets
                                            )  
    
    # Need to fix this for cases where >2 blocks
    slc_block_dims=[]
    for block in range(len(slc_block_indices)):
        if block ==0:
            slc_block_dims.append([0, slc_block_indices[0][1] - slc_block_indices[0][0] ])
        else:
            slc_block_dims.append([slc_block_dims[block-1][1]+1, 
                        slc_block_indices[block][1]- slc_block_indices[0][0]
                                 ])
    
 
    #slc_block_dims = [[0, slc_block_indices[0][1] - slc_block_indices[0][0]],
    #              [slc_block_indices[0][1] - slc_block_indices[0][0] + 1,
    #               slc_block_indices[1][1] - slc_block_indices[0][0]]
    #             ]

    slc_aoi_start_row = slc_aoi[0]
    slc_aoi_end_row   = slc_aoi[1]
    slc_aoi_start_col = slc_aoi[2]
    slc_aoi_end_col   = slc_aoi[3]
    
    # Currently only dealing with cases where slc_aoi is within one block
    # If spans more than one block then tell user to get ... chose a different
    # aoi as otherwise have to focus 2 separate raw aois
    aoi_block = -1
    for block in range(len(slc_block_dims)):   
        b_start, b_end = slc_block_dims[block]
        if b_start <=slc_aoi_start_row < slc_aoi_end_row <= b_end:
            aoi_block = block
            break
        
    if aoi_block == -1:
        print(f' SLC AOI spans 2 raw blocks - not currently supported.')
        print(f' Choose AOI with rows within {slc_block_dims}')
        return -1
    
    
    # Get AOI in raw that matches with SLC
    raw_aoi, sb = get_SM_slc_matched_raw_AOIs(slc_aoi,
                                              slc_block_indices, 
                                              slc_block_dims
                                              )
    
    # Get data for whole block - probably not very efficient!
    block_raw_data      = chunk_info[0]
    n_raw_az_lines      = block_raw_data[aoi_block].shape[0]
    n_raw_range_samples = block_raw_data[aoi_block].shape[1]
    raw_aoi        = np.squeeze(raw_aoi)
    raw_aoi_data   = block_raw_data[aoi_block][raw_aoi[0]:raw_aoi[1],
                                               raw_aoi[2]:raw_aoi[3]
                                               ]
    # Get meta and ephemerides data for block
    # Needed for full focusing
    block_metadata = chunk_info[1][block]
    l0file_block   = chunk_info[4]
    
    # Get SLC data
    slc_block_aoi_data    = get_SM_slc_aoi(slc_aoi,files[1])
    range_start_padding   = range_end_padding  = range_padding
    azimuth_start_padding = azimuth_end_padding = azimuth_padding
    # Add padding to raw data. Given the slc_aoi position does the 
    # padding fit? If not choose smallest value that fits and use at both
    # ends
    if azimuth_padding > raw_aoi[0]:
        azimuth_start_padding = raw_aoi[0]
    if azimuth_padding > n_raw_az_lines -raw_aoi[1]:
        azimuth_end_padding = n_raw_az_lines -raw_aoi[1]
        
    if range_padding > raw_aoi[2]:
        range_start_padding = raw_aoi[2]
    if range_padding > n_raw_range_samples - raw_aoi[3]:
        range_end_padding = n_raw_range_samples - raw_aoi[3]
    
    rng_padding = np.minimum(range_end_padding,range_start_padding)
    az_padding  = np.minimum(azimuth_end_padding, azimuth_start_padding)
    
    az_start_row  = raw_aoi[0] - az_padding
    az_end_row    = raw_aoi[1] + az_padding
    rng_start_col = raw_aoi[2] - rng_padding
    rng_end_col   = raw_aoi[3] + rng_padding
    
    if pad_with == 'Zeroes':
        raw_roi_padded_data = np.pad(raw_aoi_data,
                                     ((az_padding, az_padding), 
                                      (rng_padding, rng_padding)),
                                     mode="constant", constant_values=0
                                      )
    else: 
        # Pad with actual data
        raw_roi_padded_data = block_raw_data[aoi_block][az_start_row :az_end_row,
                                                        rng_start_col:rng_end_col
                                                       ]
    # Note raw_roi is  [range_start, range_length, az_start, az_length]
    raw_roi       = [rng_start_col, (rng_end_col - rng_start_col),
                     az_start_row, (az_end_row - az_start_row)
                     ]
    rda_padded_data = focus_SM_burst(block_metadata,
                                    raw_roi_padded_data,
                                    raw_roi,
                                    l0file_block,
                                   'rma'
                                    )
    
    rda_data = rda_padded_data[az_padding : -az_padding,
                               rng_padding: -rng_padding]
    
    print(f' Completed focusing ...... plotting ...')

    if plot_plots:
        slc_file = files[1]
        parts=slc_file.split('/')[6].split('_')
        plot_title=f'{parts[0]}_{parts[1]}_{parts[9]}'
        plot_2d_array_terminal(intensity(rda_data),
                  title=f'RDA Focused Raw {slc_aoi} : {plot_title}')
        
        print(f'******** Completed focusing and plotted RDA ***************')

        plot_2d_array_terminal(intensity(slc_block_aoi_data),
                      f'SLC Data AOI {slc_aoi} : {plot_title}')
        plot_2d_array_terminal(np.real(raw_aoi_data),
                      f'Raw AOI {slc_aoi} : {plot_title}')

        print(f' Function finished .....')   
        plt.show(block=True)

    
    return rda_data




slc_roi=[0,6000,14000,18000]
focused_data=focus_RDA_SM_aoi(slc_roi, files, pad_with ='Zeroes',
                     plot_plots=True, azimuth_padding=1024, range_padding=1024
                     )
