#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th 2026
 ******************   test_SM_chunk_offsets.py  *************************************
Code to:
    - Read SM Raw and SLC files
    - Read raw chunks into chunk array
    - For each chunk
               - take NxN slice of raw and slc from start and end
               - run NCC on each pair and record offsets


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
from modules.SAR_focusing_module import *
from modules.image_processing_module import *
from modules.chunk_processing_module import *
from scipy.ndimage import zoom

#Set DEBUG mode if do not want to save files
DEBUG = True
# Get list of SLC_IDs

#test_dir   = f'Houston_SM_2019_20'
#test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/jc_data/{test_dir}/'
#root_path  = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'
#test_dir   = f'SaoPaulo_SM_2019_20'
#root_path  = f'/mnt/SAR_bucket/data/SaoPaulo_SM/{test_dir}/'

test_dir   = f'Houston_SM_No_OCN'
root_path  = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'

pol1        = 'hv'
pol2        = 'vh'
pol         = [pol1, pol2]
SLC_IDs     = os.listdir(root_path)
#swaths      = ['_S1_', '_S3_', '_S6_']
swaths      = ['_S6_','_S3_']


n_files_per_swath = 150
save_fig_path     = f'./tests/test_offsets'

slice_size =[6000,6000]
pol = ['vh', 'hv']

#
plot_plots=False
swath_offsets =  process_files_for_chunk_offsets(SLC_IDs,root_path,swaths,
                                                  n_files_per_swath, pol, 
                                                  slice_size, plot_plots
                                                  )

rows = []
delta_swst = []
delta_swl = []


for entry in swath_offsets:
    swath_raw = entry[0]
    # Clean '_S6_' to 'S6'
    swath_name = swath_raw.strip('_')
    slc_id = entry[1]
    chunk_list = entry[2]
    
    # Enumerate chunks starting at index 1
    for chunk_idx, chunk_data in enumerate(chunk_list, start=0):
        bottom_tuple = chunk_data[0]        
        bottom_az_offset    = int(np.round(bottom_tuple[0]))
        bottom_range_offset = int(np.round(bottom_tuple[1]))
        # Extract the bottom offset tuple (index 1)
        top_tuple        = chunk_data[1]
        top_az_offset    = int(np.round(top_tuple[0]))
        top_range_offset = int(np.round(top_tuple[1]))
        
        parameters_tuple = chunk_data[2]
        SWST = float(parameters_tuple[0])
        SWL  = float(parameters_tuple[1])
        dt   = float(parameters_tuple[2]) # Pulse length
        f    = float(parameters_tuple[3]) # Sampling frequency
        raw_slc_az_offset = int(parameters_tuple[4])
        raw_slc_rng_offset= int(parameters_tuple[5])
        half_chirp_filter_length = int(np.round(dt*f/2))
        avg_ncc_az_offset = int(np.round((bottom_az_offset + top_az_offset)/2))
        avg_ncc_rng_offset = int(np.round((bottom_range_offset + top_range_offset)/2))
        
        rows.append({
            'Swath': swath_name,
            'SLC_ID': slc_id,
            'Chunk Number': chunk_idx,
            'Bottom Az Offset': bottom_az_offset,
            'Bottom Range Offset': bottom_range_offset,
            'Top Az Offset': top_az_offset,
            'Top Range Offset':top_range_offset,
            'SWST':SWST,
            'SWL':SWL,
            'Pulsewidth':dt,
            'Range Sampling Frequency':f,
            'Azimuth Offset': raw_slc_az_offset,
            'Range Offset' : raw_slc_rng_offset,
            'Half Chirp Filter Length':half_chirp_filter_length,
            'Avg NCC Az Offset': avg_ncc_az_offset,
            'Avg NCC Range Offset': avg_ncc_rng_offset
        })
    
    for i in range(len(chunk_list)):
        if len(chunk_list)>1:
            if i == len(chunk_list)-1:
                delta_swst.append(0)
                delta_swl.append(0)
            else:
                delta_swst.append(rows[i]['SWST'] - rows[i+1]['SWST'])
                delta_swl.append( rows[i]['SWL']  - rows[i+1]['SWL'])
        else:
            delta_swst.append(0)
            delta_swl.append(0)
                
                
                
                
# Create the DataFrame
df = pd.DataFrame(rows)
df['Delta SWST'] = delta_swst
df['Delta SWL']  = delta_swl
df['Delta SWST Range Offset'] = np.round(df['Range Sampling Frequency'] * df['Delta SWST']).astype(int)
#df['Delta SWST Range Offset']=int(np.round(df['Range Sampling Frequency']*df['Delta SWST']))
df.to_csv(f'{save_fig_path}/Houston_swath_offsets.csv', index=False)

# Can collate results using: parse_SM_offsetsw_tests_results.py
%run ./parse_SM_offsetsw_tests_results.py
