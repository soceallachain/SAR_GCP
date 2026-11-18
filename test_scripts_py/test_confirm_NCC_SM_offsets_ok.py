#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:23:47 2026

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

#file_path     = f'./tests/test_offsets'

#Set DEBUG mode if do not want to save files
DEBUG = True
# Get list of SLC_IDs

#test_dir   = f'Houston_SM_2019_20'
#test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/jc_data/{test_dir}/'
#root_path  = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'
#test_dir   = f'SaoPaulo_SM_2019_20'
test_dir    = f'SaoPaulo_SM_OCN'
root_path  = f'/mnt/SAR_bucket/data/SaoPaulo_SM/{test_dir}/'

#test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'

pol1        = 'hv'
pol2        = 'vh'
pol         = [pol1, pol2]
SLC_IDs     = os.listdir(root_path)
#swaths      = ['_S1_', '_S3_', '_S6_']
swaths      = ['_S6_','_S3_']


n_files_per_swath = 4
save_fig_path     = f'./tests/test_offsets'

slice_size =[6000,6000]
pol = ['vh', 'hv']

#
plot_plots=False



##### 
## Check new offsets
# Load csv file to dataframe
df = pd.read_csv(f'{save_fig_path}/SaoPaulo_swath_offsets.csv')

slc_id= df['SLC_ID'].iloc[0]

slc_id_path = root_path + slc_id +'/'
parts       = slc_id.split('_')
slc_id_name = f'{parts[0]}_{parts[1]}_{parts[-1]}'
pol1,pol2   = pol
chunk_failed    = False
number_failed_chunks = 0
files       = get_SM_files_from_slc_id(slc_id_path, pol1)
if files == -1:
    print(f'Warning: {slc_id} missing files with pol = {pol1}. Try {pol2}')
    files = get_SM_files_from_slc_id(slc_id_path, pol2)
    if files  == -1:
       print(f'Warning: {slc_id} missing files with pol = {pol2} as well. Definitely missing! ')
raw_file               = files[0]
slc_file               = files[1]
annotation_file        = files[2]
raw_burst_info, l0file = get_SM_raw_burst_info_v2(raw_file)
number_of_chunks       = len(raw_burst_info[0])

print(f'{slc_id} has {number_of_chunks} chunks : {raw_burst_info[0].index[:].to_list()}')

raw_slc_offsets = get_SM_raw_slc_offsets_v2(raw_file, annotation_file)

# Get chunk data and place in slice array
chunk_list     = raw_burst_info[0].index[:].to_list()
chunk_data     = []
chunk_metadata = []
raw_data       = []

for chunk in chunk_list:
    try:
        chunk_metadata.append(l0file.get_acquisition_chunk_metadata(chunk))
        chunk_data.append(l0file.get_acquisition_chunk_data(chunk))

        print(f'{slc_id} Chunk#{chunk} successfully read')
    except:
        print(f'{slc_id} Chunk#{chunk} failed!')
        chunk_metadata.append([])
        chunk_data.append([])
        number_failed_chunks+=1
        chunk_failed = True

if number_failed_chunks == 0:
    for i in range(len(chunk_data)):
        az_start    = int(raw_slc_offsets[i][0])
        az_end      = int(raw_slc_offsets[i][1])
        range_start = int(raw_slc_offsets[i][2])
        range_end   = int(raw_slc_offsets[i][3])
        #raw_data.append(chunk_data[i][az_start:az_end, range_start:range_end])
else:
    print(f'Warning : {slc_id_name} has {number_failed_chunks} failed chunk(s):'
          f' Ignoring and moving to next file'
          )


# Use context manager to properly manage rasterio handles
# Note slc_image is larger than individual raw chunks
with rasterio.open(slc_file) as slc_data:
    slc_image = slc_data.read(1)

n_slc_az_lines      = slc_image.shape[0]
n_slc_range_samples = slc_image.shape[1]
n_raw_az_lines      = sum(chunk_data[i].shape[0] for i in range(len(chunk_data)))
# n_raw_az_lines      = sum(len(chunk) for chunk in chunk_data)
offsets = []
slc_chunk_offsets = get_SM_slc_matched_chunk_dimensions(raw_slc_offsets)

chunk = 1
slc_data = slc_image[slc_chunk_offsets[chunk][0]:slc_chunk_offsets[chunk][1],:]



# Offset near start of chunk
#chunk_slice   =  chunk_data[chunk][0:slice_size[0], 0:slice_size[1]-1]
# Define slice roi
position_shift = [0,0]   # Start of chunk
# Ensure position compatible wih slice_size and slc chunk size

position_shift = np.minimum(position_shift, 
                            np.array(slc_data.shape) - slice_size
                            )




az_off = df['Avg NCC Az Offset'].iloc[chunk]
rg_off = df['Avg NCC Range Offset'].iloc[chunk]

# Add azimuth offset to first 2 elements, range offset to last 2 elements
corrected_raw_slc_offsets = (
    np.array(raw_slc_offsets[chunk]) - np.array([az_off, az_off, rg_off, rg_off])
).tolist()

#corrected_raw_slc_offsets=[]
#corrected_raw_slc_offsets[0] = raw_slc_offsets[chunk][0] + df['Avg NCC Az Offset'].iloc[0]
#corrected_raw_slc_offsets[1] = raw_slc_offsets[chunk][1] + df['Avg NCC Az Offset'].iloc[0]
#corrected_raw_slc_offsets[2] = raw_slc_offsets[chunk][2] + df['Avg NCC Range Offset'].iloc[0]
#corrected_raw_slc_offsets[3] = raw_slc_offsets[chunk][3] + df['Avg NCC Range Offset'].iloc[0]

slice_az_start = corrected_raw_slc_offsets[0] + position_shift[0]
slice_az_end   = corrected_raw_slc_offsets[0] + position_shift[0] + slice_size[0]
slice_rng_start= corrected_raw_slc_offsets[2] + position_shift[1]
slice_rng_end  = corrected_raw_slc_offsets[2] + position_shift[1] + slice_size[1]

chunk_slice    =  chunk_data[chunk][slice_az_start:slice_az_end,
                                    slice_rng_start:slice_rng_end
                                   ]
#chunk_offsets = [0, slice_size[1], 0, slice_size[0]]
# Note: chunk_offsets should be [rng_start, rng_end, az_start, az_end] unlike
# other offset variables which are [az_start, az_end, rng_start, rng_end]
chunk_offsets = [slice_rng_start, slice_size[0],
                 slice_az_start, slice_size[1]
                 ]

meta_data=chunk_metadata
focused_chunk = focus_SM_burst(meta_data[chunk], chunk_slice, chunk_offsets,
                               l0file, 'rma'
                                )
focused_chunk_slice = intensity(focused_chunk)
# plot raw and slc slices using for NCC

#slc_slice           = intensity(slc_data[0:slice_size[0], 0:slice_size[1]-1])
slc_slice           = intensity(slc_data[position_shift[0]:
                                         slice_size[0] + position_shift[0] - 1, 
                                         position_shift[1]:
                                         slice_size[1] + position_shift[1] - 1
                                         ]
                                         )
bottom_offsets      = compute_image_offset(focused_chunk_slice,
                                           slc_slice, plot_surface=True
                                           )


plot_2d_array(focused_chunk_slice,title='Top RDA Focused Slice')
plot_2d_array(slc_slice,title='Top SLC Slice')


