#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:36:18 2026

@author: joe
"""
###############################################################################
##
## Save raw and slc data to npy files
## Need large block of data of size:
##
##
###############################################################################

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
save_path     = f'/mnt/SAR_bucket/data/sand_box/rc_testing/files_for_iain/'
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
    # Reconstruct slc_block_dims to accound for any azimuth offset from NCC
    slc_block_dims = [[0,
                       slc_block_indices[0][1] - slc_block_indices[0][0]
                      ],
                      [slc_block_indices[0][1] - slc_block_indices[0][0] + 1, 
                       slc_block_indices[1][1] - slc_block_indices[0][0]
                      ]
                     ]

#Load Slc file
slc_filename = files[1]
slc_annotation_file = files[2]

# Ascending or Desceding Node
orb_node = get_orbital_node(slc_annotation_file)
if orb_node =='Ascending':
    node = 'asc'
elif orb_node == 'Descending':
    node ='desc'
else:
    node = 'none'

# Want to save slc matched raw blocks and slc
# Need slc_aoi for each block. Get from slc_block_indices()
block = 0
slc_start_row  = 0
slc_end_row    = slc_block_indices[block][1]-slc_block_indices[block][0]
slc_start_col  = 0
slc_end_col    = slc_block_indices[block][3]-slc_block_indices[block][2]
slc_block0_aoi = [slc_start_row,slc_end_row, slc_start_col, slc_end_col]

block = 1
slc_start_row = slc_end_row + 1
slc_end_row   = slc_start_row + slc_block_indices[block][1]-slc_block_indices[block][0]

slc_end_col    = slc_block_indices[block][3]-slc_block_indices[block][2]
slc_block1_aoi = [slc_start_row,slc_end_row, slc_start_col, slc_end_col]
# Save raw blocks
raw_block0_aoi,sb0 = get_SM_slc_matched_raw_AOIs(slc_block0_aoi, slc_block_indices,
                                slc_block_dims)
raw_block1_aoi, sb1 = get_SM_slc_matched_raw_AOIs(slc_block1_aoi, slc_block_indices,
                                slc_block_dims)

block_raw_data = chunk_info[0]

raw_block0_aoi = np.squeeze(raw_block0_aoi)

raw_block0_aoi_data = block_raw_data[0][raw_block0_aoi[0]:raw_block0_aoi[1],
                                        raw_block0_aoi[2]:raw_block0_aoi[3]
                                        ]
raw_block1_aoi = np.squeeze(raw_block1_aoi)

raw_block1_aoi_data = block_raw_data[0][raw_block1_aoi[0]:raw_block1_aoi[1],
                                        raw_block1_aoi[2]:raw_block1_aoi[3]
                                        ]

# Save raw blocks
np.save(f'{save_path}{slc_id}_Pol_{pol}_Node_{node}_raw_'
        f'block0.npy', raw_block0_aoi_data)


np.save(f'{save_path}{slc_id}_Pol_{pol}_Node_{node}_raw_'
        f'block1.npy', raw_block1_aoi_data)



# Save equivalent slc_blocks
slc_block0_aoi_data = get_SM_slc_aoi(slc_block0_aoi,files[1])
slc_block1_aoi_data = get_SM_slc_aoi(slc_block1_aoi,files[1])
 

# Save raw blocks
np.save(f'{save_path}{slc_id}_Pol_{pol}_Node_{node}_slc_'
        f'block0.npy', slc_block0_aoi_data)


np.save(f'{save_path}{slc_id}_Pol_{pol}_Node_{node}_slc_'
        f'block1.npy', slc_block1_aoi_data)

# Save equivalnet SLC aoi

raw_aoi=raw_block0_aoi_data[0:7296,0:19712]
np.save(f'{save_path}{slc_id}_Pol_{pol}_Node_{node}_raw_'
        f'AOI_7296x19712.npy', slc_aoi)
slc_aoi=slc_block0_aoi_data[0:7296,0:19712]
np.save(f'{save_path}{slc_id}_Pol_{pol}_Node_{node}_slc_'
        f'AOI_7296x19712.npy', raw_aoi)




# Downsample AOI
# Does not appear to work well????
slc_aoi_downsample= slc_aoi[::10, ::10].copy()
# Plot slc, abs(raw) & np.real(raw)

# Range compress raw with no padding

# Range compress raw aoi with padding
