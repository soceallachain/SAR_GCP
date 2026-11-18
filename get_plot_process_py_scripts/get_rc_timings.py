#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 06:11:01 2026

@author: joe
"""


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

import time

##############################################################################
## Determine RC timings for various data sizes 


root_path     = f'/mnt/SAR_bucket/data/sand_box/rc_testing/files_for_iain/'
filename     = f'S1B_S3_SLC__1SDH_20180214T001739_20180214T001802_009612_011540_E777_Pol_hv_Node_asc_raw_block0.npy'
npy_file     = f'{root_path}{filename}'


# Load raw data
raw_data = np.load(npy_file)


azimuth_sizes = [1024, 2048, 4096, 8192,16392]
range_sizes   = [1024, 2048, 4096, 8192,16392]

# Sweep over all sizes and record time for range compression

# All ships  for this particular file at end of range
sm_swath = 3
range_padding   = 0
azimuth_padding = 0
az_start_row    = 0
rng_end_col     = raw_data.shape[1] 

for az_size in azimuth_sizes:
    for rng_size in range_sizes:
        az_end_row    = az_start_row + az_size 
        rng_start_col = rng_end_col - rng_size
        aoi_data      = raw_data[az_start_row:az_end_row,
                                 rng_start_col:rng_end_col
                                 ]
        # Get average of 10
        elapsed_time = []
        for j in range(0:10):
            
           # Start timeer
           start   = time.perf_counter()
           rc_data = range_compress_padded_raw_data(aoi_data,
                                                 'S1SM',
                                                 sm_swath,
                                                 range_padding,
                                                 azimuth_padding
                                                 )
           elapsed = time.perf_counter() -start
           elapsed_time.append(elapsed)
        
        min_time = elapsed_time.min()
        max_time  = elpased_time.max()
        mean_time = elapsed_time.mean()
        
           
        print(f' Azimuth x Range of {az_size} x {rng_size} takes {mean_time} '
              f' seconds for range compression with Range Padding of '
              f' {range_padding}'
               )
    
        
        
        
        



        










