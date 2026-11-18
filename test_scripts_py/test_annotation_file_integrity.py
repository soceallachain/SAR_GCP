#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:13:55 2026

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
save_path     = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/'

# Vm cannot write to GCP for some reason
#save_path     = f'/home/joe/SAR_GCP/tests/slc_raw_matching/'
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

# Define AOIs by SW & NE co-ordinates
# ...now [long, lat]
S3_A = [(-94.9837, 29.0670),  (-94.8470, 29.1525)]
S3_B = [(-94.8306, 29.0862),  (-94.6690, 29.2512)]
S3_C = [(-94.6358, 29.1167),  (-94.5778, 29.3966)]
S3_scenes = [S3_A, S3_B, S3_C]

S6_A = [(-94.9837, 29.0670),  (-94.8470, 29.1525)]
S6_B = [(-94.8306, 29.0862),  (-94.6690, 29.2512)]
S6_C = [(-94.6358, 29.1167),  (-94.4837, 29.4108)]
S6_scenes = [S6_A, S6_B, S6_C]




slice_size = [5000, 5000]
plot_plots = False
n_files_per_swath = 1


#2. Get single file
#slc_id = SLC_IDs[299]  #S1A_S6_SLC__1SDH_20160611T002621_20160611T002645_011656_011D70_F9F2


#nS3 = 0

# Do for single SLC_ID
#for slc_id in [SLC_IDs[299]]:
# Do for all SLC_IDs
for slc_id in SLC_IDs:
    if '_S3_' not in slc_id and '_S6_' not in slc_id:
        continue
    elif '_S3_' in slc_id:
        swath = 'S3'
        swath_scenes = S3_scenes
    elif '_S6_' in slc_id:
        swath = 'S6'
        swath_scenes = S6_scenes

    slc_id_path = root_path + slc_id +'/'
    for pol in pols:  
       files       = get_SM_files_from_slc_id(slc_id_path, pol)
       if files != -1:
           break  # Found files
    if files == -1:
        print(f'Warning: {slc_id}:  Files not found ')      
        continue
    else:
        # Test that annotation files not corrupted
        slc_annotation_file =files[2]
        slc_burst_info     = get_SM_GridPoints(slc_annotation_file)

        # Handles numeric 0, string '0', or mixed types
        exists = (
           (slc_burst_info["line"].astype(str) == "0")
           & (slc_burst_info["pixel"].astype(str) == "0")
           ).any()
        if not exists:
            print(f'{slc_id} : Annotation File Corrupted with bad line 0')
