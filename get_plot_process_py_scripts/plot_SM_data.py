#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:35:31 2026

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


# load slc AOI
slc_dir = f'{save_path}{slc_burst_dir}/'
slc_filename = 'S1B_S3_SLC__1SDH_20180214T001739_20180214T001802_009612_011540_E777_Pol_hv_Node_asc_slc_C.npy'

raw_dir = f'{save_path}{raw_burst_dir}/'
raw_filename = f'S1B_S3_SLC__1SDH_20180214T001739_20180214T001802_009612_011540_E777_Pol_hv_Node_asc_raw_C.npy'
slc_file=f'{slc_dir}{slc_filename}'
raw_file = f'{raw_dir}{raw_filename}'


slc_aoi_C = np.load(slc_file)
raw_AOI_C = np.load(raw_file)


# Get title
full_path=Path(slc_file)
filename=full_path.stem
parts=filename.split('_')
title = f'{parts[1]}_{parts[9]}_{parts[15]}'


plot_2d_array(intensity(slc_aoi_C),f'{title} SLC', 'log_normal','Range','Azimuth')
plot_2d_array(np.real(raw_AOI_C),f'{title} Raw', 'linear', 'Range', 'Azimuth')

plot_2d_array(np.real(raw_AOI_C),'linear')
