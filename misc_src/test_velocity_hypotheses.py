#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:44:09 2025

@author: joe
"""


import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

#from SAR_focusing_module import *
from blob_module import *

from SAR_code_module import *




plt.style.use('dark_background')
# Different range sampling frequencies in different swaths for RAW and SLC 
# Need to re-sample SLC data to RAW range sampling frequency using these:
freq_multipliers=[1, 54.595959/64.345238, 46.915402/64.345238]

test_dir = (f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/'
            f'test_velocity_hypothesis/')
slc_dir  = f'{test_dir}SLC'
raw_dir  = f'{test_dir}RAW'

#SAFE_dir =f'{slc_dir}/S1A_IW_SLC__1SDV_20251014T065603_20251014T065630_061418_07AA7D_6ACF.SAFE'
#safe_file = (f'S1A_IW_SLC__1SDV_20251014T065628_20251014T065655_'
#           f'061418_07AA7D_192C.SAFE')
# Adriatic
#safe_file  =  (f'S1C_IW_SLC__1SDV_20251016T165717_20251016T165744_004590_00913B_CFBC.SAFE')
# Ireland SW coast
#safe_file  = (f'S1A_IW_SLC__1SDV_20251029T183055_20251029T183122_061644_07B38D_E59B.SAFE')
# Angola West Coast  :- Different Dates/Times
#safe_file   = (f'S1A_IW_SLC__1SDV_20251025T045219_20251025T045246_061577_07B0E2_7D24.SAFE')
#safe_file   = (f'S1A_IW_SLC__1SDV_20250919T045219_20250919T045246_061052_079BC2_6903.SAFE')
safe_file   =(f'S1A_IW_SLC__1SDV_20250814T045218_20250814T045245_060527_0786E7_A4E8.SAFE')


pol             = 'vh'
# set location want to find: GPS co-ordinates Note: [long, lat] Format required
#location =  [-9.602705, 51.389244]    # Fastnet Lighthouse
#location  = [13.781544, 44.077254]    # Off Ancona
#location =  [12.344848, 44.392972]    # Angela Cluster  (Off Ancona)
#location =  [14.379206, 42.808505]    # Emma Ovest (Off Ancona)
#location =  [12.753326, 44.131039]     # Giulia Monotubolare
#location =   [12.847415, 44.689088]    # Naomi-Pandora
#location =   [-10.5500, 51.2166]       # M3 Buoy, off Cork
#location =   [11.15232, -6.35207]       # Kizmomba B, Angola
location =   [11.05270, -6.31976]       # Kizmomba B +11.5km, Angola

#Execute code to get RAW, SLC and swath data
slc_SAFE_dir =(f'{slc_dir}/{safe_file}')
filenames_and_swath = get_raw_slc_annotation_filenames(raw_dir, slc_SAFE_dir, location, pol)
fig, burst_number = plot_raw_slc_data(filenames_and_swath, location, pol)

swath_index = filenames_and_swath[3]
fig.savefig(f"{test_dir}/{safe_file.split('.')[0]}_IW{swath_index +1}_Burst#{burst_number}.jpg", format='jpg',
         bbox_inches='tight', dpi=600
         )
#fig.clf()


# draw around both locations !!!

# extract roi around location in slc and raw and plot