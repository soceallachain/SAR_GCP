#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:34:59 2025

@author: joe
"""
# Script to find raw bursts that are needed for particular 
# .....
import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from SAR_focusing_module import *
from blob_module import *

from SAR_code_module import *

source = 'ext'
Dataset = 'Train'
sar_mode = 'RAW'
swath_of_interest = 2

non_ship_lower_size = 50
non_ship_upper_size = 100

#Open csv file
csv_dir      =f'/home/joe/Work/Projects/RSAR/Test/SampleData/Train/Reports/'
csv_filename =f'SARFish_{sar_mode}_{Dataset}_IW{swath_of_interest}_BurstInfoShipsPerBurstWithNonShipLocations.csv'
df         = pd.read_csv(f'{csv_dir}{csv_filename}')


# Find all non-ships between lower and upper size range - save as list
df1 = df[df['ship_length']>non_ship_lower_size]
df_gtxm = df1[df1['ship_length'] <non_ship_upper_size]
gt_xm_real_bursts_series = df_gtxm['Burst_Name'].str.split('_').str[:14].str.join('_') + '.npy'
gt_xm_imag_bursts_series = gt_xm_real_bursts_series.str.replace('real', 'imag')
gt_xm_slc_bursts_series  = gt_xm_real_bursts_series.str.replace('real', 'slc')

gt_xm_raw_bursts_series  = pd.concat([gt_xm_real_bursts_series, gt_xm_imag_bursts_series], ignore_index=True)

output_filename = f'Non_Ships_over_{non_ship_size}m.csv'
output_dir =f'/home/joe/tmp/'
gt_xm_raw_bursts_series.to_csv(f'{output_dir}raw_{output_filename}', index=False)
gt_xm_slc_bursts_series.to_csv(f'{output_dir}slc_{output_filename}', index=False)
