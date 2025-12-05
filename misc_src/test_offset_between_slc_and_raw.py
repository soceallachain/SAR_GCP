#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:26:56 2025

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
#f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
# Plot RAW burst
img_name = f'SLC_matched_Raw_Burst_Real_{crc_burst}'
f=plot_2d_array(np.real(slc_matched_raw_burst),img_name,'linear')
#f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)

# Want to focus raw burst that matched SLC timings
rzp = 24000
azp = 1500
rd = focus_SM_burst_with_zero_padding(meta_data, slc_matched_raw_burst,
                                      roi, p_l0file, 'rma',rzp,azp
                                      )
img_name = f'Full_focus_{crc_burst}_range_padding_{rzp}_az_padding_{azp}'
f=plot_2d_array(np.square(np.abs(rd)),img_name)
#f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)

rd = focus_SM_burst_with_zero_padding(meta_data, slc_matched_raw_burst,
                                      roi, p_l0file, 'r',rzp,azp
                                      )
img_name = f'Range_compression_{crc_burst}_range_padding_{rzp}_az_padding_{azp}'
f=plot_2d_array(np.square(np.abs(rd)),img_name)


rd = range_compress_burst_with_zero_padding(slc_matched_raw_burst, rzp, azp
                                            )
img_name = f'Range_compression_{crc_burst}_range_padding_{rzp}_az_padding_{azp}'
f=plot_2d_array(np.square(np.abs(rd)),img_name)

