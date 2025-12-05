#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:00:43 2025

@author: joe
"""

from SAR_code_module import *


# Test functions
raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data/f9fd0c2000d1a632p/RAW/S1B_IW_RAW__0SDV_20200421T054807_20200421T054840_021238_0284D5_1452.SAFE/"
raw_file_name='s1b-iw-raw-s-vh-20200421t054807-20200421t054840-021238-0284d5.dat'
raw_file=f'{raw_path}{raw_file_name}'

slc_id = 'S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96'
slc_id_crc = slc_id[-11:]

burst_number = 3
annotation_file='/home/joe/Misc/tempTest/Data/SLC_Data/S1B_IW_SLC_1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96/SLC/S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96.SAFE/annotation/s1b-iw2-slc-vh-20200421t054811-20200421t054837-021238-0284d5-002.xml'
print(get_valid_corners_of_bursts(annotation_file))

burst=burst_number



b_orig=getSwathBurstGridGPSBoundaries_OriginalAndBest(annotation_file)


b_mod = getSwathBurstGridGPSBoundaries(annotation_file)

slc_burst = get_slc_burst_with_ships(slc_id, burst)
slc_ship_burst = slc_burst[1]
plot_2d_array(slc_ship_burst**2, f'{slc_id_crc} Burst#{burst}: Size = {slc_ship_burst.shape}')
print(f' SLC Burst #{burst} has size {slc_ship_burst.shape} ')



raw_burst_data = get_raw_burst(slc_id, burst)
plot_2d_array(np.real(raw_burst_data), f'Raw Data Burst#{burst} of {slc_id_crc}', 'linear')

slc_resampled_range_size = raw_burst_data.shape[1]
resampled_slc_burst = resample(slc_ship_burst**2, slc_resampled_range_size)
plot_2d_array(resampled_slc_burst, f'{slc_id_crc} Burst#{burst}: Size = {resampled_slc_burst.shape}')
