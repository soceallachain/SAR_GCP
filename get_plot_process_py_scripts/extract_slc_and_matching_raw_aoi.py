#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:45:38 2026

@author: joe
"""


from modules.Houston_SM_Config import *


slc_id = f'S1C_S6_SLC__1SDH_20261111T002553_20261111T002620_010282_014889_B23F'
files      = get_raw_slc_files(slc_id)

# Or use gps_aoi
# gps_aoi_lon_lat = [(SW_lon,SW_lat),(NE_lon,NE_lat)]
# slc_aoi = get_SM_cartesian_aoi(gps_AOI_lon_lat, files[2])



slc_aoi = [0,20000,0,17000]
slc_aoi_data = get_SM_slc_aoi(slc_aoi,files[1])
raw_aoi_data = get_SM_raw_aoi_data(slc_aoi, files)

node = get_orbital_node(files[2])
pol  = get_pol_from_annot_filepath(files[2])
scene_name = f'block_0_20k_az_0_17k_rng'
save_dir = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/test/'

np.save(f'{save_dir}{slc_id}_Pol_{pol}_Node_{node}_raw_{scene_name}.npy', raw_aoi_data)
np.save(f'{save_dir}{slc_id}_Pol_{pol}_Node_{node}_slc_{scene_name}.npy', slc_aoi_data)
