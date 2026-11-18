#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:47:46 2025

@author: joe
"""

###########################################################################
# Global parameters for processing RAWSARAI data

#dataset      = 'Train'
dataset     = 'Public'
iw_mode      = 2
polarization = 'vh'
sar_mode     = 'RAW'
local        = 2
DEBUGx       = 0


local_raw_path  =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{dataset}/raw_data'
local_slc_path  =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{dataset}/slc_data'
local_save_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/t2' 
local_scene_mapping_path = f"/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels"

if DEBUGx:
    nonlocal_raw_path  = f"/mnt/SAR_bucket/data/Copernicus/Sandbox/Public/raw_data"
    nonlocal_slc_path  = f"/mnt/SAR_bucket/data/Copernicus/Sandbox/Public/slc_data"
    nonlocal_save_path = f"/mnt/SAR_bucket/data/Copernicus/Sandbox/test_burst_ship_labels"
    nonlocal_scene_mapping_path = f'/mnt/SAR_bucket/data/Copernicus/Info/SARFishLabels'

else:
    nonlocal_raw_path  = f"/mnt/SAR_bucket/data/Copernicus/{dataset}/raw_data"
    nonlocal_slc_path  = f"/mnt/SAR_bucket/data/Copernicus/{dataset}/slc_data"
    #nonlocal_save_path = f"/mnt/SAR_bucket/data/Copernicus/Info/SARFishLabels/Reports"
    nonlocal_save_path = f"/mnt/SAR_bucket/data/Copernicus/Sandbox/test_burst_ship_labels"
    nonlocal_scene_mapping_path = f'/mnt/SAR_bucket/data/Copernicus/Info/SARFishLabels'
