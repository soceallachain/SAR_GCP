#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:47:46 2025

@author: joe
"""

###########################################################################
# Global parameters for processing RAWSARAI data

dataset      = 'Public'
iw_mode      = 2
polarization = 'vh'
sar_mode     = 'RAW'
local        = 1



local_raw_path  =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{dataset}/raw_data'
local_slc_path  =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{dataset}/slc_data'
#save_path ='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_get_bursts_for_HM_sea_only_ships' 
local_save_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/t2' 
local_scene_mapping_path = f"/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels"
nonlocal_raw_path  = f"/mnt/smbshare/RAWSARAI_Data/Copernicus/{dataset}/raw_data"
nonlocal_slc_path  = f"/mnt/smbshare/RAWSARAI_Data/Copernicus/{dataset}/slc_data"
nonlocal_save_path = f"/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/test_results/test_get_bursts_for_HM_sea_only_ships"
nonlocal_scene_mapping_path = f'/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels'
