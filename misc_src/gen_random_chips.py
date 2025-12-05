#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:40:52 2025

@author: joe


Script to read csv file with bursts' ships' locations and to read csv file 
with bursts' non-ships' locations and generates the locations of
random chips in the bursts
"""

import pandas as pd
import numpy as np
import os
import re





local = 1
if local ==1:
    label_dir='/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/random_chip_info_dir/'
    label_ship_file='SARFish_Public_RAW_IW2_BurstChipInfo_IsVessels_PerChip.csv'
    label_non_ship_file='SARFish_Public_RAW_IW2_BurstChipInfo_NonVessels_PerChip.csv'
    label_ship_file_path=f'{label_dir}{label_ship_file}'
    label_non_ship_file_path=f'{label_dir}{label_non_ship_file}'
    save_path = ''
    chip_dir  = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/raw_public_burst_random_chip_data'
else:
    label_dir=''
    label_ship_file='SARFish_Public_RAW_IW2_BurstChipInfo_IsVessels_PerChip.csv'
    label_non_ship_file='SARFish_Public_RAW_IW2_BurstChipInfo_NonVessels_PerChip.csv'
    label_ship_file_path=f'{label_dir}{label_ship_file}'
    label_non_ship_file_path=f'{label_dir}{label_non_ship_file}'
    save_path = ''
    chip_dir  = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/raw_public_burst_random_chip_data'
    
dataset='Public'
swath = 2
if (not os.path.exists(label_ship_file_path)) or (not os.path.exists(label_non_ship_file_path)):
    print(f'Label file not found at : {label_ship_file_path} or {label_non_ship_file_path}')
    print(f'Check spelling or location')
else:
    df_ship = pd.read_csv(label_ship_file_path)
    df_non_ship = pd.read_csv(label_non_ship_file_path)
   


# Read the csv file with sea only bursts and ship locations and ship details

# For each file;for each burst; for each ship; generate chip
# inc ship class confidence

#Count other ships and non-ships in chip; inc ship class confidence; repeat chip generation
#until 100 H only sea chips generated

