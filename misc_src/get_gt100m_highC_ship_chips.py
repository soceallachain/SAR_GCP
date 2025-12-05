#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:39:58 2025

@author: joe
"""

import pandas as pd
import numpy as np
import os
import re



label_dir='/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/'
label_file='SARFish_RAW_Public_BurstChipInfo_Only_Sea_HIGHWithShipLocations.csv'
label_file_path=f'{label_dir}{label_file}'


if not os.path.exists(label_file_path):
    print(f'Label file not found at : {label_file_path}')
    print(f'Check spelling or location')
else:
    df = pd.read_csv(label_file_path)
    # Filter rows where 'length' is 100 or more
    filtered_df = df[df['ship_length'] >= 100].round(1)
    
    #Want to create list of these chip files
    filtered_df["Chip_Name"] = (
    filtered_df["Product_ID"].astype(str) + "_RAW_IW2_real_burst#" +
    filtered_df["Burst_Number"].astype(str) + "_chip#" +
    filtered_df["Chip_Number"].astype(str) + "_#HShips" +
    filtered_df["H_Ships_per_Chip"].astype(str) + ".npy"
    )
    new_label_file = label_file.replace('HIGHWithShipLocations','HighConfidenceShips_gt_100m')
    
    filtered_df.to_csv(f'{label_dir}{new_label_file}', index=False)

#del df, filtered_df

#######################################################################
# Read label file for High C ships > 100m in length to get chip details    
chip_dir = ''
high_conf_ships_gt_100m_chip_dir = ''
label_dir='/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/'
label_file='SARFish_RAW_Public_BurstChipInfo_Only_Sea_HighConfidenceShips_gt_100m.csv'
label_file_path=f'{label_dir}{label_file}'


df=pd.read_csv(label_file)

chip_file_names =df['Chip_Name']

for f in chip_filenames:
    fname = f.replace('_#HShips*.npy', '.npy')
    
