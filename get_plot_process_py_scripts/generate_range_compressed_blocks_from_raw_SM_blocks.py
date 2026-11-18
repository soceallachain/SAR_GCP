#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:14:20 2026

@author: joe
"""
# Script to generate range compressed Stripmap Data
# Accepts existing Strip raw data and generates corresponding
# range-compressed data.
#
# From file name extracts satellite and swath information which it
# uses to set the range-compression parameters based on a Look Up Table
# contained in the SAR_focusing_module

import os
import pickle
import numpy as np
from modules.Houston_SM_Config import *

src_dir = '/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/raw_burst/'
dest_dir = '/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/rc_burst/'
os.makedirs(dest_dir, exist_ok=True)
range_padding = 1000
azimuth_padding = 1000

file_list = sorted(os.listdir(src_dir))

# Tip: Use `file_list[:1]` for a single-file test run, or `file_list` for all files
for file_name in file_list:
#for file_name in file_list[2:4]:
    if not file_name.endswith('.npy'):
        continue

    slc_id = file_name[0:67]
    if '_S3_' not in slc_id and '_S6_' not in slc_id:
        print(f'Warning: {shorten_slc(slc_id)} swath not supported')
        continue
    elif '_S3_' in slc_id:
        swath = 3
    elif '_S6_' in slc_id:
        swath = 6
    satellite = 'S1SM'

    raw_data_file_path = os.path.join(src_dir, file_name)
    try:
        # 1. Load directly with np.load (checks integrity)
        raw_data = np.load(raw_data_file_path, allow_pickle=False)

        # 2. Sanity check on array contents before heavy processing
        if raw_data.size == 0 or np.isnan(raw_data).all():
            raise ValueError("Array is empty or contains only NaNs")

        # 3. Process and save
        rc_data = range_compress_padded_raw_data(
            raw_data,
            satellite,
            swath,
            range_padding,
            azimuth_padding
        )
        rc_data_file_path = os.path.join(
            dest_dir,
            file_name.replace('_raw_', '_rc_')
        )
        np.save(rc_data_file_path, rc_data)

    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError,
            ValueError, EOFError, pickle.UnpicklingError) as e:
        print(f"Skipping '{raw_data_file_path}': corrupted or unreadable ({e})")
        continue