#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:11:26 2026

@author: joe
"""



import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d
from modules.SAR_code_module import *
from modules.SAR_focusing_module import *
from scipy.ndimage import zoom
#filepath = "../data/sao_paulo/"
#filename = "s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
import mpld3





# This file tests the operation of get_SM_files_from_slc_id to verify that
# can plot S1 SLC and raw data and zoom dynamically and then use this 
# functionality to test whether the offsets in range between SM RAW and SLC
# are fixed  (for swath or node) or are random.
# 

# Do this by running test_get_SM_GridGPSBoundaries 


# Get list of SLC_IDs
#root_path    = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_OCN/'

root_path   = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN/'
pol         = 'hv'
SLC_IDs     = os.listdir(root_path)

number_of_files_with_one_chunk = 0
number_of_files_with_more_than_one_chunk = 0
# Check how many of the SM files have > 1 acquisition chunk
for slc_id in SLC_IDs[30:56]:

   #slc_id_path = root_path + SLC_IDs[14] +'/'
   slc_id_path = root_path + slc_id +'/'
   files       = get_SM_files_from_slc_id(slc_id_path, pol)   
   if files == -1:
       print(f'Warning: {slc_id} missing files')
       continue
   raw_file    = files[0]
   slc_file    = files[1]
   annot_file  = files[2]


   raw_burst_info, l0file = get_SM_raw_burst_info_v2(raw_file)
   # If > 1 acq chunks then raw_burst_info[0] has >1 rows
   number_of_chunks = len(raw_burst_info[0])
   if number_of_chunks ==1:
       number_of_files_with_one_chunk +=1
   else:
       number_of_files_with_more_than_one_chunk +=1
   
   print(f'{slc_id} has {number_of_chunks} chunks : {raw_burst_info[0].index[:].to_list()}')


print(f'Out of {len(SLC_IDs)} SM files {number_of_files_with_one_chunk} have 1 '
      f'chunk - {100*number_of_files_with_one_chunk/len(SLC_IDs)}% ')



