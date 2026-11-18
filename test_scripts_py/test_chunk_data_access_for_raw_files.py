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
root_path    = f'/mnt/SAR_bucket/data/SaoPaulo_SM/SaoPaulo_SM_2019_20/'
#root_path    = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_2019_20/'
#root_path    = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_OCN/'
#root_path   = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN/'
#root_path   = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'
pol         = 'hv'
pol2        = 'vh'
SLC_IDs     = os.listdir(root_path)

number_of_files_with_one_chunk = 0
number_of_files_with_more_than_one_chunk = 0
number_failed_SM_files = 0
number_failed_chunks   = 0
# Check how many of the SM files have > 1 acquisition chunk
for slc_id in SLC_IDs:

   #slc_id_path = root_path + SLC_IDs[14] +'/'
   #if '6E9A' not in slc_id:
   #    continue
   #if '6E9A' in slc_id:
   #    break
   slc_id_path = root_path + slc_id +'/'
   files       = get_SM_files_from_slc_id(slc_id_path, pol)   
   if files == -1:
       #print(f'Warning: {slc_id} missing files with pol = {pol}. Try {pol2}')
       files = get_SM_files_from_slc_id(slc_id_path, pol2)        
       if files  == -1:
          print(f'Warning: {slc_id} missing files with pol = {pol2} as well. Definitely missing! ')
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
   chunk_failed = False
   
   # Get chunk data
   chunk_list  = raw_burst_info[0].index[:].to_list()
   for chunk in chunk_list:
       try:
           selection      = l0file.get_acquisition_chunk_metadata(chunk)
           chunk_data     = l0file.get_acquisition_chunk_data(chunk)
           print(f'{slc_id} Chunk#{chunk} successfully read')
       except:
           print(f'{slc_id} Chunk#{chunk} failed!')
           number_failed_chunks+=1
           chunk_failed = True
   if chunk_failed:
       number_failed_SM_files +=1
               
   

   

print(f'Out of {len(SLC_IDs)} SM files {number_of_files_with_one_chunk} have 1 '
      f'chunk - {100*number_of_files_with_one_chunk/len(SLC_IDs)}% ')

print(f'Out of {len(SLC_IDs)} SM files Number of failed chunks is {number_failed_chunks}')
print(f'Out of {len(SLC_IDs)} SM files Number of failed files is {number_failed_SM_files}')
      



