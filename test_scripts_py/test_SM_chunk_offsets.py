#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th 2026
 ******************   test_SM_chunk_offsets.py  *************************************
Code to:
    - Read SM Raw and SLC files
    - Read raw chunks into chunk array
    - For each chunk
               - take NxN slice of raw and slc from start and end
               - run NCC on each pair and record offsets


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
from modules.image_processing_module import *

from scipy.ndimage import zoom








#Set DEBUG mode if do not want to save files
DEBUG = True
# Get list of SLC_IDs

#test_dir   = f'Houston_SM_2019_20'
#test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/jc_data/{test_dir}/'
#root_path  = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'
#test_dir   = f'SaoPaulo_SM_2019_20'
#root_path  = f'/mnt/SAR_bucket/data/SaoPaulo_SM/{test_dir}/'

test_dir   = f'Houston_SM_No_OCN'
root_path  = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'

pol         = 'hv'
pol2        = 'vh'
SLC_IDs     = os.listdir(root_path)
swaths      = ['_S1_', '_S3_', '_S6_']

n_files_per_swath = 5
save_fig_path     = f'./tests/test_offsets'



###############################################################################
## Function to process number of files belonging to different swaths
## Continues processing until requisite number of each are processed
##
def process_files_for_chunk_offsets(slc_ids, root_path, swaths, 
                                    n_files_per_swath, pol, slice_size
                                    ):

    #Initialize swath file counters dictionary
    swath_file = {swath: 0 for swath in swaths}
    total_files_required = n_files_per_swath * len(swaths)
    swath_offsets = []

    for slc_id in slc_ids:
        for swath in swaths:
            if swath in slc_id:
                if swath_file[swath] >= n_files_per_swath:
                    continue

                chunk_info = process_file_for_chunks(slc_id, root_path, pol)

                if chunk_info == -1:
                    continue
                else:
                    chunk_data      = chunk_info[0]
                    raw_slc_offsets = chunk_info[2]
                    files           = chunk_info[3]
                    slc_file        = files[1]
                    # Use context manager to properly manage rasterio handles
                    with rasterio.open(slc_file) as slc_data:
                        slc_image = slc_data.read(1)
                    offsets         = process_chunks_for_offsets(chunk_data, 
                                                                 slc_image,
                                                                 raw_slc_offsets, 
                                                                 slice_size
                                                                 ):
                    swath_file[swath]+=1
                    swath_offsets.append([swath, slc_id, offsets]) 
                    break # Move to the next slc_id once matched and processed

        if sum(swath_file.values()) == total_files_required:
            break


    return swath_offsets

"""
def process_files_for_chunk_offsets(slc_ids, root_path, swaths, n_files_per_swath, pol='VV'):
    # Initialize file counter dictionary for each swath name
    swath_file = {swath: 0 for swath in swaths}
    total_required_files = n_files_per_swath * len(swaths)

    for slc_id in slc_ids:
        # Check if all swaths have met their file quotas
        if sum(swath_file.values()) == total_required_files:
            break

        for swath in swaths:
            if swath in slc_id:
                # Skip if this swath quota is already satisfied
                if swath_file[swath] >= n_files_per_swath:
                    continue

                # Process chunk offsets
                chunk_info = process_file_for_chunks(slc_id, root_path, pol)
                
                # Check for processing failure
                if chunk_info == -1:
                    continue
                else:
                    swath_file[swath] += 1
                    break  # Move to the next slc_id once matched and processed

    return swath_file


"""

###############################################################################
## Function to determine offset at start and end of a single raw chunk
## Used to verify that offset is constant within a chunk
##
## Input Args: 
##             chunk      - 2D array of complex whole chunk data
##             slc_data   - 2D array of slc data matched to chunk size
##             slice_size - size of NxM slice to use for cross correlation
## Returns:
##             bottom_offset - array with azimuth and range offset near start
##             top_offset    - array with azimuth and range offset near end
def get_chunk_top_bottom_offsets(chunk, slc_data, slice_size):
   
   # Offset near start of chunk 
   chunk_slice    = chunk[0:slice_size[0], 0:slice_size[1]-1]
   slc_slice      = slc_data[0:slice_size[0], 0:slice_size[1]-1]
   bottom_offsets = compute_image_offset(chunk_slice, 
                                         slc_slice, plot_surface=False
                                         )
   # Offset near end of chunk 
   az_start_idx  = chunk.shape[0] - slice_size[0]
   az_end_idx    = chunk.shape[0] - 1
   chunk_slice   = chunk[az_start_idx:az_end_idx, 0:slice_size[1]-1]
   slc_slice     = slc_data[0:slice_size[0], 0:slice_size[1]-1]
   top_offsets   = compute_image_offset(chunk_slice, 
                                        slc_slice, plot_surface=False
                                        )
   return bottom_offsets, top_offsets

###############################################################################   
## 
## Input Args:
##            chunk_data      - array of raw chunk data
##            slc_image       - array of full slc data
##            raw_slc_offsets - ofsets measured from start of raw chunk
##            slice_size      - size of NxM slice to use for cross correlation
##
## Returns:
##            offsets          - array with top and bottom range and azimuth 
##                               offsets for each chunk
##
def process_chunks_for_offsets(chunk_data, slc_image, 
                               raw_slc_offsets, slice_size):
   
   n_slc_az_lines      = slc_image.shape[0]
   n_slc_range_samples = slc_image.shape[1]
   n_raw_az_lines      = sum(chunk_data[i].shape[0] for i in range(len(chunk_data)))
   # n_raw_az_lines      = sum(len(chunk) for chunk in chunk_data)
   offsets = [] 
   for chunk in range(len(chunk_data)):
      if chunk == 0:
         # Assign slc data
         slc_data = slc_image[0:len(chunk_data[chunk]) -1,:]
      elif chunk == len(chunk_data) -1:
         # Final chunk - make sure no overflow
         if n_slc_az_lines + raw_slc_offsets[0] > n_raw_az_lines:
            #Have overflow issue: 
            print(f'Warning:  Azimuth offset = {raw_slc_offsets[0]}: '
                  f'Too large for number of azimuth SLC lines ({n_slc_az_lines})'
                  f'Sum greater than number or chunk azimuth lines ({n_raw_az_lines})')
         slc_data = slc_image[n_slc_az_lines 
                     - len(chunk_data[chunk]):n_slc_az_lines -1, :]
      else:
       # Processing inner chunk
         slc_data = slc_image[len(chunk_data[chunk-1])  - raw_slc_offsets[0]
                             : len(chunk_data[chunk-1]) - raw_slc_offsets[0] 
                             + len(chunk_data[chunk]) -1,
                             :]
 
      offsets.append(get_chunk_top_bottom_offsets(chunk_data[chunk], slc_data, slice_size))

   return offsets



##############################################################################
##
def print_chunk_slc_offsets(offsets):
   # Header string formatted with fixed column widths
   print(f"{'Chunk':<12} | {'Bottom (Az, Range)':<22} | {'Top (Az, Range)':<22}")
   print("-" * 62)

   for i in range(len(offsets)):
       # Format each inner list/tuple into a uniform width string
       bot_str = f"[{offsets[i][0][0]}, {offsets[i][0][1]}]"
       top_str = f"[{offsets[i][1][0]}, {offsets[i][1][1]}]"

       print(f"Chunk#{i:<5} | {bot_str:<22} | {top_str:<22}")





##############################################################################
## Function to process a SM acquisition and return the raw and metadata for 
## each chunk comprising teh acquisiton and the timing based offsets and the 
## of the individual files of for the acquisition
##
def process_file_for_chunks(slc_id, root_path, pol):

   slc_id_path          = root_path + slc_id +'/'
   parts                = slc_id.split('_')
   slc_id_name          = f'{parts[0]}_{parts[1]}_{parts[-1]}'
   chunk_failed         = False
   number_failed_chunks = 0

   files       = get_SM_files_from_slc_id(slc_id_path, pol)
   if files == -1:
       print(f'Warning: {slc_id} missing files with pol = {pol}. Try {pol2}')
       files = get_SM_files_from_slc_id(slc_id_path, pol2)
       if files  == -1:
          print(f'Warning: {slc_id} missing files with pol = {pol2} as well. Definitely missing! ')
          return -1

   raw_file               = files[0]
   slc_file               = files[1]
   annotation_file        = files[2]
   raw_burst_info, l0file = get_SM_raw_burst_info_v2(raw_file)
   number_of_chunks       = len(raw_burst_info[0])

   print(f'{slc_id} has {number_of_chunks} chunks : {raw_burst_info[0].index[:].to_list()}')

   raw_slc_offsets = get_SM_raw_slc_offsets_v2(raw_file, annotation_file)

   # Get chunk data and place in slice array
   chunk_list     = raw_burst_info[0].index[:].to_list()
   chunk_data     = []
   chunk_metadata = []
   raw_data       = []
   for chunk in chunk_list:
       try:
           chunk_metadata.append(l0file.get_acquisition_chunk_metadata(chunk))
           chunk_data.append(l0file.get_acquisition_chunk_data(chunk))

           print(f'{slc_id} Chunk#{chunk} successfully read')
       except:
           print(f'{slc_id} Chunk#{chunk} failed!')
           chunk_metadata.append([])
           chunk_data.append([])
           number_failed_chunks+=1
           chunk_failed = True

   if number_failed_chunks == 0:
       for i in range(len(chunk_data)):
           az_start    = int(raw_slc_offsets[i][0])
           az_end      = int(raw_slc_offsets[i][1])
           range_start = int(raw_slc_offsets[i][2])
           range_end   = int(raw_slc_offsets[i][3])
           raw_data.append(chunk_data[i][az_start:az_end, range_start:range_end])
   else:
       print(f'Warning : {slc_id_name} has {number_failed_chunks} failed chunk(s):'
             f' Ignoring and moving to next file'
             )
       return -1


   return chunk_data, chunk_metadata, raw_slc_offsets, files


def get_slc_matched_chunk_data(chunk_data, raw_slc_offsets):

    for i in range(len(chunk_data)):
        az_start    = int(raw_slc_offsets[i][0])
        az_end      = int(raw_slc_offsets[i][1])
        range_start = int(raw_slc_offsets[i][2])
        range_end   = int(raw_slc_offsets[i][3])
        raw_data.append(chunk_data[i][az_start:az_end, range_start:range_end])

    return raw_data


