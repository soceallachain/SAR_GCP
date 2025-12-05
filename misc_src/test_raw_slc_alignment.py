#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 9th 2025

@author: joe
"""
###############################################################################
"""
# Script to  test that RAW and SLC burst data align
# Accepts paths to directories containing RAW and SLC data
# Loops through RAW directories
#      - For each directory locates equivalent SLC based on mapping csv file
#      - Locates SLC annotation file (depending on swath and assumes vh pol)
#      - Reads SLC annotation file for burst information
#      - Reads RAW file for burst metadata
#      - Reads csv mapping file for ship locations
#      - Generates bounding box around ship locations and paints in SLC burst image
#      - Plots SLC burst image with bounding boxes and binary burst image
#      - Loads saved RAW real, imag  and binary burst files and plots them
# Note: np.save saves to current directory unless supplied with path
#
"""
from lxml import etree
# system modules
import os
import io
import re
import sys
import glob
import random
import shutil
# data manipulation
import shapely
import cv2
import sys
import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.measure as skm
from skimage.morphology import erosion, dilation
from skimage import exposure
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import colors
from datetime import datetime, timedelta
from shapely import LineString, Point, Polygon
from scipy.interpolate import interp1d
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.precision', 12)
sys.path.insert(0, '/home/joe/Misc')

from  SAR_code_module import *

swath_of_interest = 2
polarization      ='vh'
swath_of_interest_index = swath_of_interest - 1
IW_mode = swath_of_interest_index

# Get SLC file to match raw based on mapping file
scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"

#scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_train.csv'
mapping_df         = pd.read_csv(scene_mapping_file)

#raw_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Train"
#slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_TrainingData"
raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"


#save_path = "/mnt/smbshare/tempTest/Data/Burst_Training_Data/"
save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/raw_burst_training/"

if not (os.path.exists(raw_path)):
    print(f' Path to {raw_path} inaccessible: Exiting!')
    sys.exit()
else:
    print(f' Path to {raw_path} found')
    raw_dirs =os.listdir(raw_path)
    raw_directories = raw_dirs
    print(raw_dirs)


for raw_directory in [raw_directories[0]]:

    raw_root_file_path   = f'{raw_path}/{raw_directory}/RAW/'
    if os.listdir(raw_root_file_path) and contains_SAFE_file(raw_root_file_path):
        raw_safe_dir  = os.listdir(raw_root_file_path)
        raw_src_path  = f'{raw_root_file_path}{raw_safe_dir[0]}/'
        raw_filenames = glob.glob(f'{raw_src_path}*{polarization}*.dat')
        # Need to sort filenames to access correct .dat file
        raw_filenames.sort(reverse=True)
        raw_filename = raw_filenames[0]
        if not (os.path.exists(raw_filename[0])):
            print(f'Error: Filename {raw_filename} not found. Continuing to next raw file ....')   
            continue
        else:
            raw_iw_data       = get_raw_burst_info(raw_filename)
            print(f' File {raw_filename} found. ')
    else:
        print(f'Error: Raw file path {raw_root_file_path} not found or does not contain .SAFE directory. Continuing to next raw file ......')
        continue


    ########################################################################### 
    ### Get matching SLC file
    target_scene_id     = raw_directory
    #Check if target scene exists in dataframe
    filtered_mapping_df = mapping_df.loc[mapping_df["scene_id"] == target_scene_id]
    if not filtered_mapping_df.empty:
        slc_id = filtered_mapping_df["SLC_product_identifier"].iloc[0]
        print(f"SLC_product_identifier for scene_id {target_scene_id}: {slc_id}")
    else:
         print(f"ERROR: No SLC match found for scene_id {target_scene_id}. Continuing to next raw file .....")
         continue

    SLC_root_file_path = f'{slc_path}/{slc_id}'
    SLC_root_file_path = SLC_root_file_path.replace("__","_")
    #p1,p2,p3,p4,p5    = getSLCFilesFromRoot(, swath_of_interest)
    slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
    slc_src_file      = slc_file_info[0]
    annotation_file   = slc_file_info[1]
    annotation_path   = slc_file_info[2]
    SLC_fnames        = slc_file_info[3]
    SAFE_dir          = slc_file_info[4]
    get_slc_info_status = slc_file_info[5]

    if get_slc_info_status != 1:
        print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
        continue

    print(f' Annotation file is: {annotation_file}')
    print(f' Annotation_dir  is: {annotation_path}')
    print(f' SLC_fname(s) are  : {SLC_fnames}')
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    Annotation_dir = annotation_path
    SLC_safe_id, _ = os.path.splitext(SAFE_dir)
    label_file     = scene_mapping_file
    #############################################################################
    if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir)
                                       and os.path.exists(slc_src_file)):
        print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
    else:
        print(f'ERROR: {annotation_file} or {Annotation_dir} '
              f' or {slc_src_file} Not found. Exiting')
        continue
        #exit(0)

    # Have SLC src and annotation file and raw filename
    ###########################################################################

    ###########################################################################
    # get binary masks for SLC bursts
    ship_coords = getShipCoordinates(label_file, swath_of_interest, SLC_safe_id)
    burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
    IW_gps_boundary           = burst_information[0]
    burst_boundaries          = burst_information[1]
    burst_gps_boundaries      = burst_information[2]
    burst_grid_boundaries     = burst_information[3]
    burst_grid_gps_boundaries = burst_information[4]
    number_of_bursts          = len(burst_boundaries)
    bounding_box_size         = 100
    border                    = 25
    swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
                                             burst_boundaries,
                                             burst_gps_boundaries,
                                             burst_grid_boundaries,
                                             burst_grid_gps_boundaries,
                                             bounding_box_size)
    # Choose burst of interest
    n = len(burst_boundaries)
    burst_counts = [burst_indices.count(i) for i in range(0, n)]
    # Print the table
    print("Burst Number:\t" ,  "\t".join(map(str, range(1, n + 1))))
    print("#Ships:      \t",  "\t".join(map(str, burst_counts)))
    burst_of_interest = burst_counts.index(max(burst_counts)) + 1
    slc_burst         = burst_of_interest
    ############################################################################
    # Load real and imaginary components of burst
    burst_real_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_real_burst#{slc_burst}.npy"
    burst_imag_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_imag_burst#{slc_burst}.npy"
    burst_bin_file  = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_binary_mask_burst#{slc_burst}.npy"
    if os.path.exists(burst_real_file) and os.path.exists(burst_imag_file) and os.path.exists(burst_bin_file):
        burst_real = np.load(burst_real_file)
        burst_imag = np.load(burst_imag_file)
        burst_bin  = np.load(burst_bin_file)
        print(f'All burst files found')
    else:
        print(f'ERROR : At least one raw burst data file not found in {save_path}. Exiting')
        exit(0)
    ###########################################################################


    if os.path.exists(slc_src_file):
        print(f"{SLC_fnames[swath_of_interest_index]} found")
    else:
        print(f"ERROR: {slc_src_file} Not found. Exiting")
        #exit(0)

    src            = rasterio.open(slc_src_file)
    img            = src.read(1)

    bursts_to_plot = [slc_burst]
    output=get_chips_and_masks(img, SLC_safe_id,
                             bursts_to_plot, burst_boundaries,
                             np.array(swath_bounding_boxes),
                             burst_indices, bounding_box_size, border
                            )

    
    
burst_chips        = output[0]
binary_burst_chips = output[1]
final_burst_img    = output[2]
final_binary_img   = output[3]
final_burst_chips  = output[4]
final_burst_binary_chips = output[5]

print(f'All data available for plotting ..... is it plotting?   ..... IS IT????')

# Plot real and imaginary burst data
fig1, ax1 = plt.subplots(1,2)
ax1[0].imshow(burst_real, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
ax1[1].imshow(burst_imag, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
figname = 'Burst_real_imag.jpg'
plt.savefig(figname, format='jpg', dpi=300)
plt.close()
###########################################################################
# Plot slc and binary mask
fig2, ax2 = plt.subplots(1,2)
ax2[0].imshow(final_burst_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
ax2[1].imshow(burst_bin,cmap='Greys_r', aspect = '10')
figname = 'SLC_bin_mask.jpg'
plt.savefig(figname, format='jpg', dpi=300)
plt.close()
