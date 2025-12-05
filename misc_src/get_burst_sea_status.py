#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:44:07 2025

@author: joe

Script to identify which files (and bursts) are wholly located at sea.
Based on gen_raw_training_data_spyder.py as it extracts the SLC_ID 
from the SLC file matching the raw data.

Should really just use the label file to find SLC_Product_ID and 
annotation file


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
#save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/raw_burst_training/"
save_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/"

if not (os.path.exists(raw_path)):
    print(f' Path to {raw_path} inaccessible: Exiting!')
    sys.exit()
else:
    print(f' Path to {raw_path} found')
    raw_dirs =os.listdir(raw_path)
    raw_directories = raw_dirs
    print(raw_dirs)
    
    
# Create dataframe for Product_ID and Burst Sea Status
columns = ["Product_ID", "Swath", "Burst_Number", "Number_Ships", "Sea_Status", "Nearest_Country", "Distance_To_Shore"]
# Create an empty DataFrame with the specified column headings
df = pd.DataFrame(columns=columns)
swath =f'IW{swath_of_interest}'

#for raw_directory in [raw_directories[4]]:
for raw_directory in raw_directories:

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
    #slc_burst         = burst_of_interest
    slc_burst         = 5
    ############################################################################
    #For each burst get sea status
    plot_burst_location = 0
    #for burst_index in [3]:
    for burst_index in range(0, n):
        a,b,c = get_sea_status(burst_gps_boundaries[burst_index], 
                              plot_burst_location)
        sea_status = not(a)
        country    = b
        distance   = f'{c:.1f}'
        num_ships_in_burst = burst_counts[burst_index]
        data = pd.DataFrame([[slc_id, swath, burst_index+1, num_ships_in_burst,
                              sea_status, country, distance]],
                            columns=df.columns)
        # Append the new row to the existing DataFrame
        df = pd.concat([df, data], ignore_index=True)
        print(f'Burst {burst_index+1} : Sea Status = {sea_status}, Closest Country={country}')
       
       
result=df.loc[df['Sea_Status']==True]
result.to_csv('/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SARFish_Train_SeaBasedBursts.csv', index=False)  
df.to_csv('/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SARFish_Train_BurstInfo.csv', index=False)  

