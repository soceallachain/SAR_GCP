#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 23 May 2025
@author: joe

Script to identify which chips have ships, H & M confidence levels and sea status and length > 100m.
Currently assumes IW2 and vh polarisaton. 
Based on gen_raw_training_data_spyder.py as it extracts the SLC_ID 
from the SLC file matching the raw data.

Should really just use the label file to find SLC_Product_ID and 
annotation file

Generates 3 csv files:
 1.  SARFish_<RAW/SLC>_Train_BurstChipInfoShipsPerChip_.csv - lists ships per burst and chip allowing for overlap at edges
 2.  SARFish_<RAW/SLC>_Train_BurstChipInfo.csv              - lists ships per burst and chip without allowing for overlap at edges
 3.  SARFish_<RAW/SLC>_Train_SeaBasedBurstsAndChips         - denotes whether bursts wholly sea based

Depending on whether to be used with RAW or SLC data. When using RAW, chips have to be resampled to IW mode raw sampling frequency.
 
NOTE: Remember to set appropriate sar_mode !!
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
# What mode are the data be used for : SLC or RAW
sar_mode = 'RAW'

#Set paramters to enable finding chip number for each ship
chip_size =[512, 3072]
# SLC range sampled at 64.345238MHz, but RAW is swath dependent
# Have to adjust slc burst size to match raw burst size depending on 
# range sampling frequency so set freq_mutipliers to RAW_f/SLC_f
freq_multipliers=[1, 54.595959/64.345238, 46.915402/64.345238]

if sar_mode == 'RAW':
   sampling_freq_mul=freq_multipliers[IW_mode]
else: 
   sampling_freq_mul = 1
 

# Get SLC file to match raw based on mapping file
#scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_Public_Labels.csv'

scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_Public_labels.csv'
mapping_df         = pd.read_csv(scene_mapping_file)

raw_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Public"
slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_Public"
#raw_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Sandbox/RAW_Public_Test"
#slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Sandbox/SLC_Public_Test"
#raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
#slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"


#save_path = "/mnt/smbshare/tempTest/Data/Burst_Training_Data/"
save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/Dump/"
#save_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/"

if not (os.path.exists(raw_path)):
    print(f' Path to {raw_path} inaccessible: Exiting!')
    sys.exit()
else:
    print(f' Path to {raw_path} found')
    raw_dirs =os.listdir(raw_path)
    raw_directories = raw_dirs
    print(raw_dirs)
    
    
# Create dataframe for Product_ID and Burst Sea Status
columns = ["Product_ID", "Swath", "Burst_Number", "Number_Ships", "Chip_Number",
           "gps_lon", "gps_lat","ship_burst_row", "ship_burst_col",
           "ship_chip_row", "ship_chip_col", "ship_length", "Confidence","Sea_Status", 
           "Nearest_Country", "Distance_To_Shore"]
# Create an empty DataFrame with the specified column headings
df = pd.DataFrame(columns=columns)
df2= pd.DataFrame(columns=columns)
swath =f'IW{swath_of_interest}'

#for raw_directory in [raw_directories[3]]:
#for raw_directory in raw_directories[:10]:
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
    #ship_coords, ship_confidence = getShipCoordinatesConfidence(label_file, swath_of_interest, SLC_safe_id)
    ship_coords, ship_confidence, ship_length = getShipCoordinatesLengthConfidence(label_file, swath_of_interest, SLC_safe_id)
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
    #for burst_index in [1]:
    for burst_index in range(0, n):
        a,b,c = get_sea_status(burst_gps_boundaries[burst_index], 
                              plot_burst_location)
        sea_status = not(a)
        country    = b
        distance   = f'{c:.1f}'
        num_ships_in_burst = burst_counts[burst_index]
        
        print(f'Burst {burst_index+1} : Sea Status = {sea_status}, Closest Country={country}')
       
        #Get burst size matching raw burst - have to re-sample
        #Must correct for start of burst being offet into SLC image
        burst_dims = np.array(burst_boundaries[burst_index])
        burst_start_offset = burst_dims[0,0]
        burst_size=[(burst_dims[1,0]-burst_start_offset),round(burst_dims[2,1]*sampling_freq_mul)]
        #Get ships in each burst
        ship_indices = np.where(np.array(burst_indices) == burst_index)[0]
        chips_with_ships=[]
        i=0
        burst_height  = burst_size[0]
        burst_width   = burst_size[1]
        chip_height   = chip_size[0]
        chip_width    = chip_size[1]
        chips_per_row = (burst_width -1)//chip_width  + 1
        chips_per_col = (burst_height -1)//chip_height  + 1
        # Find chips which can contibute ships to overlapping chips
        row_chips_to_check,col_chips_to_check = get_chips_to_check(burst_size, chip_size)
        
        for ship_number in ship_indices:
            print(f'Swath Bounding box for ship at index {ship_number} is {swath_bounding_boxes[ship_number]}')
            ship_bounding_box = Polygon(swath_bounding_boxes[ship_number])
            ship_centroid = ship_bounding_box.centroid
            # Shapely uses loc.x amd loc.y not indexing. Note [row,col] of 
            # image format means row == .y and col == .x
            # Have to re-sample in range (column , or .x) direction ????????
            ship_location = [round(ship_centroid.x) - burst_start_offset, 
                             round(ship_centroid.y * sampling_freq_mul)
                            ]
            #Get location of ship in burst [row, col]
            ship_burst_row = ship_location[0]
            ship_burst_col = ship_location[1]
            #Get location of ship in chip
            ship_chip_row = ship_burst_row % chip_height
            ship_chip_col = ship_burst_col % chip_width
            # Get chip that ship belongs to
            chip_number = get_chip_number(ship_location, chip_size, burst_size)            
            ship_gps    = ship_coords[ship_number]
            confidence  = ship_confidence[ship_number]
            length      = ship_length[ship_number]
            data        = pd.DataFrame([[slc_id, swath, burst_index+1, 
                                         num_ships_in_burst, chip_number, 
                                         ship_gps[0], ship_gps[1],
                                         ship_burst_row, ship_burst_col,
                                         ship_chip_row, ship_chip_col,
                                         length, confidence, 
                                         sea_status, country, 
                                         distance]], columns=df.columns)
            # Append the new row to the existing DataFrame
            df = pd.concat([df, data], ignore_index=True)       
            
            df2 = pd.concat([df2, data], ignore_index=True)       

            chips_with_ships.append(chip_number)
            #Check whether chip overlaps other chip
            if chip_number in row_chips_to_check:
                # Check if ship located within chip_height of top of burst
                if burst_height - ship_location[0] <= chip_height:
                    #Chip overlaps with chip above it
                    adjoining_chip = chip_number + chips_per_row
                    ship_chip_row  = chip_height - (burst_height - ship_burst_row) 
                    data        = pd.DataFrame([[slc_id, swath, burst_index+1, 
                                         num_ships_in_burst, adjoining_chip, 
                                         ship_gps[0], ship_gps[1],
                                         ship_burst_row, ship_burst_col,
                                         ship_chip_row, ship_chip_col,
                                         length, confidence,
                                         sea_status, country, distance]],
                                         columns=df.columns)
                    df2 = pd.concat([df2, data], ignore_index=True)       
                    chips_with_ships.append(adjoining_chip)
                    
            if chip_number in col_chips_to_check:
                # Check if ship located within chip_width of end of row of burst
                if burst_width - ship_location[1] <= chip_width:
                    #Chip overlaps with final chip on row
                    ship_chip_col = chip_width - (burst_width - ship_burst_col)
                    adjoining_chip = chip_number + 1                    
                    data        = pd.DataFrame([[slc_id, swath, burst_index+1, 
                                         num_ships_in_burst, adjoining_chip, 
                                         ship_gps[0], ship_gps[1],
                                         ship_burst_row, ship_burst_col,
                                         ship_chip_row, ship_chip_col,
                                         length, confidence,
                                         sea_status, country, distance]],
                                         columns=df.columns)
                    df2 = pd.concat([df2, data], ignore_index=True)       
                    chips_with_ships.append(adjoining_chip)
       
       
       
df['Ships_per_Chip']  =  df.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Chip_Number'].transform('count')
df2['Ships_per_Chip'] = df2.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Chip_Number'].transform('count')

df['H_Ships_per_Chip'] = df.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Confidence'].transform(lambda x: (x == 'HIGH').sum())
df2['H_Ships_per_Chip'] = df2.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Confidence'].transform(lambda x: (x == 'HIGH').sum())

df['M_Ships_per_Chip'] = df.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Confidence'].transform(lambda x: (x == 'MEDIUM').sum())
df2['M_Ships_per_Chip'] = df2.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Confidence'].transform(lambda x: (x == 'MEDIUM').sum())

df['L_Ships_per_Chip'] = df.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Confidence'].transform(lambda x: (x == 'LOW').sum())
df2['L_Ships_per_Chip'] = df2.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Confidence'].transform(lambda x: (x == 'LOW').sum())


result=df.loc[df['Sea_Status']==True]


result.to_csv(f'{save_path}SARFish_{sar_mode}_Public_SeaBasedBurstsAndChipsWithShipLocations.csv', index=False)  
df.to_csv(f'{save_path}SARFish_{sar_mode}_Public_BurstChipInfoWithShipLocations.csv', index=False)  
df2.to_csv(f'{save_path}SARFish_{sar_mode}_Public_BurstChipInfoShipsPerChipWithShipLocations.csv', index=False)  

df2["Chip_Name"] = (
                    df2["Product_ID"].astype(str) + "_RAW_IW2_real_burst#" +
                    df2["Burst_Number"].astype(str) + "_chip#" +
                    df2["Chip_Number"].astype(str) + "_#Ships" +
                    df2["Ships_per_Chip"].astype(str) + ".npy"
                   )


#Compute total HIGH, MEDIUM, and LOW counts per region
chip_counts = df2.groupby(['Product_ID', 'Burst_Number', 'Chip_Number'])['Confidence'].value_counts().unstack(fill_value=0)

# Filter for chips where there are only HIGH confidence ships
only_HIGH = chip_counts[(chip_counts['HIGH'] > 0) & (chip_counts['LOW'] == 0) & (chip_counts['MEDIUM'] == 0)].reset_index()

# Filter for chips where there are only MEDIUM confidence ships
only_MEDIUM = chip_counts[(chip_counts['HIGH'] == 0) & (chip_counts['LOW'] == 0) & (chip_counts['MEDIUM'] > 0)].reset_index()

# Filter for chips where there are only HIGH & MEDIUM confidence ships
only_HIGH_and_MEDIUM = chip_counts[(chip_counts['HIGH'] > 0) & (chip_counts['LOW'] == 0) & (chip_counts['MEDIUM'] > 0)].reset_index()

# Merge back with the original dataframe to extract only rows from these regions
df_only_high   = df2.merge(only_HIGH[['Product_ID', 'Burst_Number', 'Chip_Number']], on=['Product_ID', 'Burst_Number', 'Chip_Number'])
df_only_medium = df2.merge(only_MEDIUM[['Product_ID', 'Burst_Number', 'Chip_Number']], on=['Product_ID', 'Burst_Number', 'Chip_Number'])
df_only_high_and_medium = df2.merge(only_HIGH_and_MEDIUM[['Product_ID', 'Burst_Number', 'Chip_Number']], on=['Product_ID', 'Burst_Number', 'Chip_Number'])

#df_only_sea_and_high = df_only_high[df_only_high['Sea_Status']== True]
df_only_sea_and_high = df_only_high[(df_only_high['Sea_Status'] == True) & (df_only_high['ship_length'] > 100)]
df_only_sea_and_medium = df_only_medium[(df_only_medium['Sea_Status']== True) & (df_only_medium['ship_length'] > 100)]
df_only_sea_and_high_and_medium = df_only_high_and_medium[(df_only_high_and_medium['Sea_Status']== True) & (df_only_high_and_medium['ship_length'] > 100)]
#df_only_sea_and_high_and_medium = df_only_high_and_medium[df_only_high_and_medium['Sea_Status']== True]
# Save or display results
df_only_sea_and_high.to_csv(f'{save_path}SARFish_{sar_mode}_Public_BurstChipInfo_Only_Sea_HIGH_WithShipLocations.csv', index=False)
df_only_sea_and_medium.to_csv(f'{save_path}SARFish_{sar_mode}_Public_BurstChipInfo_Only_Sea_MEDIUM_WithShipLocations.csv', index=False)
df_only_sea_and_high_and_medium.to_csv(f'{save_path}SARFish_{sar_mode}_Public_BurstChipInfo_Only_Sea_HIGH_and_MEDIUM_WithShipLocations.csv', index=False)

high_conf_ship_chip_list=df_only_sea_and_high["Chip_Name"].to_list()

with open(f'{save_path}{sar_mode}_high_conf_ship_chips_sea_only_gt100.txt', 'w') as f:
    for item in high_conf_ship_chip_list:
        print(f'{save_path}{item}')
        f.write(f"{save_path}{item}\n")

med_conf_ship_chip_list=df_only_sea_and_medium["Chip_Name"].to_list()
with open(f'{save_path}{sar_mode}_medium_conf_ship_chips_sea_only_gt100.txt', 'w') as f:
    for item in med_conf_ship_chip_list:
        f.write(f"{save_path}{item}\n")

high_med_conf_ship_chip_list=df_only_sea_and_high_and_medium["Chip_Name"].to_list()
with open(f'{save_path}{sar_mode}_high_med_conf_ship_chips_sea_only_gt100.txt', 'w') as f:
    for item in high_med_conf_ship_chip_list:
        f.write(f"{save_path}{item}\n")

print(f'******* get_public_labels_for_high_and_medium_conf_ships_sea_only.py Complete! *******')
