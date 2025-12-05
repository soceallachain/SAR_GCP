#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:01:53 2025

@author: joe

File to generate the labels for public dataset in the same format as 
other SARFish labels
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

from pathlib import Path



def get_swaths_boundaries(slc_path):
    slc_fnames          = []
    any_swath           = 1
    file_info           = get_slc_related_filenames(slc_path,any_swath,'vh')
    slc_fnames          = file_info[3]
    annotation_path     = file_info[2]    
    #swath_gps_boundaries = getAllSwathBoundaries(slc_fnames, annotation_path)
    swath_gps_boundaries, burst_gps_boundaries = getAllSwathBurstGPSBoundaries(slc_fnames, annotation_path)
    return swath_gps_boundaries, burst_gps_boundaries


public_csv_dir='/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/xView3Labels/'
public_csv='/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/xView3Labels/public.csv'
scene_map_file= '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/ESA_xView3_SceneName_Mapping.csv'

public_slc_dir='/home/joe/Misc/tempTest/Data/SLC_Data/SLC_Public_Test'

# Define the column names for dfA
columns_A = [
    "partition", "product_type", "scene_id", "detect_id", "SLC_product_identifier", "swath_index",
    "detect_lat", "detect_lon", "detect_scene_column", "detect_scene_row", "top", "left", "bottom",
    "right", "vessel_length_m", "source", "is_vessel", "is_fishing",
    "global_shoreline_vector_distance_from_shore_km", "xView3_shoreline_distance_from_shore_km", "confidence"
]

# Create an empty DataFrame with the given columns
dfA = pd.DataFrame(columns=columns_A)

# Load the Public.csv file
df_public = pd.read_csv(public_csv)
df_mapping = pd.read_csv(scene_map_file)

# Identify matching columns (excluding 'top', 'bottom', 'left', 'right')
matching_columns = [col for col in df_public.columns if col in dfA.columns and col not in ['top', 'bottom', 'left', 'right']]

# Copy the matching columns to dfA
dfA[matching_columns] = df_public[matching_columns]
# Assign 'train' to all elements in the 'partition' column
dfA['partition'] = 'public'

# Assign 'SLC' to all elements in the 'product_type' column
dfA['product_type'] = 'SLC'

dfA['xView3_shoreline_distance_from_shore_km'] = df_public['distance_from_shore_km']

# Load mapping.csv
df_mapping = pd.read_csv(scene_map_file)

# Merge dfA with df_mapping on scene_id (dfA) and scene_name (df_mapping)
dfA = dfA.merge(df_mapping[['XVIEW3_SCENENAME', 'ESA_SCENENAME']], 
                left_on='scene_id', right_on='XVIEW3_SCENENAME', how='left')

# Assign ESA_Scene values to GRD_product_identifier
dfA['GRD_product_identifier'] = dfA['ESA_SCENENAME']

# Drop the extra scene_name column (from mapping.csv)
dfA.drop(columns=['XVIEW3_SCENENAME', 'ESA_SCENENAME'], inplace=True)

# Define the directory containing potential matches
slc_dirs = Path(public_slc_dir)

# Ensure the directory exists
if not slc_dirs.exists() or not slc_dirs.is_dir():
    raise FileNotFoundError(f"{slc_dirs} does not exist or is not a directory.")

# Get the list of GRD_Product_Id values
grd_product_ids = set(dfA["GRD_product_identifier"].dropna())


#Iterate through directories in slc_dirs
for child_dir in slc_dirs.iterdir():
    if child_dir.is_dir() and child_dir.name in grd_product_ids:
        slc_subdir = child_dir / "SLC"

        # Check if 'SLC' directory exists inside the matched directory
        if slc_subdir.exists() and slc_subdir.is_dir():
            # Find a directory ending with .SAFE inside the SLC subdir
            safe_dirs = [d for d in slc_subdir.iterdir() if d.is_dir() and d.suffix == ".SAFE"]

            if safe_dirs:
                safe_name = safe_dirs[0].stem  # Get prefix before .SAFE

                # Update the corresponding row in dfA
                dfA.loc[dfA["GRD_product_identifier"] == child_dir.name, "SLC_product_identifier"] = safe_name
                # Rename the child directory to match the extracted prefix
                modified_safe_name = safe_name.replace("_SLC__", "_SLC_")
                new_dir = slc_dirs / modified_safe_name
                
                if not new_dir.exists():  # Avoid overwriting an existing directory
                    child_dir.rename(new_dir)
                    print(f"Renamed {child_dir} → {new_dir}")
                else:
                    print(f"Skipping rename: {new_dir} already exists.")            
                
            else:
                print(f"Directory {safe_dirs.name} does not exist.")
ii=0
swath_boundary_values={}                    
for slc_product_id in dfA['SLC_product_identifier'].dropna().unique():
        slc_path = f'{public_slc_dir}/{slc_product_id}'   
        slc_path = slc_path.replace('SLC__','SLC_')
        swaths_boundaries, burst_boundaries = get_swaths_boundaries(slc_path)
        swath_boundary_values['SLC_product_identifier'] = swaths_boundaries
        ii=ii+1
        loc = dfA[dfA['SLC_product_identifier']==slc_product_id][['detect_lon', 'detect_lat']].to_numpy()
        iw_mode=[]
        minus1_mode_count = 0
        for i in range(0, len(loc)):
            # Get swath that location belongs to
            mode=getGrid(loc[i], swaths_boundaries)
            # If location not found in swath - check bursts as burst edges not 
            # always aligned with swath edges
            if mode == -1:
                minus1_mode_count = minus1_mode_count + 1
                for j in range(0,3):
                    # Find burst location belongs to - check all swaths
                    b_mode = getGrid(loc[i], burst_boundaries[j])
                    if b_mode == -1:
                        continue
                    else:
                        break
                # If find location in a burst set mode to swath burst is in    
                if b_mode != -1:
                    mode = j
            #Add 1 to mode to match IW1,2 or 3
            mode = mode + 1
            iw_mode.append(mode)    
        indices = dfA[dfA['SLC_product_identifier'] == slc_product_id].index
        dfA.loc[indices, 'swath_index'] = iw_mode 
        #dfA.to_csv('/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_Public_Labels.csv', index=False)

print(f'/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_Public_Labels.csv  written!')
print(f'********************************* generate_public_labels - Complete! **********************************************')

                    
        
  
