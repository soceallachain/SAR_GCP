#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:57:52 2026

@author: joe
"""

import time

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
#from modules.chunk_processing_module import *
from modules.SAR_focusing_module import *
from modules.image_processing_module import *
from scipy.ndimage import zoom

test_dir   = f'Houston_SM_No_OCN'

root_path     = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'
pol1       = 'hv'
pol2       = 'vh'
pol3       = 'vv'
pol4       = 'hh'
pols       = [pol1, pol2, pol3, pol4]
SLC_IDs    = os.listdir(root_path)
swaths     = ['_S2_', '_S4_','_S5_','_S6_','_S3_']
slice_size = [5000, 5000]
plot_plots=False
#Load raw file
slc_id= SLC_IDs[501]
slc_id_path = root_path + slc_id +'/'

for pol in pols:
   files       = get_SM_files_from_slc_id(slc_id_path, pol)
   if files != -1:
       break  # Found files
if files == -1:
   print(f'Warning: {slc_id}:  Files not found ')
   

raw_filename = files[0]
slc_filename = files[1]
annotation_filename = files[2]   
save_path    = f'/mnt/SAR_bucket/data/sand_box/rc_testing/timing_results/'

# Define AOIs by SW & NE co-ordinates

S3_scenes = {
    'S3_A': [(29.052400, -94.901824), (29.224150, -94.529800)],
    'S3_B': [(29.188800, -94.750160), (29.270070, -94.675420)],
    'S3_C': [(29.203490, -94.662500), (29.337000, -94.555500)],
}



# Plot SLC data with Ship locations highlighted
# Get ship gps locations from csv file

# Choose SLC file


# Choose scene
scene = 'S3_C'
slc_aoi_gps  = S3_scenes[scene]
gps_AOI_lon_lat  = [(lon, lat) for lat, lon in slc_aoi_gps]
slc_aoi          = get_SM_cartesian_aoi(gps_AOI_lon_lat, files[2])
gps_aoi_polygon  = get_SM_gps_aoi(slc_aoi, files[2])

# Read SLC file and get AOI data
slc_aoi_data = get_SM_slc_aoi(slc_aoi,slc_filename)


#Plot AOI

parts      = slc_filename.split('/')[6].split('_')
plot_title = f'{parts[0]}_{parts[1]}_{parts[9]}'
fig, ax    = plot_2d_array(intensity(slc_aoi_data),
                        f'SLC AOI {scene} {np.squeeze(slc_aoi)}:{plot_title}')
# Define AOI region for ship search : will be A,B or C for S3


## Get ship data 
#  Load csv file for scene
csv_dir       = f'/mnt/SAR_bucket/data/sand_box/ship_detection/'
csv_filename  = f'Feb14_2018_AOI_Data_Scene_{scene}.csv' 
csv_file      = f'{csv_dir}{csv_filename}'
df            = pd.read_csv(csv_file)
ships_lon_lat = df[['Longitude', 'Latitude']]

## For each GPS location determine [row,col] location in [azimuth, range] 
#  Get grid information for mappings and interpolations
slc_grid_info   = get_SM_GridGPSBoundaries(annotation_filename)

cart_coords = df.apply(
    lambda row: get_SM_cartesian_point([row['Longitude'], row['Latitude']], slc_grid_info),
    axis=1
)
df[['SLC Row', 'SLC Col']] = np.array(cart_coords.tolist()).round().astype(int)


# Place markers at each expected ship position
row_offset = slc_aoi[0]
col_offset = slc_aoi[2]
scene_offset =[row_offset, col_offset]
df[['Scene Row','Scene Col']] = np.array(cart_coords.tolist()).round().astype(int) - scene_offset


ship_locations = df[['Scene Row','Scene Col']].to_numpy()
add_circles_to_figure(ax, ship_locations, radius=25, edge_colour='red', linewidth=2)

# Add ship information to each marker

# Plot RDA focused aoi
focused_data=focus_RDA_SM_aoi(slc_aoi, files, pad_with ='Zeroes',
                  plot_plots=True, azimuth_padding=1024, range_padding=1024
                           )

fig, ax = plot_2d_array(intensity(focused_data),f'Focused Raw AOI: {scene} {np.squeeze(slc_aoi)}:{plot_title}')

add_circles_to_figure(ax, ship_locations, radius=25, edge_colour='red', linewidth=2)
#track_and_annotate_vessels(ax, df, interval_sec=0.25)
interactive_vessel_annotator(ax,df)

#  .... Or could just add labels
# .... need to add ship tracks as well
# .... plot aoi + 10% , outline aoi, plot ships and label ships in buffer
# region as well
# .... need to segment slc to identify numbers of ships and compare with AIS