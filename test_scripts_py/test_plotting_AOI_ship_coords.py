#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:16:12 2026

@author: joe
"""

# Load libraries and other sundries


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



def get_raw_slc_files(slc_id):
    
    slc_id_path = root_path + slc_id +'/'
    
    for pol in pols:
       files       = get_SM_files_from_slc_id(slc_id_path, pol)
       if files != -1:
           break  # Found files
    if files == -1:
       print(f'Warning: {slc_id}:  Files not found ')
       
    return files





pol1       = 'hv'
pol2       = 'vh'
pol3       = 'vv'
pol4       = 'hh'
pols       = [pol1, pol2, pol3, pol4]

test_dir   = f'Houston_SM_No_OCN'
label_dir = f'Labels'
root_path     = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'


# Define label file  Open csv file
csv_root_path = f'/mnt/SAR_bucket/data/Houston_SM/'
slc_aoi_path  = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/slc_burst/'
csv_filename  = f'Houston_AIS_Cadastre_Scene_Labels.csv_AOI_ship_coords_20261102180236.csv'
csv_file      = f'{csv_root_path}{label_dir}/{csv_filename}'
df            = pd.read_csv(csv_file)



# Define scenes of interest:
    
# Define AOIs by SW & NE co-ordinates
# ...now [long, lat]
Sea_scene = [(-95.0493, 29.0538), (-95.0063, 29.1162)]


S3_A = [(-94.9837, 29.0670),  (-94.8470, 29.1525)]
S3_B = [(-94.8306, 29.0862),  (-94.6690, 29.2512)]
S3_C = [(-94.6358, 29.1167),  (-94.5778, 29.3966)]
S3_scenes = [S3_A, S3_B, S3_C, Sea_scene]

S6_A = [(-94.9837, 29.0670),  (-94.8470, 29.1525)]
S6_B = [(-94.8306, 29.0862),  (-94.6690, 29.2512)]
S6_C = [(-94.6358, 29.1167),  (-94.4837, 29.4108)]
S6_scenes = [S6_A, S6_B, S6_C, Sea_scene]


# Get list of scenes
# Exclude missing values before taking unique items as a list:
slc_ids = df['scene_name'].dropna().unique().tolist()

# Pick one!
slc_id =slc_ids[2]

# Pick AOI
aoi_name ='C'

# Get ships in slc_id AOI 'A'
ship_aoi_row_col = df[
    (df['scene_name'] == slc_id) & (df['AOI_name'] == aoi_name)
][['aoi_row', 'aoi_col']].to_numpy()

# Get slc file
files = get_raw_slc_files(slc_id)
slc_file = files[1]


# get aoi data
aoi_file = f'{slc_aoi_path}{slc_id}_Pol_hv_Node_asc_slc_{aoi_name}.npy'


if os.path.exists(aoi_file):
    print("File exists")
    aoi_data = np.load(aoi_file)
    ax=plot_2d_array(intensity(aoi_data))
    # Plot ships on aoi plot
    add_circles_to_figure(ax[1], ship_aoi_row_col, radius=25, edge_colour='red', linewidth=2)
else:
    print(f'AOI file not found - check existence or spelling')

df1=df[
    (df['scene_name'] == slc_id) & (df['AOI_name'] == aoi_name)
][['MMSI', 'length','longitude','latitude','slc_row','slc_col', 'aoi_row', 'aoi_col']]


# Returns a new DataFrame with renamed columns
df1 = df1.rename(columns={'length': 'Length', 'aoi_row': 'Scene Row','aoi_col':'Scene Col'})


interactive_vessel_annotator(ax[1],df1)




