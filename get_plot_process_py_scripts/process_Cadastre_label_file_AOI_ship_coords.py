#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:29:25 2026

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
label_dir = f'Labels'
root_path     = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'
pol1       = 'hv'
pol2       = 'vh'
pol3       = 'vv'
pol4       = 'hh'
pols       = [pol1, pol2, pol3, pol4]
#SLC_IDs    = os.listdir(root_path)
swaths     = ['_S2_', '_S4_','_S5_','_S6_','_S3_']
slice_size = [5000, 5000]
plot_plots=False
#Load raw file
#slc_id= SLC_IDs[500]

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


#Sea_scene = [(-95.04925, 29.05383), (-95.00629, 29.11621)]


def get_raw_slc_files(slc_id):
    
    slc_id_path = root_path + slc_id +'/'
    
    for pol in pols:
       files       = get_SM_files_from_slc_id(slc_id_path, pol)
       if files != -1:
           break  # Found files
    if files == -1:
       print(f'Warning: {slc_id}:  Files not found ')
       
    return files





def get_slc_row_col_and_scene_row_col(df_row):
    slc_id = df_row['scene_name']
    files = get_raw_slc_files(slc_id)
    raw_file = files[0]
    slc_file = files[1]
    annot_file = files[2]
    gps_lon_lat = df_row[['longitude', 'latitude']]
    
    slc_row_col = np.round(get_SM_cartesian_point_using_annotation(gps_lon_lat, 
                                                          annot_file)
                           ).astype(int)
    
    if '_S3_' in slc_id:
        scene_AOIs = S3_scenes
    elif '_S6_' in slc_id:
        scene_AOIs = S6_scenes
        
    else:
        print(f' Error : Neither S3 or S6 scene!')
        
    found = False    
    for i, scene in enumerate(scene_AOIs):
        gps_AOI_lon_lat  = [(lon, lat) for lon, lat in scene]
        scene_AOI = get_SM_cartesian_aoi(gps_AOI_lon_lat, files[2])
        gps_aoi_polygon  = get_SM_gps_aoi(scene_AOI, files[2])

        if isPointInROI(gps_lon_lat, gps_aoi_polygon):
            print(f'Point {gps_lon_lat} in {scene_AOI}')
            found = True
            # get offset in to slc file for scene
            row_col_offset = np.round(get_SM_cartesian_point_using_annotation(
                                            gps_AOI_lon_lat[0], annot_file)
                                      ).astype(int)
            scene_row_col = slc_row_col - row_col_offset
            break
    
    if found:
        if i==0:
            found_scene ='A'
        elif i==1:
            found_scene ='B'
        elif i==2:
            found_scene ='C'
        else:
            found_scene ='Sea'
    else:
        found_scene ='Z'
    
    return slc_row_col, scene_row_col, found_scene




def process_scene_group(group):
    slc_id = group.name
    files = get_raw_slc_files(slc_id)

    if files == -1:
        group['slc_row'] = np.nan
        group['slc_col'] = np.nan
        group['aoi_row'] = np.nan
        group['aoi_col'] = np.nan
        group['AOI_name'] = 'Z'
        return group

    annot_file = files[2]

    # Select candidate scenes
    if '_S3_' in slc_id:
        scene_AOIs = S3_scenes
    elif '_S6_' in slc_id:
        scene_AOIs = S6_scenes
    else:
        scene_AOIs = []

    # 1. Precalculate polygons and offsets ONCE for this entire scene
    scene_data = []
    scene_labels = ['A', 'B', 'C', 'Sea']
    for idx, scene in enumerate(scene_AOIs):
        gps_AOI_lon_lat = [(lon, lat) for lon, lat in scene]
        scene_AOI = get_SM_cartesian_aoi(gps_AOI_lon_lat, annot_file)
        gps_aoi_polygon = get_SM_gps_aoi(scene_AOI, annot_file)
        row_col_offset = np.round(
            get_SM_cartesian_point_using_annotation(
                gps_AOI_lon_lat[0], annot_file
            )
        ).astype(int)

        label = scene_labels[idx] if idx < len(scene_labels) else 'Sea'
        scene_data.append((label, gps_aoi_polygon, row_col_offset))

    # 2. Iterate only over the coordinate rows belonging to this scene
    slc_rows, slc_cols = [], []
    aoi_rows, aoi_cols = [], []
    aoi_names = []

    for _, row in group.iterrows():
        gps_pt = (float(row['longitude']), float(row['latitude']))

        # Point coordinates in full SLC
        slc_pt = np.round(
            get_SM_cartesian_point_using_annotation(gps_pt, annot_file)
        ).astype(int)

        matched_scene = 'Z'
        aoi_rc = [np.nan, np.nan]

        for label, poly, offset in scene_data:
            if isPointInROI(gps_pt, poly):
                matched_scene = label
                aoi_rc = slc_pt - offset
                break

        if matched_scene == 'Z':
            slc_rows.append(np.nan)
            slc_cols.append(np.nan)
            aoi_rows.append(np.nan)
            aoi_cols.append(np.nan)
        else:
            slc_rows.append(slc_pt[0])
            slc_cols.append(slc_pt[1])
            aoi_rows.append(aoi_rc[0])
            aoi_cols.append(aoi_rc[1])

        aoi_names.append(matched_scene)

    group['slc_row'] = slc_rows
    group['slc_col'] = slc_cols
    group['aoi_row'] = aoi_rows
    group['aoi_col'] = aoi_cols
    group['AOI_name'] = aoi_names
    return group







#1. Open csv file
csv_root_path = f'/mnt/SAR_bucket/data/Houston_SM/'
csv_filename  = f'Houston_AIS_Cadastre_Scene_Labels.csv'
csv_file      = f'{csv_root_path}{label_dir}/{csv_filename}'
df            = pd.read_csv(csv_file)



# Apply group-wise to the entire DataFrame
df = df.groupby('scene_name', group_keys=False).apply(process_scene_group)

# Save the updated DataFrame to CSV
df.to_csv(f'{csv_root_path}{label_dir}/{csv_filename}_AOI_ship_coords.csv', index=False)



#2. For each lon lat get row col in slc
#    For each row determin the row/col position for each gps location


#3. Define scenes of interest

#4. For each lon lat, for each scene test if belongs to scene and label
"""
#5. Get scene offsets and get scene row and col
slc_row_col, scene_row_col, aoi_name =get_slc_row_col_and_scene_row_col(df_row)
df_row['slc_row']= slc_row_col[0]
df_row['slc_col']= slc_row_col[1]
df_row['aoi_row']= scene_row_col[0]
df_row['aoi_col']= scene_row_col[1]
df['AOI_name'] = aoi_name
"""
