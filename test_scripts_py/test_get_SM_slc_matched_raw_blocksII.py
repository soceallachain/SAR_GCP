#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:13:55 2026

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
#from modules.chunk_processing_module import *
from modules.SAR_focusing_module import *
from modules.image_processing_module import *
from scipy.ndimage import zoom

#Set DEBUG mode if do not want to save files
DEBUG = True
# Get list of SLC_IDs

#test_dir   = f'Houston_SM_2019_20'
test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/jc_data/{test_dir}/'
#root_path  = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'
#test_dir   = f'SaoPaulo_SM_2019_20'
#root_path  = f'/mnt/SAR_bucket/data/SaoPaulo_SM/{test_dir}/'

#test_dir   = f'S1A_D_S1_S6_Samples'
root_path     = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'
#save_path     = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/'

# Vm cannot write to GCP for some reason
save_path     = f'/home/joe/SAR_GCP/tests/slc_raw_matching/'
raw_burst_dir = f'raw_burst'
slc_burst_dir = f'slc_burst'
rc_burst_dir  = f'rc_burst'

pol1       = 'hv'
pol2       = 'vh'
pol3       = 'vv'
pol4       = 'hh'
pols       = [pol1, pol2, pol3, pol4]
SLC_IDs    = os.listdir(root_path)
swaths     = ['_S2_', '_S4_','_S5_','_S6_','_S3_']

slice_size = [5000, 5000]
plot_plots = False
n_files_per_swath = 1


#2. Get single file
#slc_id = SLC_IDs[500]


nS3 = 0

# Do for all SLC_IDs
#for slc_id in SLC_IDs:
    
# Do for single SLC_ID
for slc_id in [SLC_IDs[500]]:
    if '_S3_' not in slc_id:
        continue
    
    if nS3 == 50:
        break
    
    slc_id_path = root_path + slc_id +'/'
    nS3+=1     
    
    for pol in pols:  
       files       = get_SM_files_from_slc_id(slc_id_path, pol)
       if files != -1:
           break  # Found files
    if files == -1:
        print(f'Warning: {slc_id}:  Files not found ')      
    else:
        # Get block raw data and metadata, and time-based-offsets
        scene_files = files
        chunk_info  = process_file_for_chunks(scene_files)
        timing_based_offsets = chunk_info[2]
        slc_block_dims  = get_SM_slc_matched_chunk_dimensions(timing_based_offsets)
        ncc_offsets = process_chunks_for_offsets(chunk_info, slice_size, plot_plots)
        slc_block_indices = get_SM_slc_block_offsets(ncc_offsets,
                                                    timing_based_offsets
                                                    )
        
        # Ascending or Desceding Node
        orb_node = get_orbital_node(files[2])
        if orb_node =='Ascending':
            node = 'asc'
        elif orb_node == 'Descending':
            node ='desc'
        else:
            node = 'none'

        aoi_polygons = []
        S3_scenes = [
                     ("A", [(29.052400, -94.901824), (29.224150, -94.529800)]),
                     ("B", [(29.188800, -94.750160), (29.270070, -94.675420)]),
                     ("C", [(29.203490, -94.662500), (29.337000, -94.555500)]),
                      ]

        for scene_name, scene in S3_scenes:
            gps_AOI_lon_lat = [(lon, lat) for lat, lon in scene]

            slc_aoi = get_SM_cartesian_aoi(gps_AOI_lon_lat, files[2])
            gps_aoi_polygon = get_SM_gps_aoi(slc_aoi, files[2])

            slc_matched_data = get_SM_slc_matched_raw_AOIs(
                slc_aoi,
                slc_block_indices,
                slc_block_dims,
            )
            slc_matched_raw_aoi = slc_matched_data[0]

            chunk_raw_data = chunk_info[0]

            raw_aoi_data_list = [
                chunk_raw_data[i][
                    slc_matched_raw_aoi[i][0]:slc_matched_raw_aoi[i][1],
                    slc_matched_raw_aoi[i][2]:slc_matched_raw_aoi[i][3],
               ]
               for i in range(len(slc_matched_raw_aoi))
            ] 

            raw_aoi_data = np.concatenate(raw_aoi_data_list, axis=0)
            slc_aoi_data = get_SM_slc_aoi(slc_aoi, files[1])
            aoi_polygons.append(gps_aoi_polygon)

        raw_path = (
            f"{save_path}{raw_burst_dir}/"
            f"{slc_id}_Pol_{pol}_Node_{node}_raw_{scene_name}.npy"
        )  
        slc_path = (
            f"{save_path}{slc_burst_dir}/"
            f"{slc_id}_Pol_{pol}_Node_{node}_slc_{scene_name}.npy"
        )

        np.save(raw_path, raw_aoi_data)
        np.save(slc_path, slc_aoi_data)



