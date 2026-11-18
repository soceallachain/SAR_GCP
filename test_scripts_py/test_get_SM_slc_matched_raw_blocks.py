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
save_path     = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/'

# Vm cannot write to GCP for some reason
#save_path     = f'/home/joe/SAR_GCP/tests/slc_raw_matching/'
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

# Define AOIs by SW & NE co-ordinates
# ...now [long, lat]
S3_A = [(-94.9837, 29.0670),  (-94.8470, 29.1525)]
S3_B = [(-94.8306, 29.0862),  (-94.6690, 29.2512)]
S3_C = [(-94.6358, 29.1167),  (-94.5778, 29.3966)]
S3_scenes = [S3_A, S3_B, S3_C]

S6_A = [(-94.9837, 29.0670),  (-94.8470, 29.1525)]
S6_B = [(-94.8306, 29.0862),  (-94.6690, 29.2512)]
S6_C = [(-94.6358, 29.1167),  (-94.4837, 29.4108)]
S6_scenes = [S6_A, S6_B, S6_C]




slice_size = [5000, 5000]
plot_plots = False
n_files_per_swath = 1


#2. Get single file
#slc_id = SLC_IDs[299]  #S1A_S6_SLC__1SDH_20160611T002621_20160611T002645_011656_011D70_F9F2


#nS3 = 0

# Do for single SLC_ID
#for slc_id in [SLC_IDs[299]]:
# Do for all SLC_IDs
for slc_id in SLC_IDs:
    if '_S3_' not in slc_id and '_S6_' not in slc_id:
        continue
    elif '_S3_' in slc_id:
        swath = 'S3'
        swath_scenes = S3_scenes
    elif '_S6_' in slc_id:
        swath = 'S6'
        swath_scenes = S6_scenes

    slc_id_path = root_path + slc_id +'/'
    for pol in pols:  
       files       = get_SM_files_from_slc_id(slc_id_path, pol)
       if files != -1:
           break  # Found files
    if files == -1:
        print(f'Warning: {slc_id}:  Files not found ')      
        continue
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

        # Depending on  the ncc_offsets there may be additional azimuth offset
        # changes in which case revided slc_block_dims must be determined
        slc_block_dims=[]
        for block in range(len(slc_block_indices)):
           if block ==0:
               slc_block_dims.append([0,
                   slc_block_indices[0][1] - slc_block_indices[0][0] ]
                     )
           else:
               slc_block_dims.append([slc_block_dims[block-1][1]+1,
                          slc_block_indices[block][1]- slc_block_indices[0][0]
                                 ])

        
        # Ascending or Desceding Node
        orb_node = get_orbital_node(files[2])
        if orb_node =='Ascending':
            node = 'asc'
        elif orb_node == 'Descending':
            node ='desc'
        else:
            node = 'none'
        
        
        j=0
        aoi_polygons = []
        for scene in swath_scenes:
            #gps_AOI_lon_lat  = [(lon, lat) for lat, lon in scene]
            # .... redundant .. I know!
            gps_AOI_lon_lat  = [(lon, lat) for lon, lat in scene]
            slc_aoi          = get_SM_cartesian_aoi(gps_AOI_lon_lat, files[2])
            gps_aoi_polygon  = get_SM_gps_aoi(slc_aoi, files[2])
            slc_matched_data = get_SM_slc_matched_raw_AOIs(slc_aoi,
                                                           slc_block_indices,
                                                           slc_block_dims
                                                           )
            slc_matched_raw_aoi = slc_matched_data[0]
            start_block         = slc_matched_data[1]
            # Get raw data  
            chunk_raw_data = chunk_info[0]
            raw_aoi_data_list=[]
            for i in range(len(slc_matched_raw_aoi)):
                raw_aoi_data_list.append(chunk_raw_data[i][
                                 slc_matched_raw_aoi[i][0]:slc_matched_raw_aoi[i][1],
                                 slc_matched_raw_aoi[i][2]:slc_matched_raw_aoi[i][3]
                                                           ]
                                          )
            raw_aoi_data =np.concatenate( np.array(raw_aoi_data_list), axis=0) 
            
            # Get SLC data
            slc_aoi_data = get_SM_slc_aoi(slc_aoi,files[1])
            
               # Save data 
            if j == 0:
                scene_name ='A'
            elif j == 1:
                scene_name = 'B'
            else:
                scene_name = 'C'
                 
            np.save(f'{save_path}{raw_burst_dir}/{slc_id}_Pol_{pol}_Node_{node}_raw_{scene_name}.npy', raw_aoi_data)
            np.save(f'{save_path}{slc_burst_dir}/{slc_id}_Pol_{pol}_Node_{node}_slc_{scene_name}.npy', slc_aoi_data)
            j+=1

            print(f'Completed {slc_id}')
            aoi_polygons.append(gps_aoi_polygon)
            
        print(f' Scene A GPS Polygon : {aoi_polygons[0]} ')
        print(f' Scene B GPS Polygon : {aoi_polygons[1]} ')
        print(f' Scene C GPS Polygon : {aoi_polygons[2]} ')
            
            
            #plot raw data & slc date amd range compressed and fully focused
            
            #Compare slc to fully focused using NCC
