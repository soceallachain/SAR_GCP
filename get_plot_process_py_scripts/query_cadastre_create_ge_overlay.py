#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:14:27 2026

query_cadastre_create_ge_overlay.py

Accepts SLC_ID and determines query time then gets file gps
boundaries and uses time and boundaries to query Marine Cadastre
for all vessels broadcasting AIS with in the ROI defined by the 
file gps boundaries




@author: joe
"""

from query_cadastre_and_predict_positions import *


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
from modules.chunk_processing_module import *
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

pol1        = 'hv'
pol2        = 'vh'
pol         = [pol1, pol2]
SLC_IDs     = os.listdir(root_path)
#swaths      = ['_S1_', '_S3_', '_S6_']
swaths      = ['_S6_','_S3_']


slc_id=SLC_IDs[500] # S1B_S3_SLC__1SDH_20180214T001739....._E777

slc_id_path = root_path + slc_id +'/'
parts       = slc_id.split('_')
slc_id_name = f'{parts[0]}_{parts[1]}_{parts[-1]}'
pol1,pol2   = pol
chunk_failed    = False
number_failed_chunks = 0
files       = get_SM_files_from_slc_id(slc_id_path, pol1)
if files == -1:
    print(f'Warning: {slc_id} missing files with pol = {pol1}. Try {pol2}')
    files = get_SM_files_from_slc_id(slc_id_path, pol2)
    if files  == -1:
       print(f'Warning: {slc_id} missing files with pol = {pol2} as well. Definitely missing! ')
raw_file               = files[0]
slc_file               = files[1]
annotation_file        = files[2]

# Get file, grid and sub-grid  boundaries 
grids = get_SM_GridGPSBoundaries(annotation_file)
file_gps_boundary = grids[0]

# Get times of acquistion and set window 
window_size_hours = 2   # Window centred on query time

query_times = [annotation_file.split('_')[10]]
qtime       = datetime.strptime(query_times[0],'%Y%m%dT%H%M%S')
start_time  = qtime - timedelta(hours=window_size_hours//2)
end_time    = start_time + timedelta(hours=window_size_hours)
start_str   = start_time.strftime('%Y%m%dT%H%M%S')
end_str     = end_time.strftime('%Y%m%dT%H%M%S')

polygon_vertices = np.array(file_gps_boundary, dtype=float).tolist()


# Query AIS data
ais_data = query_marine_cadastre_ais(
    vertices=polygon_vertices,
    start_str=start_str,
    end_str=end_str,
)

if not ais_data.empty:
    eval_times = query_times

    # Generate predictions
    predictions = predict_ship_positions(
        ais_data=ais_data,
        query_times=eval_times,
        min_sog_threshold=0.3,
        max_extrapolation_hours=12.0,
    )

    # Generate KML overlay and open Google Earth automatically
    kml_path = export_to_kml_and_open(
        predictions=predictions,
        polygon_vertices=polygon_vertices,
        output_filename="ship_predictions.kml",
        circle_radius_m=100.0,  # 200m circle around ships
        auto_open=True,
    )


