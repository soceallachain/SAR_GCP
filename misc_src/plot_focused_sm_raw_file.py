#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:24:06 2026

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
from SAR_code_module import *
from SAR_focusing_module import *
from scipy.ndimage import zoom
#filepath = "../data/sao_paulo/"
#filename = "s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
filepath = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_RAW__0SDH_20180801T001745_20180801T001811_012062_01635C_9872.SAFE/"
#filepath = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S6_RAW_0SDV_20210226T214354_20210226T214424_036766_045287_FD68/RAW/S1A_S6_RAW__0SDV_20210226T214354_20210226T214424_036766_045287_FD68.SAFE/"

#filename = "s1a-s6-raw-s-vv-20210226t214354-20210226t214424-036766-045287.dat"
filename  = "s1b-s3-raw-s-hv-20180801t001745-20180801t001811-012062-01635c.dat"
annot_filepath = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A.SAFE/annotation/"
annot_filename = "s1b-s3-slc-hv-20180801t001746-20180801t001810-012062-01635c-002.xml"
inputfile = filepath+filename
l0file = sentinel1decoder.Level0File(inputfile)
l0file.packet_metadata
l0file.ephemeris

annot_file = annot_filepath+annot_filename
selected_burst = 8
selection = l0file.get_burst_metadata(selected_burst)
radar_data = l0file.get_burst_data(selected_burst)

# Want to focus entire SM file
# Break into more manageable sizes

az_tile_length  = 2**12
number_of_tiles = int(np.ceil(radar_data.shape[0]/az_tile_length))

                   
#Set Offsets to access sub-images
range_offset  = 0
range_length  = radar_data.shape[1]-range_offset
azimuth_offset= 0
azimuth_length= az_tile_length - azimuth_offset

roi=np.array([range_offset, range_length, azimuth_offset, azimuth_length])

# Do full focusing
rma='rma'

ds_slc = []
# Extract tiles, focus, downsample and then re-assemble to generate low 
# resolution image
for i in range(0, number_of_tiles):
#for i in range(0, 2):    

    if i==number_of_tiles-1:
        
        last_tile = radar_data[i*az_tile_length:radar_data.shape[0],
                               0:radar_data.shape[1]]
        gap = az_tile_length - last_tile.shape[0]
        rd = np.pad(last_tile, ((gap,0),(0,0)), mode = 'constant')
        
    else:
        rd=radar_data[i*az_tile_length:(az_tile_length)*(i+1),
                      0:radar_data.shape[1]]
        
    sm_focused_data = focus_SM_burst(selection,rd, roi, l0file,rma)
    downsampled_data = zoom(sm_focused_data, zoom=(1/8, 1/10), order=1)
    ds_slc.append(downsampled_data)
    
#downsampled_slc = np.array(ds_slc)
downsampled_slc = np.vstack(ds_slc)

# Plot final image
plt.gca().set_aspect('auto')
plt.figure(figsize=(6,6))
#plt.title("Sentinel-1 Processed SAR Image")
plt.title('_'.join(filename.split('-')[0:6]) +'_Downsampled_8x_10x')
#plt.imshow(abs(rd_data[:,:]), vmin=0, vmax=2000, origin='lower')
#plt.imshow(np.square(abs(sm_focused_data)), origin='lower', norm=colors.LogNorm(vmin=5000, vmax=1000000))
#plt.imshow(np.square(abs(downsampled_data)), origin='lower', norm=colors.LogNorm(vmin=300, vmax=10000))
plt.imshow(np.square(abs(downsampled_slc)), origin='lower', norm=colors.LogNorm(vmin=5000, vmax=1000000))
plt.xlabel("Range (samples)")
plt.ylabel("Azimuth (samples)")

plt.show()


