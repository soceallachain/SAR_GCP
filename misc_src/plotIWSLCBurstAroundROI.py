
"""
Script to read IW SLC files and get burst boundary information for swath of 
annotation file.
Determines what burst a given GPS location is in, if any
Plots region around GPS location
"""

import os
import io
import re
import sys
import random
import shutil
# data manipulation
from matplotlib.patches import Rectangle

import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.precision', 12)

sys.path.insert(0, '/home/joe/Misc')

from  SAR_code_module import *




path ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
# If set overwrite all existing SAR data overwritten else only overwrite particular mode being acquired
OVER_WRITE_ALL = False

GRD_dir    ="Alexandria Egypt_1"
SAFE_dir   ="product/S1A_IW_SLC__1SDV_20240915T160541_20240915T160608_055678_06CC8D_85B8.SAFE"
SLC_fnames = ["s1a-iw1-slc-vh-20240915t160543-20240915t160608-055678-06cc8d-001.tiff",
              "s1a-iw2-slc-vh-20240915t160541-20240915t160606-055678-06cc8d-002.tiff",
              "s1a-iw3-slc-vh-20240915t160542-20240915t160607-055678-06cc8d-003.tiff"]
         


#SAFE_dir   ="product/S1A_IW_SLC__1SDV_20240805T155728_20240805T155755_055080_06B602_DF20.SAFE"
#SLC_fnames = ["s1a-iw1-slc-vh-20240805t155730-20240805t155755-055080-06b602-001.tiff",
#              "s1a-iw2-slc-vh-20240805t155728-20240805t155754-055080-06b602-002.tiff",
#              "s1a-iw3-slc-vh-20240805t155729-20240805t155755-055080-06b602-003.tiff"]

src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fnames[2]


if os.path.exists(src_file):
    print(f"{src_file} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


annotation_file = path + GRD_dir +'/SLC/' + SAFE_dir  + "/annotation/" + change_extension(SLC_fnames[2],"xml")
info=getSwathBurstGridGPSBoundaries(annotation_file)
Alexandria_port = [29.876964, 31.190277]
Burst=getGrid(Alexandria_port, info[2])
print(f' Burst containing Alexandria [GPS Coords: {Alexandria_port} is Burst #{Burst}')

SLC_ref_grid=np.array(info[1][Burst])
gps_grid = np.array(info[2][Burst])

Alexandra_SLC_coords=np.round(extrapolate(Alexandria_port, gps_grid, SLC_ref_grid))


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")
max_line=2000
min_line=0
max_range=7000
min_range=2000
map_grid=np.array([(0,0), (0, max_line-min_line), 
                   (max_range-min_range, max_line-min_line), (max_range-min_range,0)])
SLC_grid =np.array([(min_line, min_range), (max_line, min_range), 
                    (max_line, max_range), (min_line, max_range)])
map_point=np.round(extrapolate(Alexandra_SLC_coords,SLC_grid,map_grid))
rec_xsize = 50
rec_ysize = 50
rec_xpos  = map_point[0] - rec_xsize//2
rec_ypos  = map_point[1] - rec_ysize//2
rect      = Rectangle((rec_xpos, rec_ypos), rec_xsize, rec_ysize, linewidth=2, edgecolor='red', facecolor='none')

                    
sub_array=array[min_line:max_line, min_range:max_range]
im=plt.imshow(abs(np.flip(np.flip(sub_array,1),1)), origin='lower', norm=colors.LogNorm(vmin=30, vmax=500))
# Show the plot
#plt.colorbar(im)
plt.show()


