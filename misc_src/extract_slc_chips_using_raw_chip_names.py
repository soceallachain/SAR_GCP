#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:52:14 2025

@author: joe
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

freq_scale_factor = sampling_freq_mul 

slc_dir         ='/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming/SLC_files'
chip_source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming"   # Change to your source directory
destination_dir ='/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming/slc_chips'
# Get all '*_real_*.npy files in the source directory except the binary_mask files and imag files
#files = glob.glob(os.path.join(source_dir, "*.npy"))
files = [f for f in glob.glob(os.path.join(chip_source_dir, "*.npy")) 
          if '_binary_mask_' not in os.path.basename(f) 
          and '_imag_' not in os.path.basename(f) 
          and '_N' not in os.path.basename(f)
          ]

slc_names=[]
for f in files:
    print(f'Filename is: {f}')
"""
    file_information = extract_product_burst_and_chip_information(f)
    slc_id = file_information[0].replace('SLC__', 'SLC_')
    burst_number = int(file_information[1])
    chip_number  = int(file_information[2])
    chip_data    = get_slc_chip_and_bursts(slc_id, slc_dir, burst_number, 
                            chip_number, freq_scale_factor
                            )
    slc_chip_name =f'{destination_dir}/{slc_id}_{sar_mode}_IW{swath_of_interest}_slc_burst#{burst_number}_chip#{chip_number}.npy'
    print(f'SLC_Chip Name is: {slc_chip_name}')
    slc_names.append(slc_chip_name)
    #np.save(slc_chip_name, chip_data[0])
"""   
print(f'{slc_names}') 
