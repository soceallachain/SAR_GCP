#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:13:10 2025

@author: joe
"""


import multiprocessing


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


SLC_ID       = 'S1A_IW_SLC_1SDV_20200401T171834_20200401T171904_031937_03B005_3043'
burst_number = 4
chip_number  = 16
chip_size    = [512, 3072]
slc_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming/SLC_files"



source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming"   # Change to your source directory
#destination_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Test3/Sea"
destination_dir = source_dir
# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)


# Get all .npy files in the source directory except the binary_mask files
#files = glob.glob(os.path.join(source_dir, "*.npy"))
files = [f for f in glob.glob(os.path.join(source_dir, "*.npy")) if '_binary_mask_' not in os.path.basename(f) and '_N' not in os.path.basename(f)]

filename=files[5]

#plot chip before normalization

chip_data = np.load(filename)
normed_chip_data = normalise(filename,1,-1)
if '_real_' in filename:
    bin_fname =filename.replace('_real_','_binary_mask_')
else:
    bin_fname =filename.replace('_imag_','_binary_mask_')
bin_chip_data = np.load(bin_fname)

clahed_normed_data = apply_clahe_to_data(normed_chip_data)

fig1, ax1 = plt.subplots(4,1)
ax1[0].imshow(chip_data, vmin=chip_data.min(),vmax = chip_data.max(), aspect = 'auto')
ax1[1].imshow(normed_chip_data, vmin=normed_chip_data.min(),vmax = normed_chip_data.max(), aspect = 'auto')
ax1[3].imshow(bin_chip_data, vmin=bin_chip_data.min(),vmax = bin_chip_data.max(), cmap='binary_r', aspect = 'auto')
ax1[2].imshow(clahed_normed_data, vmin=clahed_normed_data.min(),vmax = clahed_normed_data.max(), aspect = 'auto')
ax1[0].set_title(f'Chip - No normalization')
ax1[1].set_title(f'Chip Normalized')
ax1[3].set_title(f'Binary Mask Chip')
ax1[2].set_title(f'Chip - CLAHEd & Normalized')  

fig1.suptitle(f'{os.path.basename(filename)}', y=0.95)
plt.show()


import os

def extract_product_burst_and_chip_information(file):
    if not os.path.exists(file):
        print(f'{file} not found ....')
    
    else:
        filename     = os.path.basename(file)
        parts        = filename.split("_")
        product_id   = "_".join(parts[0:10])
        burst_index  = next(i for i, part in enumerate(parts) if "burst#" in part)
        chip_index   = next(i for i, part in enumerate(parts) if "chip#" in part)
        burst_number = parts[burst_index].split("#")[1]
        chip_number  = parts[chip_index].split("#")[1]
        return product_id, burst_number, chip_number
    



slc_dir ='/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming/SLC_files'

a=extract_product_burst_and_chip_information(filename)
slc_id =a[0].replace('SLC__','SLC_')
burst_number=int(a[1])
chip_number=int(a[2])

# get SLC chip - need to re-sample to match RAW sampling frequency

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
chip_data = get_slc_chip_and_bursts(slc_id, slc_dir, burst_number, chip_number, freq_scale_factor)

plot_chip(chip_data[0], slc_id, burst_number, chip_number)



a,b,c = get_slc_burst_with_ships(slc_id, slc_dir, burst_number)
burst = a
burst_with_tagged_ships = b
burst_binary = c

fig_1, ax_1 =  plot_output_of_get_slc_burst_data(slc_id, burst_number,
                                  burst, 
                                  burst_with_tagged_ships, 
                                  burst_binary)

draw_chip(burst, chip_number, chip_size,ax_1)




"""
# Use multiprocessing to speed up file renaming
if __name__ == "__main__":
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(normalise, files)

    print("File renaming completed!")
"""
