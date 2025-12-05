#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:28:20 2025

@author: joe
"""

import rasterio
import numpy as np

import numpy as np
#import matplotlib as plt
from skimage.measure import find_contours, label
import itertools
from SAR_code_module import *
import platform
import psutil
import os

from SAR_focusing_module import *
from pypapi import events, papi_high
import time
import pickle


# Path to your ICEYE GeoTIFF file
file_path = f'/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/c85ce5de0e3da0acp'
#file_name = f'owiWindQuality.tif'
#file_name='owiMask.tif'
file_name ='bathymetry.tif'
#file_name = f'owiWindDirection.tif'
#file_name = f'owiWindSpeed.tif'

with rasterio.open(f'{file_path}/{file_name}') as src:
    # Read the first band (amplitude)
    amplitude = src.read(1)
    
    # Read additional bands if available (e.g., phase or second SLC band)
    if src.count > 1:
        band2 = src.read(2)
    
    # Metadata
    meta = src.meta
    print(meta)

# amplitude is now a 2D numpy array with shape like (rows, cols)
print(amplitude.shape)
#roi=amplitude[0:3000,0:10000]

plot_2d_array(amplitude,file_name, 'linear')

