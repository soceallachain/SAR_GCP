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
file_path = '/home/joe/Work/Projects/RSAR/Datasets/Iceye/Iceye_NorthSea/ICE_OPER_XN_SM__GRD_20190922T034822_N52-501_E003-547_01/ICEYE_X5_GRD_SM_10346_20190922T034813.tif'

with rasterio.open(file_path) as src:
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
roi=amplitude[0:3000,0:10000]

plot_2d_array(roi,'ICEYE_X5_GRD_SM_10346_20190922T034813.tif -[0:3000,0:10000]')

