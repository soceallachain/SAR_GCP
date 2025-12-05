#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:16:06 2025

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
from getch import getch
from  SAR_code_module import *





ship_location = [12213, 7796]
chip_dims     = [512,3072]
burst_dims    = [1424, 21516]

burst_chip_number = get_chip_number(ship_location, chip_dims, burst_dims)
print(f'Chip number = {burst_chip_number}')


#binary_mask = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Chip_Data/S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B_RAW_IW2_binary_mask_burst#1_n11.npy'
root_path = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Chip_Data/'
name = 'S1B_IW_SLC__1SDV_20201117T064109_20201117T064136_024301_02E347_3F94_RAW_IW2_binary_mask_burst#1'

i =1 
while True:
    #fname = f'{root_path}{name}_n{i}_#Ships1.npy'
    fname = f'{root_path}{name}_n{i}.npy'
    
    try:
        mask = np.load(fname)
    except FileNotFoundError:
        print(f"File {fname} not found. Exiting loop.")
        break

    # Display the image
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title(f'{fname}')   
    plt.show()
    
    ###############################################################
    ## Keyboard input does not work well!
    key = input("Press Space to continue, or Enter to quit: ")
    i += 1
    plt.close('all')  # Close all figures to avoid memory issues
    key = getch()
    if key == " ":  # Space key
       print(f"Displaying {fname}...")
       
    elif key == "\n":  # Enter key
       print("Enter pressed. Exiting...")
       break
    
    
    if key == "":  # Enter key
        print("Exiting the loop.")
        break
    elif key == " ":  # Space bar
        print(f"Displaying {fname}...")
    else:
        print("Invalid input, try again.")
