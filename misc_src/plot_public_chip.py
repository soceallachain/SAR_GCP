#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:33:08 2025

Code to plot SLC chip of interest
Requires SLC_Product_ID, Burst Number and Chip Number
Note: will have to re-sample SLC to RAW sampling frequency if want to 
      compare directly with raw chips
      
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





SLC_ID       = 'S1A_IW_SLC_1SDV_20200401T171834_20200401T171904_031937_03B005_3043'
burst_number = 4
chip_number  = 16
chip_size    = [512, 3072]
slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data/SLC_Public_Test"


a,b,c = get_slc_burst_with_ships(SLC_ID, slc_path, burst_number)
burst = a
burst_with_tagged_ships = b
burst_binary = c

fig_1, ax_1=plot_output_of_get_slc_burst_data(SLC_ID, burst_number,
                                  burst, 
                                  burst_with_tagged_ships, 
                                  burst_binary)

chip_data = get_slc_chip_and_bursts(SLC_ID, slc_path, burst_number, chip_number)
draw_chip(burst_with_tagged_ships, chip_number, chip_size, ax_1)

plot_chip(chip_data[0], SLC_ID, burst_number, chip_number)







