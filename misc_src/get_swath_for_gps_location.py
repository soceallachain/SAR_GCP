#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:54:36 2025

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

from pathlib import Path

def get_swaths_boundaries(slc_path):
    slc_fnames          = []
    any_swath           = 1
    file_info           = get_slc_related_filenames(slc_path,any_swath,'vh')
    slc_fnames          = file_info[3]
    annotation_path     = file_info[2]    
    swath_gps_boundaries = getAllSwathBoundaries(slc_fnames, annotation_path)
    return swath_gps_boundaries

def get_swath_ll(lat, long, slc_path):
    gps_loc=[lat, long]
    iw_mode = get_swath(gps_loc, slc_path)
    return(iw_mode)
    

def get_swath(gps_loc, slc_path):
    swaths_boundaries = get_swaths_boundaries(slc_path)
    iw_mode           = getGrid([gps_loc[0], gps_loc[1]], swaths_boundaries) + 1
    return(iw_mode)
    


#slc_root_path = '/home/joe/Misc/tempTest/Data/SLC_Data/SLC_Public_Test/S1A_IW_SLC_1SDV_20200401T171834_20200401T171904_031937_03B005_3043'
slc_root_path = '/home/joe/Misc/tempTest/Data/SLC_Data/SLC_Public_Test/S1A_IW_GRDH_1SDV_20200401T171834_20200401T171903_031937_03B005_A062'
gps_loc=[5.6, 59.0]


IW_mode = get_swath(gps_loc, slc_root_path)