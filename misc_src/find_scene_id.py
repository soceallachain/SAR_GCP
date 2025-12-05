#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:20:31 2025

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

def get_scene_id(SLC_Product_ID):
    """
    Function to find scene_id associated with Product_ID
    Parameters
    ----------
    SLC_Product_ID : 
    Returns
    -------
    scene_id 

    """
    scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
    df          = pd.read_csv(scene_mapping_file)
    SLC_PRODUCT_ID = SLC_Product_ID.replace("__","_")
    filtered_df = df.loc[df["SLC_product_identifier"] == SLC_Product_ID]
    if not filtered_df.empty:
        scene_id = filtered_df["scene_id"].iloc[0]
        #print(f"scene_id  for SLC_product_identifier {SLC_Product_ID}: {scene_id}")
    else:
        scene_id=''
        #print(f"ERROR: No SLC match found for scene_id {target_scene_id}. Continuing to next raw file .....")
    
    return scene_id


SLC_ID="S1B_IW_SLC__1SDV_20201117T064109_20201117T064136_024301_02E347_3F94"
#SLC_ID="S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B"
#SLC_ID="S!B_IW_SLC"
s_id=get_scene_id(SLC_ID)    
if not s_id:
    print(f'No matching Scene_Id found for SLC_Product_ID:{SLC_ID}')
else:
    print(f'Scene_ID for SLC_product_identifier {SLC_ID}: {s_id}')
