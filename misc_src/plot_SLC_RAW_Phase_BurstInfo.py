#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:48:54 2025

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

#SLC_Product_ID ='S1B_IW_SLC__1SDV_20200619T051035_20200619T051102_022098_029EF7_F467'
SLC_Product_ID ='S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B'
burst_num_to_plot =3
plot_IW2_RAW_SLC(SLC_Product_ID, burst_num_to_plot)


import subprocess

# Display burst Ship and Sea Information
bash_script = "/home/joe/Misc/bash_scripts/get_burst_sea_info.sh"
arg1 = SLC_Product_ID

try:
    result = subprocess.run(["bash", bash_script, arg1], check=True, text=True, capture_output=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: {e.stderr}")
