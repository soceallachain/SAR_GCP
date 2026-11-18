#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:40:37 2026

@author: joe
"""


import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d
from modules.SAR_code_module import *
from modules.SAR_focusing_module import *
from scipy.ndimage import zoom
import mpld3

root_path   = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_OCN/'

SLC_IDs     = os.listdir(root_path)

# Check all SLC_IDs in directory
for slc_id in SLC_IDs:


   slc_id_path = root_path + slc_id +'/'
   # Check all polarisations
   for pol in ('vh','hv','vv','hh'):
       files = get_SM_files_from_slc_id(slc_id_path, pol)
       if files != -1:
           break
       
   if files == -1:
       print(f'Warning: {slc_id} missing files')
   else:    
       raw_file    = files[0]
       slc_file    = files[1]
       annot_file  = files[2]
   
       swath = annot_file.split('/SLC/')[0].split('_SLC')[0].split('_')[-1]
       SLC_CRC = annot_file.split('/SLC/')[0].split('_')[-1]

       print(f" {SLC_CRC} {swath} Node: {get_orbital_node(annot_file)} ")