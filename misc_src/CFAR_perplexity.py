import numpy as np
from scipy.stats import norm

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

def cfar_detector(slc_data, num_train, num_guard, pfa):
    """
    CFAR detector for SLC SAR data with Gaussian-distributed clutter.
    
    Parameters:
    slc_data (np.array): Complex-valued SLC SAR data
    num_train (int): Number of training cells
    num_guard (int): Number of guard cells
    pfa (float): Desired probability of false alarm
    
    Returns:
    np.array: Binary detection map
    """
    # Convert SLC data to intensity
    intensity = np.abs(slc_data)**2
    
    # Calculate the number of cells on each side
    num_side = num_train + num_guard
    
    # Pad the intensity array
    padded_intensity = np.pad(intensity, ((num_side, num_side), (num_side, num_side)), mode='reflect')
    
    # Initialize the detection map
    detection_map = np.zeros_like(intensity, dtype=bool)
    
    # Calculate the threshold factor
    threshold_factor = norm.ppf(1 - pfa)
    
    # Perform CFAR detection
    for i in range(num_side, padded_intensity.shape[0] - num_side):
        for j in range(num_side, padded_intensity.shape[1] - num_side):
            # Extract the cell under test (CUT)
            cut = padded_intensity[i, j]
            
            # Extract the training cells
            training_cells = np.concatenate([
                padded_intensity[i-num_side:i+num_side+1, j-num_side:j-num_guard-1].flatten(),
                padded_intensity[i-num_side:i+num_side+1, j+num_guard+1:j+num_side+1].flatten(),
                padded_intensity[i-num_side:i-num_guard-1, j-num_guard:j+num_guard+1].flatten(),
                padded_intensity[i+num_guard+1:i+num_side+1, j-num_guard:j+num_guard+1].flatten()
            ])
            
            # Calculate the mean and standard deviation of the training cells
            mean = np.mean(training_cells)
            std = np.std(training_cells)
            
            # Calculate the adaptive threshold
            threshold = mean + threshold_factor * std
            
            # Perform detection
            detection_map[i-num_side, j-num_side] = cut > threshold
    
    return detection_map

# Example usage

SLC_ID ="S1B_IW_SLC_1SDV_20200619T051035_20200619T051102_022098_029EF7_F467"
burst  = 3

slc_data = get_single_burst_image_data(SLC_ID, burst)
num_train = 20
num_guard = 10
pfa = 1e-3

detection_result = cfar_detector(slc_data, num_train, num_guard, pfa)


fig2, ax2 = plt.subplots(1,2)
ax2[0].imshow(np.abs(slc_data), norm=colors.LogNorm(vmin=10,vmax = 300), aspect = 'auto')
ax2[1].imshow(detection_result, cmap='Greys_r', aspect = 'auto')
ax2[0].set_title(f'SLC Amplitude #{burst}')
ax2[1].set_title(f'CFAR Mask for  Burst #{burst}')
fig2.suptitle(f'{SLC_ID}', y=0.8)


