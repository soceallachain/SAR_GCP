###############################################################################
# getNumberOfShipsInBursts.py
#
# Script to search through a directory of SAFE SLC files and return number of 
# ships in each burst of each file

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
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.precision', 12)



from  SAR_code_module import *




swath_of_interest = 2
swath_of_interest_index = swath_of_interest - 1
path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/TrainingTestData/"

#SLCTrainingFileDirectories = ["S1A_IW_GRDH_1SDV_20200116T051412_20200116T051437_030821_038933_5CD5", 
#                              "S1A_IW_GRDH_1SDV_20200519T171753_20200519T171824_032637_03C7B2_1F74"]


SLCTrainingFileDirectories = os.listdir(path)
###############################################################################
# Test with 1 file first




#SLC_root_file_path   = f'{path}/{SLCTrainingFileDirectories[8]}'
#print(f'****     SLC_root_file_path = {SLC_root_file_path}')

for SLC_dir in SLCTrainingFileDirectories:
    
    SLC_root_file_path   = f'{path}/{SLC_dir}'
    print(f'****     SLC_root_file_path = {SLC_root_file_path}')
    p1,p2,p3,p4,p5    = getSLCFilesFromRoot(SLC_root_file_path, swath_of_interest)
    src_file          = p1
    annotation_file   = p2
    annotation_path   = p3
    SLC_fnames        = p4
    SAFE_dir          = p5
    del p1,p2,p3,p4,p5  
    print(f' Annotation file is: {annotation_file}')
    print(f' Annotation_dir  is: {annotation_path}')
    print(f'SLC_fname are : {SLC_fnames}')
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    Annotation_dir = annotation_path
    SLC_safe_id, _ = os.path.splitext(SAFE_dir)
    label_file     = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
    #############################################################################
    if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir) 
                                       and os.path.exists(src_file)):
        print(f'{annotation_file} and {Annotation_dir} and {src_file} found')
    else:
        print(f'ERROR: {annotation_file} or {Annotation_dir} '
              f' or {src_file} Not found. Exiting')
        exit(0)
    
    
    ship_coords = getShipCoordinates(label_file, swath_of_interest, SLC_safe_id)
    
    
    IW_gps            = getAllSwathBoundaries(SLC_fnames, Annotation_dir)
    burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
    
    IW_gps_boundary           = burst_information[0]
    burst_boundaries          = burst_information[1]
    burst_gps_boundaries      = burst_information[2]
    burst_grid_boundaries     = burst_information[3]
    burst_grid_gps_boundaries = burst_information[4]
    
    bounding_box_size = 100
    #swath_bounding_boxes = np.array(bounding_boxes)
    swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
                                             burst_boundaries,
                                             burst_gps_boundaries,
                                             burst_grid_boundaries,
                                             burst_grid_gps_boundaries,
                                             bounding_box_size)
    
    print(f' Number of ships in Burst# 9 of {src_file} is {burst_indices.count(9-1)}')



print(f'SLC Trainig File Directory contains : {SLCTrainingFileDirectories}')

"""
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src            = rasterio.open(src_file)
img            = src.read(1)
bursts_to_plot = [5]
border         = 25     
# Put border of 25 pixels around ship chips => total size of chip is
# (box_size/2 + border)*2 = 150x150
#bursts_to_plot     =list(range(1,nBursts+1))


print(f' Burst Boundaries are : {np.array(burst_boundaries)}')
print(f' Burst GPS Boundaries are : {np.array(burst_gps_boundaries)}')

"""

"""
all_burst_chips,b,c,d,e = get_chips_and_masks(img, SLC_safe_id,
                                              bursts_to_plot,burst_boundaries,
                                              np.array(swath_bounding_boxes),
                                              burst_indices, bounding_box_size, border
                                              )
"""
##############################################################################
# save bifurcated binary ship chips
save_ship_chips=False
if save_ship_chips ==True:
    ship_chips_to_save=[]
    burst_number = bursts_to_plot[5]
    for ship in ship_chips_to_save:
        np.save(f"{SLC_safe_id}_burst#{burst_number}_ship#{ship}",
                np.array(d[ship*149:(ship+1)*149,:]).astype(np.int16))
        np.save(f"{SLC_safe_id}_burst#{burst_number}_ship#{ship}_mask", 
                np.array(e[ship*149:(ship+1)*149,:]).astype(np.uint8))
    
    
    amplitude_ship_chip = np.load(f"{SLC_safe_id}_burst#{burst_number}_ship#{ship}.npy")
    binary_ship_chip    = np.load(f"{SLC_safe_id}_burst#{burst_number}_ship#{ship}_mask.npy")
    mpl.rcParams['figure.figsize'] = [14.0, 8.0]
    fig2, ax2 = plt.subplots(1,2)
    ax2[0].imshow(amplitude_ship_chip, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
    ax2[1].imshow(binary_ship_chip,cmap='Greys_r', aspect = '10')
    plt.show()
    
    



