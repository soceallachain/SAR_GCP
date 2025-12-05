#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:44:07 2025

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




def get_slc_burst_data_with_ships(SLC_ID, burst_to_plot, swath_of_interest=2, polarization ='vh'):
    """
    Function to return burst data, burst data with ships
    tagged and the binary mask all as np-arrays

    Parameters
    ----------
    SLC_ID : TYPE
        DESCRIPTION.
    burst_to_plot : TYPE
        DESCRIPTION.
    swath_of_interest : TYPE, optional
        DESCRIPTION. The default is 2.
    polarization : TYPE, optional
        DESCRIPTION. The default is 'vh'.

    Returns
    -------
    None.

    """
    scene_id = get_scene_id(SLC_ID)
    if scene_id:
        print(f'Scene_id is: {scene_id}')
    else:
        print(f'scene_id failed')
        sys.exit()
    #swath_of_interest = 2
    #polarization      ='vh'
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index

    # Get SLC file to match raw based on mapping file
    scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"

    #scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_train.csv'
    mapping_df         = pd.read_csv(scene_mapping_file)

    #raw_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Train"
    #slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_TrainingData"
    raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
    slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"


    #save_path = "/mnt/smbshare/tempTest/Data/Burst_Training_Data/" plt.subplots
    #save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/raw_burst_training/"
    save_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/"

    if not (os.path.exists(raw_path)):
	    print(f' Path to {raw_path} inaccessible: Exiting!')
	    sys.exit()
    else:
	    print(f' Path to {raw_path} found')

    for raw_directory in [scene_id]:
        raw_root_file_path   = f'{raw_path}/{raw_directory}/RAW/'
        if os.listdir(raw_root_file_path) and contains_SAFE_file(raw_root_file_path):
            raw_safe_dir  = os.listdir(raw_root_file_path)
            raw_src_path  = f'{raw_root_file_path}{raw_safe_dir[0]}/'
            raw_filenames = glob.glob(f'{raw_src_path}*{polarization}*.dat')
            # Need to sort filenames to access correct .dat file
            raw_filenames.sort(reverse=True)
            raw_filename = raw_filenames[0]
            if not (os.path.exists(raw_filename[0])):
                print(f'Error: Filename {raw_filename} not found. Continuing to next raw file ....')
                continue
            else:
                raw_iw_data       = get_raw_burst_info(raw_filename)
                print(f' File {raw_filename} found. ')
        else:
            print(f'Error: Raw file path {raw_root_file_path} not found or does not contain .SAFE directory. Continuing to next raw file ......')
            continue
    	########################################################################### 
    	### Have matching SLC file
        slc_id = SLC_ID
        SLC_root_file_path = f'{slc_path}/{slc_id}'
        SLC_root_file_path = SLC_root_file_path.replace("__","_")
        #p1,p2,p3,p4,p5    = getSLCFilesFromRoot(, swath_of_interest)
        slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
        slc_src_file      = slc_file_info[0]
        annotation_file   = slc_file_info[1]
        annotation_path   = slc_file_info[2]
        SLC_fnames        = slc_file_info[3]
        SAFE_dir          = slc_file_info[4]
        get_slc_info_status = slc_file_info[5]
    
        if get_slc_info_status != 1:
    	    print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
    	    continue
    
        print(f' Annotation file is: {annotation_file}')
        print(f' Annotation_dir  is: {annotation_path}')
        print(f' SLC_fname(s) are  : {SLC_fnames}')
        print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
       	Annotation_dir = annotation_path
       	SLC_safe_id, _ = os.path.splitext(SAFE_dir)
       	label_file     = scene_mapping_file
       	#############################################################################
       	if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir)
       					   and os.path.exists(slc_src_file)):
       	    print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
       	else:
       	    print(f'ERROR: {annotation_file} or {Annotation_dir} '
       		  f' or {slc_src_file} Not found. Exiting')
       	    continue
       	    #exit(0)
       
       	# Have SLC src and annotation file and raw filename
       	###########################################################################
       
       	###########################################################################
       	# get binary masks for SLC bursts
       	ship_coords = getShipCoordinates(label_file, swath_of_interest, SLC_safe_id)
       	burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
       	IW_gps_boundary           = burst_information[0]
       	burst_boundaries          = burst_information[1]
       	burst_gps_boundaries      = burst_information[2]
       	burst_grid_boundaries     = burst_information[3]
       	burst_grid_gps_boundaries = burst_information[4]
       	number_of_bursts          = len(burst_boundaries)
       	bounding_box_size         = 100
       	border                    = 25
       	swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
       						 burst_boundaries,
       						 burst_gps_boundaries,
       						 burst_grid_boundaries,
       						 burst_grid_gps_boundaries,
       						 bounding_box_size)
       	# Choose burst of interest
       	n = len(burst_boundaries)
       	burst_counts = [burst_indices.count(i) for i in range(0, n)]
       	# Print the table
       	print("Burst Number:\t" ,  "\t".join(map(str, range(1, n + 1))))
       	print("#Ships:      \t",  "\t".join(map(str, burst_counts)))
       	burst_of_interest = burst_counts.index(max(burst_counts)) + 1   
       	#slc_burst         = burst_of_interest
       	slc_burst         = burst_to_plot
       	############################################################################
       	# Load real and imaginary components of burst
       	burst_real_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_real_burst#{slc_burst}.npy"
       	burst_imag_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_imag_burst#{slc_burst}.npy"
       	burst_bin_file  = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_binary_mask_burst#{slc_burst}.npy"
       	if os.path.exists(burst_real_file) and os.path.exists(burst_imag_file) and os.path.exists(burst_bin_file):
       	    burst_real = np.load(burst_real_file)
       	    burst_imag = np.load(burst_imag_file)
       	    burst_bin  = np.load(burst_bin_file)
       	    print(f'All burst files found')
       	else:
       	    print(f'ERROR : At least one raw burst data file not found in {save_path}. Exiting')
       	    exit(0)
       	###########################################################################
       
       
       	
       	if os.path.exists(slc_src_file):
       	    print(f"{SLC_fnames[swath_of_interest_index]} found")
       	else:
       	    print(f"ERROR: {slc_src_file} Not found. Exiting")
       	    #exit(0)
       
       	src            = rasterio.open(slc_src_file)
       	img            = src.read(1)
        #Make list of slc_burst as list expected by function
       	bursts_to_plot = [slc_burst]
       	output=get_chips_and_masks(img, SLC_safe_id,
       				 bursts_to_plot, burst_boundaries,
       				 np.array(swath_bounding_boxes),
       				 burst_indices, bounding_box_size, border
       				)
        # Need index for burst    
        burst_boundary = burst_boundaries[slc_burst - 1]
        burst_img = create_roi_from_polygon(img, burst_boundary) 
        intensity_burst_image = np.square(np.abs(burst_img))
        burst_img_with_tagged_ships = output[2]
        burst_binary_ground_truth   = output[3]
        return intensity_burst_image, burst_img_with_tagged_ships, \
               burst_binary_ground_truth       
       
    
def plot_output_of_get_slc_burst_data(SLC_safe_id, slc_burst,
                                      intensity_burst_image, final_burst_img, 
                                      final_binary_img):
    
   

    fig1, ax1 = plt.subplots(3,1)
    ax1[0].imshow(intensity_burst_image, norm=colors.LogNorm(vmin=10,vmax = 30000), aspect = 'auto')
    ax1[1].imshow(final_burst_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = 'auto')
    ax1[2].imshow(final_binary_img, cmap='Greys_r', aspect = 'auto')
    ax1[0].set_title(f'SLC Amplitude #{slc_burst} Max = {final_burst_img.max():.2f}')
    ax1[1].set_title(f'SLC Amplitude, Ships Boxed #{slc_burst} Max = {final_burst_img.max():.2f}')
    ax1[2].set_title(f'Binary Mask for  Burst #{slc_burst}')
    fig1.suptitle(f'{SLC_safe_id}', y=0.95)

    plt.show()
    return fig1, ax1[0]



def get_chip_aoi_in_burst(chip_number, chip_size, burst_size):
    chip_width  = chip_size[1]
    chip_height = chip_size[0]
    burst_width = burst_size[1]
    burst_height= burst_size[0]
        
    num_chips_per_burst_row = (burst_width -1) // chip_width +1
    row = ((chip_number-1) // num_chips_per_burst_row) + 1
    col_start = ((chip_number- 1) % num_chips_per_burst_row )*chip_width
    if burst_width - col_start < chip_width:
        col_start = burst_width - chip_width 
    col_end = col_start + chip_width -1 
    row_start = (row - 1)*chip_height
    if burst_height - row_start < chip_height:
        row_start = burst_height - chip_height
    row_end = row_start + chip_height -1
    chip_aoi = [[row_start, col_start], [row_end, col_start], 
                [row_end, col_end], [row_start, col_end]]
    
    return chip_aoi


def get_chip_image(burst, chip_number, chip_size):
    burst_size = burst.shape
    aoi = get_chip_aoi_in_burst(chip_number, chip_size, burst_size
    chip = burst[aoi[0][0]:aoi[1][0]+1,aoi[0][1]:aoi[2][1]+1]
    return chip


def draw_chip(burst, chip_number, chip_size,ax_handle):
    chip_boundary = get_chip_aoi_in_burst(chip_number, chip_size, burst_size)
    #chip = get_chip_image(burst, chip_boundary)
    x_min = min(point[1] for point in chip_boundary)
    y_min = min(point[0] for point in chip_boundary)
    # Compute width and height
    width = max(point[1] for point in chip_boundary) - x_min
    height = max(point[0] for point in chip_boundary) - y_min
    # Draw the rectangle (x, y start at top-left)
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='k', facecolor='none')
    ax_handle.add_patch(rect)
    


#SLC_Product_ID ='S1B_IW_SLC__1SDV_20200619T051035_20200619T051102_022098_029EF7_F467'
SLC_Product_ID ='S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B'
burst_num_to_plot = 1
burst_image, burst_ship_tagged_image, burst_binary_image = get_slc_burst_data_with_ships(SLC_Product_ID, burst_num_to_plot)
fig, ax = plot_output_of_get_slc_burst_data(SLC_Product_ID, burst_num_to_plot, 
                                  burst_image, burst_ship_tagged_image,
                                  burst_binary_image)

chip_size=[512,3072]
burst_size = burst_image.shape
chip_number = 15

draw_chip(burst_image,chip_number, chip_size, ax)


"""
chip_boundary = get_chip_aoi_in_burst(chip_number, chip_size, burst_size)    

x_min = min(point[1] for point in chip_boundary)
y_min = min(point[0] for point in chip_boundary)

# Compute width and height
width = max(point[1] for point in chip_boundary) - x_min
height = max(point[0] for point in chip_boundary) - y_min

# Draw the rectangle (x, y start at top-left)
rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='k', facecolor='none')
ax.add_patch(rect)

# Show the plot
#plt.show()

"""



import subprocess

bash_script = "/home/joe/Misc/bash_scripts/get_burst_sea_info.sh"
arg1 = SLC_Product_ID
"""
try:
    result = subprocess.run(["bash", bash_script, arg1], check=True, text=True, capture_output=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: {e.stderr}")
"""
