#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:12:36 2025

Script top perform CFAR on chips used for testing performance of Resnet18 & 34
models.
Will process each chip and classify whether ship present or not
Need to specify train and guard parameters
Assumen pFa of 1e-2

As parse through eachp chip record whether has ship(s) or not



@author: joe
"""
from skimage.measure import find_contours, label
from scipy.ndimage import label, find_objects, center_of_mass, distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import itertools
import numpy as np
import pandas as pd

from SAR_code_module import *
from SAR_focusing_module import *

from blob_module import *


def parse_filename(filename):
    # Remove the extension
    base_name = filename.rsplit('.', 1)[0]
    
    # Split by underscores
    parts = base_name.split('_')
    
    # Numeric component after last underscore
    offset_mag = parts[-1]
    #Check if chip centered, in which case no magnitude and have to re-arrange
    if offset_mag =='center':
        # String between 16th underscore (index 14) and last underscore (exclusive)
        offset_dir = offset_mag
        offset_mag = 0
    else:
        offset_dir = '_'.join(parts[16:-1])
    
    slc_filename ='_'.join(parts[0:14])
    slc_filename += '.npy'
    if '_data_' in slc_filename:
        slc_filename = slc_filename.replace('_data_','_slc_')
        crc_burst    = parts[9]+'_' + parts[13]
        ship_flag    = parts[14].replace('Ship#','')
        
    else:
        slc_filename = slc_filename.replace('_real_','_slc_')
        crc_burst    =' '
        ship_flag    = ' '
    
    return offset_dir, int(offset_mag), slc_filename, crc_burst, ship_flag

# Determine shift directions
def parse_offset_dir(offset_dir, offset_mag):
    match offset_dir:
        case "center":
            offset =[0, 0]
        case "right":
            offset = [0, offset_mag]
        case "left":
            offset = [0, -1*offset_mag]
        case "up":
            offset = [offset_mag, 0]
        case "down":
            offset = [-1*offset_mag, 0]
        case "up_right":
            offset = [int(offset_mag/np.sqrt(2)), int(offset_mag/np.sqrt(2))]
        case "up_left":
            offset = [int(offset_mag/np.sqrt(2)), -1*int(offset_mag/np.sqrt(2))]
        case "down_right":
            offset = [-1*int(offset_mag/np.sqrt(2)), int(offset_mag/np.sqrt(2))]
        case "down_left":
            offset = [-1*int(offset_mag/np.sqrt(2)), -1*int(offset_mag/np.sqrt(2))]
        case '_':
            print(f'Error: Unexpected chip shift. Exiting....')
            
    
    return offset

#Determine ship coordinates for all ships in list
def get_ship_coordinates_df(df, match_criteria):
    #find all ship locations associated with crc_burst
    #get coordinates of ship defined by 'ship_flag'    
    crc_burst_match = match_criteria[0]
    ship_flag       = match_criteria[1]
    filtered_df     = df[df['CRC Burst Name'] 
                                 == crc_burst_match].iloc[[int(ship_flag)-1]]
    df_ship_coords  = filtered_df[['Ship Row', 
                                  'Ship Col']].astype(int).values.tolist()    
    ship_coords     = np.concatenate(df_ship_coords).tolist()
    return ship_coords


# Create chip based on magnitude and direction of shift and ship coordinates
def gen_shifted_chip(src, ship_coords, offset, chip_size):
    
    burst_size = src.shape
    half_chip_row_size = chip_size[0]//2
    half_chip_col_size = chip_size[1]//2
    
    chip_row_start = ship_coords[0] - half_chip_row_size + offset[0]
    if chip_row_start < 0:
        chip_row_start = 0
    elif chip_row_start > burst_size[0] - chip_size[0]:
        chip_row_start = burst_size[0] - chip_size[0]
    chip_row_end   = chip_row_start + chip_size[0] 
    
    chip_col_start = ship_coords[1] - half_chip_col_size + offset[1]
    if chip_col_start < 0:
        chip_col_start = 0
    elif chip_col_start > burst_size[1] - chip_size[1]:
        chip_col_start = burst_size[1] - chip_size[1]
    chip_col_end   = chip_col_start + chip_size[1] 
    
    shifted_chip   = src[chip_row_start:chip_row_end, 
                         chip_col_start:chip_col_end]
    
    return shifted_chip
    
# Set Train and Guard paramaters

train_band = 10
guard_band = 50
pfa        = 1e-2


src_dir  = f'/media/joe/WD_12TB_HD1/test_data/'
dest_dir = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/slc_chip_data/'
slc_dir  = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/slc_burst_data/'
slc_NonShip_dir  = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Public/slc_burst_data/'
csv_dir  = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/Reports/'
csv_file = f'Train_RAW_IW2_ships_gt100m_Blaise_annotation_wSLC_ID_27Aug2025.csv'
df       = pd.read_csv(f'{csv_dir}{csv_file}')

chip_list =sorted(os.listdir(src_dir))
chip_size =[512,512]

i=0

#for file in chip_list[:10]:
for file in chip_list:
    # split filename and find matching slc burst
    file_info = parse_filename(file)
    
    #open slc burst
    
    #Check if non-ship chip
    if 'NonShip' in file:
        i = i + 1
        file_path =f'{slc_NonShip_dir}{file_info[2]}'
        if not os.path.isfile(file_path):
            print(f'File not found : {file_path}')
            continue
        slc_burst = np.load(file_path)
        #set offset to 0 and generate random ship_coord
        offset = [0, 0]
        #generate ship coords near centre of burst
        ship_coords=[int(np.random.rand()*400 + 300), 
                     int(np.random.rand()*10e3 + 5e3)]
        chip = abs(gen_shifted_chip(slc_burst, ship_coords, offset, chip_size))
        intensity_chip =np.square(chip)
        chip_name = '_'.join(file.split('_')[0:15]) + 'NoShip_chip#' + f'{str(i)}.npy'
        #print(f'{chip_name}')
        np.save(f'{dest_dir}{chip_name}', intensity_chip)
    else:
        file_path = f'{slc_dir}{file_info[2]}'
        if not os.path.isfile(file_path):
            print(f'File not found : {file_path}')
            continue
        slc_burst = np.load(file_path)
        offset = parse_offset_dir(file_info[0], file_info[1])
        ship_coords = get_ship_coordinates_df(df, file_info[3:5])
        chip = abs(gen_shifted_chip(slc_burst, ship_coords, offset, chip_size))
        intensity_chip =np.square(chip)
        chip_name = ('_'.join(file.split('_')[0:15]) + '_' +str(ship_coords[0])
                                                     + '_' +str(ship_coords[1])
                                                     + '_' + file_info[0] 
                                                     + '_' + str(file_info[1]) 
                                                     + '_chip' '.npy')
        #print(f'{chip_name}')
        np.save(f'{dest_dir}{chip_name}', intensity_chip)
    
        
    
 