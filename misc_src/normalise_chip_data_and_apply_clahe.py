#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:13:10 2025

@author: joe
"""
import numpy as np
import os 
import glob
import multiprocessing
import matplotlib.pyplot as plt


def normalise(file, range_max=1, range_min=-1):
    data = np.load(file)
    max  = data.max()
    min  = data.min()
    normed_data = ((data - min) / (max - min)) * (range_max - range_min) + range_min
    root        = file.split('.')[0]
    #dir_path    = os.path.dirname(file)
    new_filename = f'{root}_N.npy'
    np.save(new_filename, normed_data)
    os.remove(file)
    return normed_data


def normalise_to_amplitude(I_file):
    I        = np.load(I_file)
    Q_file   = I_file.replace('_real_','_imag_')
    Q        = np.load(Q_file)
    A        = np.sqrt(I*I + Q*Q)
    max_A    = A.max()
    I_normed = I/max_A
    Q_normed = Q/max_A
    scale_factor = np.abs(I_normed).max()
    I_normed = I_normed/scale_factor
    Q_normed = Q_normed/scale_factor
    I_root   = I_file.split('.')[0]
    Q_root   = Q_file.split('.')[0]
    I_normed_filename = f'{I_root}_N.npy'
    Q_normed_filename = f'{Q_root}_N.npy'
    np.save(I_normed_filename, I_normed)
    np.save(Q_normed_filename, Q_normed)
    os.remove(I_file)
    os.remove(Q_file)
    return I_normed, Q_normed




# Place files to normalise in normalized directory - will removed them after normalization
source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_high_confidence_sea_chips_cleaned_chips_normed"   # Change to your source directory
#source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming"   # Change to your source directory
#destination_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Test3/Sea"
destination_dir = source_dir
# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)


# Get all 'real' .npy  files in the source directory
files = [f for f in glob.glob(os.path.join(source_dir, "*.npy")) 
         if '_real_' in os.path.basename(f)
         and '_N' not in os.path.basename(f)
        ]

# Use multiprocessing to speed up file renaming
if __name__ == "__main__":
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(normalise_to_amplitude, files)

    print("Chip normalization completed!")
    print('Processing CLAHE data ....')


import cv2

def apply_clahe(file, clip_limit=2.0, tile_grid_size=(8,8)):
    data = np.load(file)
    data_8bit = ((data +1)/2 *255).astype(np.uint8)
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    clahe_data = clahe.apply(data_8bit)
    #np.save(clahe_fname, clahe_data)
    return clahe_data

def apply_clahe_to_data(data, clip_limit=2.0, tile_grid_size=(8,8)):
    data_8bit = ((data +1)/2 *255).astype(np.uint8)
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    clahe_data = clahe.apply(data_8bit)
    #np.save(clahe_fname, clahe_data)
    return clahe_data
    
def get_clahe(file, destination_dir):
    data=apply_clahe(file)
    fname_base = os.path.basename(file.split('.')[0])
    clahe_name = f'{destination_dir}/{fname_base}C.npy'
    np.save(clahe_name, data)
    return data

    

source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_high_confidence_sea_chips_cleaned_chips_normed"   # Change to your source directory
dest_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_high_confidence_sea_chips_cleaned_chips_normed_clahed"   # Change to your destination directory
#source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming" 
#dest_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming/normed_clahed/" 
os.makedirs(dest_dir, exist_ok=True)

# Get all .npy files in the source directory except the binary_mask files
files = [f for f in glob.glob(os.path.join(source_dir, "*.npy")) if '_binary_mask_' not in os.path.basename(f) and '_N' in os.path.basename(f)]

# Use multiprocessing to speed up file renaming

if __name__ == "__main__":
    num_processes = os.cpu_count() or 4  # Default to 4 if None
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(get_clahe, zip(files, [dest_dir] * len(files)))


print("Chip normalization and CLAHE completed!")
print('Processing CLAHE data ....')
