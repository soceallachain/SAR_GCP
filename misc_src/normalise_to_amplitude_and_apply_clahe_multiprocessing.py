#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:13:10 2025
Script to:1. Read all non-binary_mask chips files in source diurectory
             Open each file
             Determne max and min in each file
             Normalise the data
             Save the normlaised data to a file with the same root name but with a '_N' appended prior to the .py extentsion
             Removes the original file
          2. Applies CLAHE to normalised data
             Saves clahed files 

@author: joe
"""
import numpy as np
import os 
import sys
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


import os
import numpy as np

def normalise_to_amplitude(I_file, dest_dir):
    try:
        if not os.path.exists(I_file):
            print(f'I-component File not found: {I_file}')
            return f"Missing I: {I_file}"

        I = np.load(I_file)
        Q_file = I_file.replace('_real_', '_imag_')

        if not os.path.exists(Q_file):
            print(f'Q-component File not found: {Q_file}')
            return f"Missing Q: {Q_file}"

        Q = np.load(Q_file)
        A = np.sqrt(I * I + Q * Q)
        max_A = A.max()
        if max_A == 0:
            print(f"Warning: max amplitude is zero in {I_file} and {Q_file}")
            return f"Zero amplitude: {I_file}"

        I_normed = I / max_A
        Q_normed = Q / max_A

        scale_factor = np.abs(I_normed).max()
        if scale_factor == 0:
            print(f"Warning: scale factor is zero in {I_file}")
            return f"Zero scale: {I_file}"

        I_normed = I_normed / scale_factor
        Q_normed = Q_normed / scale_factor

        I_root = os.path.basename(I_file).split('.')[0]
        Q_root = os.path.basename(Q_file).split('.')[0]
        I_normed_filename = os.path.join(dest_dir, f'{I_root}_N.npy')
        Q_normed_filename = os.path.join(dest_dir, f'{Q_root}_N.npy')

        # Ensure destination directory exists
        os.makedirs(dest_dir, exist_ok=True)
        np.save(I_normed_filename, I_normed)
        np.save(Q_normed_filename, Q_normed)

        # Explicitly delete large arrays to free memory
        del I, Q, A, I_normed, Q_normed

        return f"Success: {I_file}"

    except Exception as e:
        print(f"Error processing {I_file}: {e}")
        return f"Error: {I_file} ({e})"



def normalise_to_amplitude_old(I_file, dest_dir):
    if not os.path.exists(I_file):
        print(f'I-component File not found: {I_file}')

    I        = np.load(I_file)
    Q_file   = I_file.replace('_real_','_imag_')

    if not os.path.exists(Q_file):
        print(f'Q-component File not found: {Q_file}')
    Q        = np.load(Q_file)
    A        = np.sqrt(I*I + Q*Q)
    max_A    = A.max()
    I_normed = I/max_A
    Q_normed = Q/max_A
    scale_factor = np.abs(I_normed).max()
    I_normed = I_normed/scale_factor
    Q_normed = Q_normed/scale_factor
    #I_root   = I_file.split('.')[0]
    #Q_root   = Q_file.split('.')[0]
    I_root   = os.path.basename(I_file.split('.')[0])
    Q_root   = os.path.basename(Q_file.split('.')[0])
    I_normed_filename = f'{dest_dir}/{I_root}_N.npy'
    Q_normed_filename = f'{dest_dir}/{Q_root}_N.npy'
    np.save(I_normed_filename, I_normed)
    np.save(Q_normed_filename, Q_normed)
    return I_normed, Q_normed


from datetime import datetime

#Start timer
start_time =datetime.now()

source_dir = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw'

destination_dir = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw_normed'

os.makedirs(destination_dir, exist_ok=True)


# Get all .npy files in the source directory except the binary_mask files
files = [f for f in glob.glob(os.path.join(source_dir, "*.npy")) if '_imag_' not in os.path.basename(f) and '_binary_mask_' not in os.path.basename(f) and '_N' not in os.path.basename(f)]
#print('Files are:')
#print('\n'.join(files))
# Use multiprocessing to speed up file renaming
if __name__ == "__main__":
    num_processes = min(4, os.cpu_count())  # Default to 4 or less
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(normalise_to_amplitude, zip(files, [destination_dir] * len(files)))

    print("Chip normalization completed!")





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

"""
def apply_clahe(file, clip_limit=2.0, tile_grid_size=(8,8)):
    data = np.load(file)
    data_8bit = ((data +1)/2 *255).astype(np.uint8)
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    clahe_data = clahe.apply(data_8bit)
    #np.save(clahe_fname, clahe_data)
    return clahe_data
    
def get_clahe(file, destination_dir):
    data=apply_clahe(file)
    fname_base = os.path.basename(file.split('.')[0])
    clahe_name = f'{destination_dir}{fname_base}C.npy'
    np.save(clahe_name, data)
    return data

"""


source_dir = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw_normed'

dest_dir = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw_normed_clahed'

os.makedirs(dest_dir, exist_ok=True)

# Get all .npy files in the source directory except the binary_mask files
files = [f for f in glob.glob(os.path.join(source_dir, "*.npy")) if '_binary_mask_' not in os.path.basename(f) and '_N' in os.path.basename(f)]

# Use multiprocessing to speed up file renaming

if __name__ == "__main__":
    num_processes = min(4, os.cpu_count())  # Default to 4 or less
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(get_clahe, zip(files, [dest_dir] * len(files)))

    print("Chip normalization and CLAHE ompleted!")






#Stop timer
end_time = datetime.now()
delta = end_time - start_time

# Extract days
days = delta.days

# Extract hours, minutes, seconds from the remainder
hours, remainder = divmod(delta.seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f'******************* Completed at  {end_time} ************')
print(f'Execution time was {days}days:{hours}hours:{minutes}:{seconds}seconds')


