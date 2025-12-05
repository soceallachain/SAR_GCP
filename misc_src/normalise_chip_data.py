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


source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming"   # Change to your source directory
#destination_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Test3/Sea"
destination_dir = source_dir
# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)


# Get all .npy files in the source directory except the binary_mask files
files = [f for f in glob.glob(os.path.join(source_dir, "*.npy")) if '_binary_mask_' not in os.path.basename(f) and '_N' not in os.path.basename(f)]

# Use multiprocessing to speed up file renaming
if __name__ == "__main__":
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(normalise, files)

    print("Chip normalization completed!")
