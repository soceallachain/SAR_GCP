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


def normalise(file, range_max, range_min):
    data = np.load(file)
    max  = data.max()
    min  = data.min()
    normed_data = ((data - min) / (max - min)) * (range_max - range_min) + range_min
    root        = file.split('.')[0]
    #dir_path    = os.path.dirname(file)
    new_filename = f'{root}_N.npy'
    np.save(new_filename, normed_data)
    #os.remove(file)
    return normed_data


source_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming"   # Change to your source directory
#destination_dir = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/Test3/Sea"
destination_dir = source_dir
# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)


# Get all .npy files in the source directory except the binary_mask files
#files = glob.glob(os.path.join(source_dir, "*.npy"))
files = [f for f in glob.glob(os.path.join(source_dir, "*.npy")) if '_binary_mask_' not in os.path.basename(f) and '_N' not in os.path.basename(f)]

filename=files[2]

#plot chip before normalization

chip_data = np.load(filename)
normed_chip_data = normalise(filename,1,-1)
if '_real_' in filename:
    bin_fname =filename.replace('_real_','_binary_mask_')
else:
    bin_fname =filename.replace('_imag_','_binary_mask_')
bin_chip_data = np.load(bin_fname)

fig1, ax1 = plt.subplots(3,1)
ax1[0].imshow(chip_data, vmin=chip_data.min(),vmax = chip_data.max(), aspect = 'auto')
ax1[1].imshow(normed_chip_data, vmin=normed_chip_data.min(),vmax = normed_chip_data.max(), aspect = 'auto')
ax1[2].imshow(bin_chip_data, vmin=bin_chip_data.min(),vmax = bin_chip_data.max(), aspect = 'auto')
ax1[0].set_title(f'Chip - No normalization')
ax1[1].set_title(f'Chip Normalized')
ax1[2].set_title(f'Binary Mask Chip')

fig1.suptitle(f'{os.path.basename(filename)}', y=0.95)
plt.show()






"""
# Use multiprocessing to speed up file renaming
if __name__ == "__main__":
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(normalise, files)

    print("File renaming completed!")
"""