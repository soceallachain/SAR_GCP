#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:55:48 2025

@author: joe
"""
import re
import os
import shutil
import multiprocessing



not_local=1
if not_local:
    label_dir  = 'mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels'
    # Define the source and destination directories
    source_directory_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/temp_ship_gt100m"
    destination_directory_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/temp_ship_gt100m_renamed"
    # Ensure the destination directory exists
    os.makedirs(destination_directory_path, exist_ok=True)

else:
    label_dir='/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/'
    source_directory_path = ""
    destination_directory_path = ""
    # Ensure the destination directory exists
    os.makedirs(destination_directory_path, exist_ok=True)


label_file = 'SARFish_RAW_Public_BurstChipInfo_Only_Sea_HighConfidenceShips_gt_100m.csv'
label_file_path = f'{label_dir}{label_file}'
if os.path.exists(label_file_path):
    df=pd.read_csv(label_file_path)
    chip_file_names =df['Chip_Name']
else:
    print(f'Label file not found: {label_file_path}')
    print('Check spelling and/or location')

# Strip everything after 'chip#xx' 
text_strings_real = [re.sub(r'(_chip#\d+).*\.npy$', r'\1', fname) for fname in chip_file_names]
text_strings_imag = text_strings_real.replace('_real_','_imag_')
text_strings_binary_mask = 

def copy_file(filename):
    # Check if any of the text strings are in the filename
    
    match_index = next((i for i, ts in enumerate(text_strings) if ts in filename), None)
    if match_index:
        # Get the full path to the original file and the destination file
        original_file = os.path.join(source_directory_path, filename)
        new_filename = chip_file_names[match_index]
        
        
        
        # Copy all the files : real, imag and binary_mask
        for s in ['_real_', '_imag_', '_binary_mask_']:
            destination_file = os.path.join(destination_directory_path, new_filename.replace('_real_',s))
            try:
                shutil.copy2(original_file, destination_file)
                print(f"Copied: {filename} to {destination_directory_path}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")

def main():
    # Get the list of files in the source directory
    filenames = os.listdir(source_directory_path)

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use the pool to map the copy_file function to each filename
        pool.map(copy_file, filenames)

    print("Copying complete.")

if __name__ == "__main__":
    main()
