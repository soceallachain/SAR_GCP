#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:22:45 2025

@author: joe
"""
import pandas as pd
import numpy as np
import os
import re

local = 1
if local ==1:
    label_dir='/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Dump/'
    label_file='SARFish_Public_SeaBasedBursts.csv'
    label_file_path=f'{label_dir}{label_file}'
    save_path = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Dump/'
    chip_dir  = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/raw_public_burst_chip_data'
else:
    label_dir=''
    label_file='SARFish_Public_SeaBasedBursts.csv'
    label_file_path=f'{label_dir}{label_file}'
    save_path = ''
    chip_dir  = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/raw_public_burst_chip_data'
    
dataset='Public'
swath = 2
if not os.path.exists(label_file_path):
    print(f'Label file not found at : {label_file_path}')
    print(f'Check spelling or location')
else:
    df = pd.read_csv(label_file_path)
    sea_based_bursts_list = df["Burst_Name"].to_list()
    with open(f'{save_path}{dataset}_IW{swath}_sea_based_bursts_list.txt', 'w') as f:
        for item in sea_based_bursts_list:
            print(f'{save_path}{item}')
            f.write(f"{item}\n")


sea_only_ship_chip_list_filename = f'{dataset}_RAW_IW{swath}_is_vessel_sea_only_chip_list.txt'
sea_only_non_vessel_chip_list_filename = f'{dataset}_RAW_IW{swath}_non_vessel_sea_only_chip_list.txt'
sea_only_burst_list_filename = f'{dataset}_IW{swath}_sea_based_bursts_list.txt'
public_chip_list_filename    = f'{dataset}_bursts_chip_list.txt'

sea_only_burst_list_file = os.path.join(save_path, sea_only_burst_list_filename)
if not os.path.exists(sea_only_burst_list_file):
    print(f'Error: File {sea_only_burst_list_file} not found')

public_chip_list_file   = os.path.join(save_path, public_chip_list_filename)    
if not os.path.exists(public_chip_list_file):
    print(f'Error: File {public_chip_list_file} not found')
    
sea_only_ship_chip_list_file = os.path.join(save_path, sea_only_ship_chip_list_filename)
if not os.path.exists(sea_only_ship_chip_list_file):
    print(f'Error: File {public_chip_list_file} not found')

sea_only_non_vessel_chip_list_file = os.path.join(save_path, sea_only_non_vessel_chip_list_filename)
if not os.path.exists(sea_only_non_vessel_chip_list_file):
    print(f'Error: File {public_chip_list_file} not found')


with open(sea_only_burst_list_file,'r') as f:
    sea_only_burst_list = [burst.strip() for burst in f]    

with open(public_chip_list_file,'r') as f:
    public_chip_list = [chip.strip() for chip in f]

with open(sea_only_ship_chip_list_file,'r') as f:
    sea_only_ship_chip_list = [chip.strip() for chip in f]    
    
with open(sea_only_non_vessel_chip_list_file,'r') as f:
    sea_only_non_vessel_chip_list = [chip.strip() for chip in f]    


# Need to strip '_#Ships*' from end of each vessel file to allow comparison
# with empty chips to enable their exclusion
sea_only_ship_chip_list = [
    re.sub(r'_#Ships\d+', '', filename) for filename in sea_only_ship_chip_list
]

sea_only_non_vessel_chip_list = [
    re.sub(r'_#Ships\d+', '', filename) for filename in sea_only_non_vessel_chip_list
]

    
# Need to generate list of sea-only-chips for dataset
# Find chips that belong to sea-only-bursts in public chip list

import re

def extract_prefix(filename):
    """
    Extract substring from start up to and including '_burst#<number>'.
    Returns None if pattern not found.
    """
    match = re.search(r'(.*_burst#\d+)', filename)
    if match:
        return match.group(1)
    return None

def find_encompassing_filenames(list1, list2):
    # Extract prefixes from list2 filenames
    prefixes = [extract_prefix(f) for f in list2]
    prefixes = [p for p in prefixes if p is not None]

    matched_files = []

    for f1 in list1:
        for prefix in prefixes:
            if f1.startswith(prefix):
                matched_files.append(f1)
                break  # Stop checking prefixes once matched

    return matched_files

def find_sea_only_chips(all_chips, sea_bursts):
    return(find_encompassing_filenames(all_chips, sea_bursts))


# Search for chips that belong to sea based bursts
public_sea_only_chips = find_sea_only_chips(public_chip_list, sea_only_burst_list)

# Want sea only chips devoid of ships and non-vessels
# Form sets
# empty_sea_chips = sea_chips - (sea_chips_with_ships and sea_chips_with_non_vessels)
empty_sea_chips = list(set(public_sea_only_chips)
                       - (set(sea_only_ship_chip_list) 
                       | set(sea_only_non_vessel_chip_list)))

number_reduced_chips = 1000
if len(empty_sea_chips) > number_reduced_chips:
    reduced_empty_sea_chips=empty_sea_chips[0:number_reduced_chips]
else:
    print(f'WARNING: Number of available empty chips is less than {number_reduced_chips} requested!.')
    print(f'         Setting reduced number to same size as total, which is {len(empty_sea_chips)}.')
    reduced_empty_sea_chips=empty_sea_chips[0:len(empty_sea_chips)]
    
# So far have only identified _real_ chips. Need to add _imag_ and _binary_mask_
imag_sea_chips = [
    filename.replace('_real_','_imag_') for filename in empty_sea_chips
]

mask_sea_chips = [
    filename.replace('_real_','_binary_mask_') for filename in empty_sea_chips
]

empty_sea_chips.extend(imag_sea_chips)
empty_sea_chips.extend(mask_sea_chips)

#Save empty_sea_chips list to file 
with open(f'{save_path}{dataset}_IW{swath}_empty_sea_chips_list.txt', 'w') as f:
     for item in empty_sea_chips:
         print(f'{save_path}{item}')
         f.write(f"{item}\n")


r_imag_sea_chips = [
    filename.replace('_real_','_imag_') for filename in reduced_empty_sea_chips
]


r_mask_sea_chips = [
    filename.replace('_real_','_binary_mask_') for filename in reduced_empty_sea_chips
]

reduced_empty_sea_chips.extend(r_imag_sea_chips)
reduced_empty_sea_chips.extend(r_mask_sea_chips)

#Save empty_sea_chips list to file 
with open(f'{save_path}{dataset}_IW{swath}_empty_sea_chips_list_reduced.txt', 'w') as f:
     for item in reduced_empty_sea_chips:
         print(f'{save_path}{item}')
         f.write(f"{item}\n")





psoc=set(public_sea_only_chips)
sosc=set(sea_only_ship_chip_list)
sonvc=set(sea_only_non_vessel_chip_list)
pcl=set(public_chip_list)

# Copy empty sea chips to a directory and tar the directory
    