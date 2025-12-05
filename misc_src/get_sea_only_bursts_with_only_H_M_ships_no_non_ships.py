#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 17:40:16 2025

@author: joe
"""
import sys
import os
import re
import pandas as pd

# 1. Read each file and store the filenames in a list
def read_filenames(filepath):
    with open(filepath, 'r') as f:
        # Remove whitespace and skip empty lines
        return [line.strip() for line in f if line.strip()]
def remove_ships_suffix(filename):
    """
    Removes the '_#Ships*.npy' suffix from the filename, where * is any integer.
    """
    return re.sub(r'_(#Ships|#NonShips)\d+\.npy$', '', filename)

dataset='Public'
iw_mode = 2
min_ship_sizes = [0, 25, 50, 100]

#path_ = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/test_results/test_get_bursts_for_HM_sea_only_ships'
path = '/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_get_bursts_for_HM_sea_only_ships'
#Merge ship and non-ship dataframes

ship_df_filename= f'{path}/SARFish_RAW_Public_BurstInfoShipsPerBurstWithShipLocations.csv'
non_ship_df_filename =f'{path}/SARFish_RAW_Public_BurstInfoShipsPerBurstWithNonShipLocations.csv'
all_ships_df_filename =f'{path}/RAW_{dataset}_IW{iw_mode}_AllBurstsShipAndNonShipInfo.csv'


if os.path.exists(ship_df_filename):
    ship_df = pd.read_csv(ship_df_filename)
else:
    print(f'Error: File {ship_df_filename} not found! Exiting .......')
    sys.exit(0)

if os.path.exists(non_ship_df_filename):
    non_ship_df = pd.read_csv(non_ship_df_filename)
else:
    print(f'Error: File {non_ship_df_filename} not found! Exiting .......')
    sys.exit(0)


merged_df = pd.concat([ship_df, non_ship_df], axis=0, ignore_index=True)
first_col = merged_df.columns[0]
sorted_df = merged_df.sort_values(by=first_col).reset_index(drop=True)
sorted_df.to_csv(f'{path}/RAW_{dataset}_IW{iw_mode}_AllBurstShipAndNonShipInfo.csv')


for min_ship_size in min_ship_sizes:

    #Open file containing list of sea-based bursts containing H+M ships
    sea_based_burst_with_H_M_ships = f'RAW_{dataset}_IW{iw_mode}_sea_only_bursts_high_or_med_conf_ship_gt{min_ship_size}m.txt'

    #Open file containing list of sea-based bursts containing H+M non-ships
    # Want all non-ships so set min size to 0m
    sea_based_burst_with_non_ships = f'RAW_{dataset}_IW{iw_mode}_sea_only_bursts_high_or_med_conf_non_ship_gt0m.txt'
    #Save sea only bursts with ships and no non-ships to text file
    sea_based_bursts_with_ships_no_non_ships =f'RAW_{dataset}_IW{iw_mode}_sea_only_bursts_with_H_M_ships_gt{min_ship_size}m_no_non_ships.txt'

    # Replace these with your actual file paths
    file1_path = f'{path}/{sea_based_burst_with_H_M_ships}'
    file2_path = f'{path}/{sea_based_burst_with_non_ships}'
    file3_path=f'{path}/{sea_based_bursts_with_ships_no_non_ships}'

    list1 = read_filenames(file1_path)
    list2 = read_filenames(file2_path)
    # Remove '_#Ships*.npy' suffix
    list1 = [remove_ships_suffix(f) for f in list1]
    list2 = [remove_ships_suffix(f) for f in list2]

    # 2. Sort each list and remove duplicates
    list1 = sorted(set(list1))
    list2 = sorted(set(list2))

    list1 = [os.path.basename(path) for path in list1]
    list2 = [os.path.basename(path) for path in list2]

    debug =0
    if debug ==1:
        print('**************** Sea-based burst with H & M ships ***************')
        for f in list1:
            print(f)
        print('*************** Sea-based bursts with non-vessels ***************')
        for f2 in list2:
            print(f2)
        print('*********************************************')

    # 3. Convert each list to a set
    set1 = set(list1)
    set2 = set(list2)

    # 4. Subtract the set of filenames in File 2 from File 1
    diff_set = set1 - set2
    common_set = set1 & set2
    # 5. Remove the path (keep only the filename)

    # Optional: Sort the result
    filenames_only = sorted(list(diff_set))

    # Print the result
    with open(file3_path, 'w') as f:
        for filename in filenames_only:
            print(filename)
            f.write(f'{filename})\n')
    print(f'Number of {dataset} IW{iw_mode} sea-based bursts with ships larger than {min_ship_size}m and no non-ships is: {len(diff_set)}')
    print(f'Number of {dataset} IW{iw_mode} sea-based bursts with ships and non- ships is: {len(common_set)}')


print(f'Number of {dataset} IW{iw_mode} sea-based bursts with non- ships is: {len(list2)}')       


print('*************** get_sea_only_bursts_with_only_H_M_ships_no_non_ships.py Complete! ***************')                                                                                                                                                              




