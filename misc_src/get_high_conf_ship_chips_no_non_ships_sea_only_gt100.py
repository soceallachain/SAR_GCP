#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:49:03 2025
Script to read in list of '_real_' chips and format names and
append names of '_imag_' and '_binary_mask_' chip files, deletes 
duplicates and sorts
@author: joe
"""

import os
import re

def read_and_process_file(file_path):
    """Read a file containing absolute file paths, strip paths, and remove '_#Shipsyy' part."""
    processed = []
    pattern = re.compile(r'(.*_chip#\d+)_#Ships\d+(\.npy)$')
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            filename = os.path.basename(line)
            # Remove '_Shipsyy' part if present
            match = pattern.match(filename)
            if match:
                new_filename = match.group(1) + match.group(2)
            else:
                new_filename = filename  # If pattern not found, keep as is
            processed.append(new_filename)
    return processed

def main():
    src_dir = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels'
    filename1 = 'RAW_high_conf_ship_chips_sea_only_gt100.txt'  # Replace with your first filename list
    filename2 = 'RAW_non_vessel_sea_only_chip_list.txt'  # Replace with your second filename list
    output_filename = 'RAW_high_conf_ship_no_nonships_only_sea_gt100.txt'
    file1    = f'{src_dir}/{filename1}'
    file2    = f'{src_dir}/{filename2}'
    
    output_file = f'{src_dir}/{output_filename}'  # Output file for the filtered list

    # Step 1-4: Read, strip paths, remove _Shipsyy
    list1 = read_and_process_file(file1)
    list2 = read_and_process_file(file2)
    
    print(list1[:5])

    # Step 5: Remove from list1 any elements that are also in list2
    set2 = set(list2)
    filtered_list1 = [item for item in list1 if item not in set2]
    # Remove duplicates and sort 
    filtered_list1 = sorted(list(set(filtered_list1)))
    f_list2 = [item for item in list1 if item in set2]
    print(f'{len(f_list2)} common files being removed  are: {f_list2}')
    imag_list = [filename.replace('_real_', '_imag_') for filename in filtered_list1]
    mask_list = [filename.replace('_real_', '_binary_mask_') for filename in filtered_list1]
    filtered_list1.extend(imag_list)
    filtered_list1.extend(mask_list)

    # Save the result
    with open(output_file, 'w') as f:
        for item in filtered_list1:
            f.write(item + '\n')

    print(f"Filtered list saved to {output_file}")

if __name__ == '__main__':
    main()
