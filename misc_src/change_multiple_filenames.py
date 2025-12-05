#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:02:37 2025

@author: joe
"""

import os

# Define the source directory (current directory in this case)
source_dir = './your_directory_path'  # Update this to your target directory path

def rename_files(directory):
    # List all files in the directory (without walking into subdirectories)
    for file_name in os.listdir(directory):
        # Check if the file contains '_n' and if it's a file (not a directory)
        if '_n' in file_name and os.path.isfile(os.path.join(directory, file_name)):
            # Create the new file name by replacing '_n' with '_chip#'
            new_name = file_name.replace('_n', '_chip#')

            # Construct full paths
            old_file_path = os.path.join(directory, file_name)
            new_file_path = os.path.join(directory, new_name)

            # Rename the file
            try:
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')
            except Exception as e:
                print(f"Error renaming file {file_name}: {e}")

# Run the rename function
rename_files(source_dir)
