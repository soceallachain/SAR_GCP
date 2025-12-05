"""
Script to copy RAW I and Q files, binary_mask files and 
slc files to a directory.



"""
import os
import re
import sys
import shutil
# Settings

src_dir   = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/raw_public_burst_chip_data'
dest_dir  = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_m_and_hc_sea_ships_no_nonships_gt100m_chips/raw'
#src_dir   = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/raw_public_burst_chip_data'
#dest_dir  = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_m_and_hc_sea_ships_no_nonships_gt100m_chips/raw'
file_list = 'RAW_high_and_med_combined_conf_ship_no_nonships_only_sea_gt100.txt'
file_list_path = f'/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/{file_list}'
with open(file_list_path, 'r') as file:
    files = [line.strip() for line in file]


#Copy real files to cleaned raw directory
#Ensure raw directory exists
os.makedirs(f'{dest_dir}', exist_ok=True)
for f in files:
    shutil.copy(f'{src_dir}/{f}',f'{dest_dir}/{f}')
print(f'Raw real files copied from {src_dir}/raw to {dest_dir}/raw')    
