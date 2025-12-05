"""
Script to copy RAW I and Q files, binary_mask files and 
slc files to a directory.



"""
import os
import re
import sys
import shutil
# Settings
image_dir = "/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_gt100m_chips/images"

src_dir  = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_gt100m_chips'
dest_dir = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_gt100m_chips/cleaned'
file_list      = 'image_files_to_keep_list.txt'
file_list_path = f'{image_dir}/{file_list}'
with open(file_list_path, 'r') as file:
    image_files = [line.strip() for line in file]

real_files               = [os.path.splitext(f)[0] + '.npy' for f in image_files]
imag_files               = [f.replace('_real_', '_imag_') for f in real_files]
bin_mask_files           = [f.replace('_real_', '_binary_mask_') for f in real_files]
slc_files                = [f.replace('_real_', '_slc_') for f in real_files]
normed_real_files        = [os.path.splitext(f)[0] + '_N.npy' for f in real_files]
normed_imag_files        = [os.path.splitext(f)[0] + '_N.npy' for f in imag_files]
normed_clahed_real_files = [os.path.splitext(f)[0] + '_NC.npy' for f in real_files]
normed_clahed_imag_files = [os.path.splitext(f)[0] + '_NC.npy' for f in imag_files]

#Make the directory for the cleaned data if it does not exist
os.makedirs(dest_dir,exist_ok=True)

#Copy real files to cleaned raw directory
#Ensure raw directory exists
os.makedirs(f'{dest_dir}/raw', exist_ok=True)
for f in real_files:
    shutil.copy(f'{src_dir}/raw/{f}',f'{dest_dir}/raw/{f}')
print(f'Raw real files copied from {src_dir}/raw to {dest_dir}/raw')    

for f in imag_files:
    shutil.copy(f'{src_dir}/raw/{f}',f'{dest_dir}/raw/{f}')
print(f'Raw imag files copied from {src_dir}/raw to {dest_dir}/raw')    

for f in bin_mask_files:
    shutil.copy(f'{src_dir}/raw/{f}',f'{dest_dir}/raw/{f}')
print(f'Binary mask files copied from {src_dir}/raw to {dest_dir}/raw')    

#Ensure slc directory exists and copy slc files to it
os.makedirs(f'{dest_dir}/slc', exist_ok=True)
for ff in slc_files:
    f = re.sub(r'_#HShips\d+.*\.npy$', '.npy', ff)
    shutil.copy(f'{src_dir}/slc/{f}',f'{dest_dir}/slc/{ff}')
print(f'Slc files copied from {src_dir}/slc to {dest_dir}/slc')    

#Ensure normed_raw directory exists and copy slc files to it
os.makedirs(f'{dest_dir}/raw_normed', exist_ok=True)
for f in normed_real_files:
    shutil.copy(f'{src_dir}/raw_normed/{f}',f'{dest_dir}/raw_normed/{f}')
for f in normed_imag_files:
    shutil.copy(f'{src_dir}/raw_normed/{f}',f'{dest_dir}/raw_normed/{f}')

#Ensure clahed_normed_raw directory exists and copy slc files to it
os.makedirs(f'{dest_dir}/raw_normed_clahed', exist_ok=True)
for f in normed_clahed_real_files:
    shutil.copy(f'{src_dir}/raw_normed_clahed/{f}',f'{dest_dir}/raw_normed_clahed/{f}')
for f in normed_clahed_imag_files:
    shutil.copy(f'{src_dir}/raw_normed_clahed/{f}',f'{dest_dir}/raw_normed_clahed/{f}')
print(f'Normed_Clahed files copied from {src_dir}/raw_normed_clahed to {dest_dir}/raw_normed_clahed') 



print(f'File transfer complete .......')
