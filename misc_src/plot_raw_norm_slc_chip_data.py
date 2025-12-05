#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:58:37 2025

@author: joe

Script to loop through  'real' chips in a directory and find corresponding
binary mask chip and normed and normed_clahed chip in neighbouring directories
and plot these along with the slc image for the chip.
Then view all 4 plots and decide if chip data clean enough to be used for 
training purposes. 
If no binary mask set binary_mask to zeros but check if exists
If using sea only chip - no binary mask as no labelled ships
"""
from  SAR_code_module import *
#plt.ion()
plt.ioff()

# NOTE: Set to 1 if using sea only chips
no_binary_mask = 0
chip_source_dir               ='/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw' 
slc_chip_source_dir           ='/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/slc'
normed_chip_source_dir        ='/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw_normed'
normed_clahed_chip_source_dir ='/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw_normed_clahed'

image_dir                     = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/images'


files = [f for f in glob.glob(os.path.join(chip_source_dir, "*.npy")) if '_real_' in os.path.basename(f)]
with open(f'{image_dir}/images_completed.log', 'r') as f:
    completed_files_list = [line.strip() for line in f]

files_not_found=[]

for f in files[:5]:
#for i in range(0,10):
#    f=files[i]
    
    chip_filename             = os.path.basename(f)
    normed_chip_filename      = chip_filename.replace('.npy','_N.npy')
    normed_clahed_chip_filename = chip_filename.replace('.npy','_NC.npy')
    file_parts                = extract_product_burst_and_chip_information(f)    
    slc_chip_filename         = f'{file_parts[0]}_RAW_IW2_slc_burst#{file_parts[1]}_chip#{file_parts[2]}.npy' 
    #slc_chip_filename         = slc_chip_filename.replace('SLC__', 'SLC_')
    f_name = f"{chip_filename.replace('.npy','.jpg')}"
    if f_name in completed_files_list:
        print(f'File {f_name} already processed. Skipping ....')
        continue
    else:
        print(f'Now processing {slc_chip_filename}') 

    if os.path.isfile(os.path.join(normed_chip_source_dir, normed_chip_filename)):
        normed_chip_data = np.load(os.path.join(normed_chip_source_dir, normed_chip_filename))
    else:
        print(f'Error: Normed File {os.path.join(normed_chip_source_dir, normed_chip_filename)} not found. Skipping ...')
        files_not_found.append(os.path.join(normed_chip_source_dir, normed_chip_filename))
        continue


    if os.path.isfile(os.path.join(normed_clahed_chip_source_dir, normed_clahed_chip_filename)):
        normed_clahed_chip_data = np.load(os.path.join(normed_clahed_chip_source_dir, normed_clahed_chip_filename))
    else:
        print(f'Error: Normed-CLAHED File {os.path.join(normed_clahed_chip_source_dir, normed_clahed_chip_filename)} not found. Skipping ...')
        files_not_found.append(os.path.join(normed_clahed_chip_source_dir, normed_clahed_chip_filename))
        continue

    chip_data                 = np.load(f)
    normed_chip_data          = np.load(os.path.join(normed_chip_source_dir, normed_chip_filename))
    normed_clahed_chip_data   = np.load(os.path.join(normed_clahed_chip_source_dir, 
                                                   normed_clahed_chip_filename))

    if os.path.isfile(os.path.join(slc_chip_source_dir, slc_chip_filename)):
        slc_chip_data = np.load(os.path.join(slc_chip_source_dir, slc_chip_filename))
    else:
        print(f'Error: SLC File {os.path.join(slc_chip_source_dir, slc_chip_filename)} not found. Skipping ...')
        files_not_found.append(os.path.join(slc_chip_source_dir, slc_chip_filename))
        continue

    if not no_binary_mask:
        binary_mask_filename      = f.replace('_real_', '_binary_mask_')
        binary_mask_data          = np.load(binary_mask_filename)
    else:
        binary_mask_data          = np.zeros_like(chip_data)
    print(f'Processed {chip_filename} \n {slc_chip_filename} \n') 
    # Plot all 4 data on one figure
       
    fig1, ax1 = plt.subplots(4,1)
    ax1[0].imshow(chip_data, vmin=chip_data.min(),vmax = chip_data.max(), aspect = 'auto')
    #ax1[1].imshow(normed_chip_data, vmin=normed_chip_data.min(),vmax = normed_chip_data.max(), aspect = 'auto')
    ax1[1].imshow(normed_clahed_chip_data, vmin=normed_clahed_chip_data.min(),vmax = normed_clahed_chip_data.max(), aspect = 'auto')
    ax1[2].imshow(binary_mask_data, vmin=binary_mask_data.min(),vmax = binary_mask_data.max(), cmap='binary_r', aspect = 'auto')
    ax1[3].imshow(slc_chip_data,  norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
    
    ax1[0].set_title(f'Chip - No normalization', fontsize=6)
    #ax1[1].set_title(f'Chip Normalized')
    ax1[1].set_title(f'Chip - CLAHEd & Normalized', fontsize=6)  
    ax1[2].set_title(f'Binary Mask Chip',fontsize=6)
    ax1[3].set_title(f'SLC Chip',fontsize=6)
    for ax in ax1:  
        ax.tick_params(axis='both', labelsize=6)  # Set tick label size
    
    
    fig1.suptitle(f'{os.path.basename(f)}', y=0.95, fontsize=6)
    f_name = f"{image_dir}/{chip_filename.replace('.npy','.jpg')}"
    #f_name = f'{image_dir}/{file_parts[0]}_burst#{file_parts[1]}_chip#{file_parts[2]}.jpg'
    plt.savefig(f_name, format='jpeg', dpi=300)
    plt.close(fig1) 
    #plt.show()

print('**************************** Completed ****************************')
if len(files_not_found)>0:
    print(f' ............ but following files were not found : {files_not_found}')
