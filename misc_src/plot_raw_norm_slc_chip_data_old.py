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
"""
from  SAR_code_module import *

raw_burst_dir    = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/raw_public_burst_data'
slc_source_dir    = '/mnt/smbshare/RAWSARAI_Data/Copernicus/'
chip_source_dir               = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_high_confidence_sea_chips'
slc_chip_source_dir           = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_hc_slc'
normed_chip_source_dir        = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_high_confidence_sea_chips_normed'
normed_clahed_chip_source_dir = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_high_confidence_sea_chips_normed_clahed'

image_dir = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/chip_images'

files = [f for f in glob.glob(os.path.join(chip_source_dir, "*.npy")) if '_real_' in os.path.basename(f)]


SLC_ID_CRC = "EB4B"
burst_number = 3
chip_number  = 5
indices = [i for i, s in enumerate(files) if (SLC_ID_CRC in s and f'burst#{burst_number}' in s and f'chip#{chip_number}' in s)]
if indices:
    f = files[indices[0]]
else:
    print(f' SLC_ID CRC of {SLC_ID_CRC} with burst#{burst_number} and'
          f' chip#{chip_number} not found in Public High Confidence Ship Dataset.'
          f' Check if values correct! Exiting ........')
    #exit(0)
        
for f in files:
#for i in range(0,10):
#    f=files[i]
    
    chip_filename        = os.path.basename(f)
    normed_chip_filename = chip_filename.replace('.npy','_N.npy')
    normed_clahed_chip_filename = chip_filename.replace('.npy','_NC.npy')
    binary_mask_filename = f.replace('_real_', '_binary_mask_')
    file_parts = extract_product_burst_and_chip_information(f)    
    slc_chip_filename = f'{file_parts[0]}_RAW_IW2_slc_burst#{file_parts[1]}_chip#{file_parts[2]}.npy' 
    slc_chip_filename = slc_chip_filename.replace('SLC__', 'SLC_')
    chip_data = np.load(f)
    binary_mask_data = np.load(binary_mask_filename)
    normed_chip_data = np.load(os.path.join(normed_chip_source_dir, normed_chip_filename))
    normed_clahed_chip_data = np.load(os.path.join(normed_clahed_chip_source_dir, 
                                                   normed_clahed_chip_filename)
                                      )
    slc_chip_data = np.load(os.path.join(slc_chip_source_dir,
                                         slc_chip_filename)
                            )

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
    
    
    fig1.suptitle(f'{os.path.basename(f)}', y=0.95, fontsize=8)
    f_name = f'{image_dir}/{file_parts[0]}_burst#{file_parts[1]}_chip#{file_parts[2]}.jpg'
    plt.savefig(f_name, format='jpeg', dpi=300)
    plt.close(fig1) 
    #plt.show()
        
   
    
SLC_ID = file_parts[0]
burst_number = file_parts[1]  
slc_path = 
a,b,c = get_slc_burst_with_ships(SLC_ID, slc_path, burst_number)
burst = a
burst_with_tagged_ships = b
burst_binary = c

plot_output_of_get_slc_burst_data(SLC_ID, burst_number,
                                  burst, 
                                  burst_with_tagged_ships, 
                                  burst_binary)
    

plot_IW2_RAW_SLC(SLC_Product_ID, burst_num_to_plot)
