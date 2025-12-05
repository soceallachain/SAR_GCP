#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:19:37 2025
Script to test CFAR algorithm

Opeerates on cleaned/curated set of SLC files that are all sea-based and 
contain only labelled ships. Chips with non-labelled vessels or features 
have been removed .

@author: joe
"""
from  SAR_code_module import *


def plot_hist(data, number_bins, xlabel='Values', ylabel = 'Frequency',
              title='Histogram', min_threshold=None, max_threshold=None):
    
    if min_threshold is None:
        min_threshold = data.min()
    if max_threshold is None:
        max_threshold = data.max()
    plt.figure(figsize=(8, 6))
    plt.hist(data, number_bins, range =(min_threshold, max_threshold),edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
    
def extract_ship_numbers(filename):
    parts = filename.split('#Ships')
    if len(parts) > 1:
        number_part = parts[1].split('_Sea')[0]
        return number_part
    return None
    

# Set CFAR paramters
num_guard = 5
num_train = 10
pfa = 1e-3

kernel1 = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])
kernel2 = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])


slc_src_dir = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_hc_slc_cleaned_chips'
binary_mask_src_dir = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_high_confidence_sea_chips_cleaned_chips'
image_dir ='/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/chip_images/CFAR_gamma_hc_sea_images_cleaned_g60_t180_pfa_1e_3'
# Check direxctory exists and not empty
if os.path.isdir(slc_src_dir) and len(os.listdir(slc_src_dir)):
    files = sorted(os.listdir(slc_src_dir))




a=[] 
for f in [files[25]]:   
#for f in files:
    slc_chip_filename = f'{slc_src_dir}/{f}'
    
   
    binary_mask_filename = f.replace('_slc_', '_binary_mask_')
    #binary_mask_filename = binary_mask_filename.replace('_SLC_', '_SLC__')
    binary_mask_filename = binary_mask_filename.replace('.npy', '_')    
    binary_mask_filename =glob.glob(f'{binary_mask_src_dir}/{binary_mask_filename}*.npy')[0]
    number_of_ships = extract_ship_numbers(binary_mask_filename)
    
    if os.path.exists(slc_chip_filename) and os.path.exists(binary_mask_filename):
        try:
            slc = np.load(slc_chip_filename)
            binary_mask = np.load(binary_mask_filename)
            a.append('success')
            print(f'Processing {slc_chip_filename}')
        except IOError:
            print(f'Error opening {slc_chip_filename} or {binary_mask_filename}')
    else:
        print(f'Error: {slc_chip_filename} or {binary_mask_filename} not found. Continuing to next file....')
        continue
    
            
    CFAR_slc = cfar_detector(slc,num_train, num_guard, pfa)
    detection_result_er2 = erosion(CFAR_slc, kernel1)
    #detection_result_er2 = erosion(detection_result_er2, kernel1)
    detection_result_er2 = dilation(detection_result_er2, kernel1)
    #detection_result_er2 = dilation(detection_result_er2, kernel1)
    
    
    chip_info    = extract_product_burst_and_chip_information(slc_chip_filename)
    SLC_ID       = chip_info[0]
    burst_number = chip_info[1]
    chip_number  = chip_info[2]
    
    fig, ax = plt.subplots(3,1)
    ax[0].imshow(slc, norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
    ax[1].imshow(binary_mask,         cmap='binary_r', aspect = 'auto')
    ax[2].imshow(detection_result_er2,cmap='binary_r', aspect = 'auto') 
    ax[0].set_title(f'SLC for   Burst #{burst_number} Chip#{chip_number}',fontsize=6)
    ax[1].set_title(f'Binary Mask for Burst #{burst_number} Chip#{chip_number}', fontsize=6)
    ax[2].set_title(f'CFAR  for  Burst #{burst_number} Chip#{chip_number}', fontsize=6) 
    for axx in ax:  
        axx.tick_params(axis='both', labelsize=5)  # Set tick label size
    fig.suptitle(f'{SLC_ID} : #Ships={number_of_ships} Pfa = {pfa}', y=0.95, fontsize=7)
    f_name = f'{image_dir}/CFAR_{SLC_ID}_burst#{burst_number}_chip#{chip_number}_g{num_guard}_t{num_train}_pfa{pfa}.jpg'
    plt.savefig(f_name, format='jpeg', dpi=300)
    plt.close(fig) 

print(f'##############################   Completed {len(a)} files ################################################')
#plot_hist(slc.flatten(), 100)   
#test_slc_chip='/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_hc_sea_chips/public_hc_slc/S1A_IW_SLC_1SDV_20200426T051114_20200426T051141_032294_03BC92_D7BE_RAW_IW2_slc_burst#6_chip#6.npy'    
#test_slc=np.load(test_slc_chip)
