#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:53:31 2025

@author: joe
"""


from  SAR_code_module import *




swath_of_interest = 2
polarization      ='vh'
sar_mode = 'RAW'
swath_of_interest_index = swath_of_interest - 1
IW_mode = swath_of_interest_index
freq_multipliers=[1, 54.595959/64.345238, 46.915402/64.345238]


burst_number = 2
chip_size=[512, 3072]
chip_number = 20

if sar_mode == 'RAW':
   sampling_freq_mul=freq_multipliers[IW_mode]
else:
   sampling_freq_mul = 1

freq_scale_factor = sampling_freq_mul

image_dir =""

#SLC_ID = 'S1A_IW_SLC__1SDV_20200914T075756_20200914T075825_034352_03FE84_B317'
SLC_ID = 'S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B'
#SLC_ID='S1A_IW_SLC__1SDV_20200421T051411_20200421T051438_032221_03BA16_C5BE'
#SLC_ID='S1B_IW_SLC__1SDV_20200803T075720_20200803T075748_022756_02B2FF_E5D2'

burst_dir ="/"
chip_dir  ="/'"


burst_filename = f'{burst_dir}/{SLC_ID}_RAW_IW{swath_of_interest}_real_burst#{burst_number}.npy'
chip_filename  = f'{burst_dir}/{SLC_ID}_RAW_IW{swath_of_interest}_real_burst#{burst_number}_chip#{chip_number}.npy'
raw_burst_data = np.load(burst_filename)
raw_burst      =np.real(raw_burst_data)
# Get real component

raw_chip_data  = np.load(chip_filename)
raw_chip  = np.real(raw_burst_data)
normed_raw_chip = normalise_data(raw_chip,1,-1)

slc_burst_types       = get_slc_chip_and_bursts(SLC_ID, burst_number,
                                                chip_number, freq_scale_factor)
slc_chip              = slc_burst_types[0]
slc_burst             = slc_burst_types[1]
slc_burst_with_ships  = slc_burst_types[2]
slc_burst_binary_mask = slc_burst_types[3]
slc_binary_mask_chip  = get_chip_data(slc_burst_binary_mask,chip_number, chip_size)


#plot raw_chip, raw_normed_chip, slc_chip, slc_binary_mask_chip
fig1, ax1 = plt.subplots(4,1)
ax1[0].imshow(raw_chip, vmin=raw_chip.min(),vmax = raw_chip.max(), aspect = 'auto')
ax1[1].imshow(normed_raw_chip, vmin=normed_raw_chip.min(),vmax = normed_raw_chip.max(), aspect = 'auto')
ax1[2].imshow(slc_binary_mask_chip, vmin=slc_binary_mask_chip.min(),vmax = slc_binary_mask_chip.max(), cmap='binary_r', aspect = 'auto')
ax1[3].imshow(np.abs(slc_chip),  norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
 
ax1[0].set_title(f'Chip - No normalization', fontsize=6)
ax1[1].set_title(f'Chip - Normalized', fontsize=6)  
ax1[2].set_title(f'Binary Mask Chip',fontsize=6)
ax1[3].set_title(f'SLC Chip',fontsize=6)
for ax in ax1:  
    ax.tick_params(axis='both', labelsize=6)  # Set tick label size
 

fname=f'{SLC_ID}_RAW_IW2_burst#{burst_number}_chip#{chip_number}'
fig1.suptitle(f'{fname}', y=0.95, fontsize=8)
#f_name = f'{image_dir}/{file_parts[0]}_burst#{file_parts[1]}_chip#{file_parts[2]}.jpg'
#plt.savefig(f_name, format='jpeg', dpi=300)
#plt.close(fig1) 
plt.show()
     

#plot raw burst with chip
fig2, ax2 = plt.subplots(2,1)
ax2[0].imshow(np.real(raw_burst), vmin=np.real(raw_burst).min(),vmax = np.real(raw_chip).max(), aspect = 'auto')
ax2[1].imshow(np.abs(slc_burst_with_ships), norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
ax2[0].set_title(f'RAW Burst#{burst_number} with Chip#{chip_number}', fontsize=6)
ax2[1].set_title(f'SLC Burst#{burst_number} with Chip#{chip_number}', fontsize=6)  
fig2.suptitle(f'{fname}', y=0.95, fontsize=8)
#plot slc_burst_with_ships and slc_chip
draw_chip(slc_burst_with_ships, chip_number, chip_size, ax2[1])
draw_chip(slc_burst_with_ships, chip_number, chip_size, ax2[0])