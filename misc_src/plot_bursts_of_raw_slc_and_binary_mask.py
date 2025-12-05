#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from SAR_code_module import *

plt.ion()

raw_burst_dir = '/home/joe/Work/Projects/RSAR/Test/SampleData/Public/raw_burst_data'
slc_burst_dir = '/home/joe/Work/Projects/RSAR/Test/SampleData/Public/slc_burst_data'
"""
raw_burst_dir = '/media/joe/WD_12TB_HD1/jc_work/Copernicus/Public/raw_burst_data'
slc_burst_dir = '/media/joe/WD_12TB_HD1/jc_work/Copernicus/Public/slc_burst_data'

raw_burst_file = 'S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794_RAW_IW2_real_burst#1.npy'
slc_burst_file = 'S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794_RAW_IW2_slc_burst#1.npy'
binary_mask    = 'S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794_RAW_IW2_binary_mask_burst#1.npy'
"""

burst_name='S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794'
burst_number = 4
raw_burst_file = f'{burst_name}_RAW_IW2_real_burst#{burst_number}.npy'
slc_burst_file = f'{burst_name}_RAW_IW2_slc_burst#{burst_number}.npy'
binary_mask    = f'{burst_name}_RAW_IW2_binary_mask_burst#{burst_number}.npy'

#raw_burst_file = 'S1B_IW_SLC__1SDV_20200607T051035_20200607T051102_021923_0299AC_0F7B_RAW_IW2_real_burst#4.npy'
#slc_burst_file = 'S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96_RAW_IW2_slc_burst#3.npy'
#binary_mask    = 'S1B_IW_SLC__1SDV_20200607T051035_20200607T051102_021923_0299AC_0F7B_RAW_IW2_binary_mask_burst#4.npy'
#Load raw burst
raw_burst =np.load(f'{raw_burst_dir}/{raw_burst_file}')

#load slc burst
slc_burst =np.load(f'{slc_burst_dir}/{slc_burst_file}')
#load binary mask
#mask_burst =np.load(f'{raw_burst_dir}/{binary_mask}')

#plot raw burst using plot_2d_array

crc=burst_name.split('_')[9]
plot_2d_array(raw_burst, f'Raw: {crc}_burst#{burst_number}', 'linear')
#plot slc
plot_2d_array(np.square(np.abs(slc_burst)), f'SLC: {crc}_burst#{burst_number}')
#plot binary
#plot_2d_array(mask_burst, f'Bin Mask: {binary_mask}', 'binary')

crop1=raw_burst[]



#input()
