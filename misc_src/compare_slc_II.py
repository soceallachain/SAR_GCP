#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:28:43 2025

@author: joe
"""

import numpy as np
#import matplotlib as plt
from skimage.measure import find_contours, label
import itertools
from SAR_code_module import *
import platform
import psutil
import os

from SAR_focusing_module import *
from pypapi import events, papi_high
import time
import pickle


def get_statistics(roi):    
    mean = roi.mean()
    std  = roi.std()
    SNR  = mean/std
    return mean, std, SNR




Dataset= 'Public'
source = 'int'

if source == 'ext':
    raw_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_data'
    slc_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_data'
    slc_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_burst_data'

    #Get ship locations in SLC file
    path = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}'

elif source == 'int':    
    raw_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/raw_data'
    slc_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/slc_data'
    slc_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/raw_burst_data'
    
    #Get ship locations in SLC file
    path = f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}'
    
elif source == 'Spain':
    raw_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/raw_data'
    slc_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/slc_data'
    slc_burst_dir =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/raw_burst_data'

    #Get ship locations in SLC file
    path = f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}'


else:
    print(f'Source errorL Exiting.....')
    sys.exit()


    
csv_path = f'{path}/Reports/'
img_path='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results'

#  Read CSV
csv_file = os.path.join(csv_path, 'Public_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m.csv')

# Extract burst names and ship locations
df             = pd.read_csv(csv_file)    
burst_names    = df['Burst_Name'].to_list()
ship_locations = df[['ship_burst_row', 'ship_burst_col']].to_numpy()




crop_size=[512, 512]


columns = [
    "Ship Number", "Burst Name", "Signal Strength",
    "Burst Mean", "Burst StdDev", "Ship Row"
    "Ship Col", "Crop Mean", "Crop StdDev",
    "CWS Mean", "CWS StdDev",
    "Ship Crop Power"
]

# Create an empty DataFrame with these columns
df = pd.DataFrame(columns=columns)

for i in range(2,3):
#for i in range(len(ship_locations)):
    b = burst_names[i].split('_#')[0]
    burst_number =b.split('#')[1]
    burst_id     =b.split('_RAW')[0]

    crc_burst           = f"{burst_id.split('_')[9]}_burst#{burst_number}"
    slc_burst_file      = f'{burst_id}_RAW_IW2_slc_burst#{burst_number}.npy'
    slc_burst           = np.load(f'{slc_burst_dir}/{slc_burst_file}')
    raw_real_burst_file = f'{burst_id}_RAW_IW2_real_burst#{burst_number}.npy'
    raw_real_burst      = np.load(f'{raw_burst_dir}/{raw_real_burst_file}')
    raw_imag_burst_file = f'{burst_id}_RAW_IW2_imag_burst#{burst_number}.npy'
    raw_imag_burst      = np.load(f'{raw_burst_dir}/{raw_imag_burst_file}')
    # Plot SLC and equivalent RAW

    img_name = f'SLC_{crc_burst}'
    f=plot_2d_array(np.square(np.abs(slc_burst)), img_name)
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    # Plot RAW burst
    img_name = f'SLC_matched_Raw_Burst_Real_{crc_burst}'
    f=plot_2d_array(raw_burst,img_name,'linear')
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)

    #Plot crops about ship location
    ship_location = ship_locations[i]
    r,c = place_roi(raw_burst, crop_size, ship_location)    
    
    cropped_raw_real_burst = raw_real_burst[r:r + crop_size[0],
                                            c:c + crop_size[1]]
    cropped_raw_imag_burst = raw_imag_burst[r:r + crop_size[0],
                                            c:c + crop_size[1]]
    
    #SLC is complex, 
    cropped_slc_burst = slc_burst[r:r + crop_size[0],
                                 c:c + crop_size[1]]
    img_name = f"Raw_Burst_Real_Cropped_around_ship_{ship_location}__{crop_size}_{crc_burst}"
    f=plot_2d_array(cropped_raw_real_burst,img_name,'linear')
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    
    img_name = f"SLC_Burst__Cropped_around_ship_{ship_location}__{crop_size}_{crc_burst}"
    f=plot_2d_array(np.square(np.abs(cropped_slc_burst)),img_name)
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    #plt.close('all')    
    
    
    #Get crop around ship with geometric aspect ratio 6:1 range-azimuth
    #as azimuth pixels are ~14m, while range are ~2.3
    
    gcrop_size = [85, 500]
    xcrop_size = [50, 50]
    r,c = place_roi(raw_burst, gcrop_size, ship_location)    
    
    cropped_raw_burst= raw_burst[r:r + gcrop_size[0],
                                 c:c + gcrop_size[1]]
    
    cropped_slc_burst= np.square(np.abs(slc_burst[r:r + gcrop_size[0],
                                 c:c + gcrop_size[1]]))
    
    #burst statistics
    raw_burst_I_stats = get_statistics(raw_real_burst)
    raw_burst_Q_stats = get_statistics(raw_imag_burst)
    slc_burst_I_stats = get_statistics(np.real(slc_burst))
    slc_burst_Q_stats = get_statistics(np.imag(slc_burst))
    raw_crop_I_stats = get_statistics(cropped_raw_real_burst)
    raw_crop_Q_stats = get_statistics(cropped_raw_imag_burst)
    slc_crop_I_stats = get_statistics(np.real(cropped_slc_burst))
    slc_crop_Q_stats = get_statistics(np.imag(cropped_slc_burst))
    
    slc_I = np.real(slc_burst)
    slc_Q = np.imag(slc_burst)
    #plot histogram of I and Q
    
    plot_hist_with_fit(slc_I,50,5,'gaussian', f'SLC_I {crc_burst}')
    plot_hist_with_fit(slc_Q,50,5,'gaussian', f'SLC_Q {crc_burst}')
    
    slc_intensity = np.square(np.abs(slc_burst))
    mean_slc_intensity = slc_intensity.mean()
    std_slc_intensity  = slc_intensity.std()
    slc_intensity_filtered = exclude_high_outliers(slc_intensity,10)
    mean_filtered = slc_intensity_filtered.mean()
    std_filtered  = slc_intensity_filtered.std()
    
    num_bins = 30
    thresh   = 10  #Values > mean + 10*sigma excluded
    plot_hist_with_fit(slc_intensity, num_bins, thresh,'weibull','Slc Intensity: {crc_burst}')
    plot_hist_with_fit(np.sqrt(slc_intensity), num_bins, thresh,'gamma', 'Slc Amplitude: {crc_burst}')
    #Plot crops side by side 5x3
    
    #local crop around ship statistics
    mean_crop = cropped_slc_burst.mean()
    std_crop  = cropped_slc_burst.std()
    
  
    crop_without_ship = np.hstack((cropped_slc_burst[:, 0:151], cropped_slc_burst[:, 350:500]))
    mean_cws = crop_without_ship.mean()
    std_cws = crop_without_ship.std()
    #Intensity plots
    plot_hist_with_fit(crop_without_ship, num_bins, thresh,'weibull',f'{crc_burst}_ship#{i}')
    plot_hist_with_fit(crop_without_ship, num_bins, thresh,'gamma',f'{crc_burst}_ship#{i}')
    #Amplitude plot
    plot_hist_with_fit(np.sqrt(crop_without_ship), num_bins, thresh,'gamma',f'{crc_burst}_ship#{i}')
    
    
    #Get ship brightness/area 
    r,c = place_roi(raw_burst, xcrop_size, ship_location)    
    xcropped_slc_burst= np.square(np.abs(slc_burst[r:r + xcrop_size[0],
                                 c:c + xcrop_size[1]]))
    xcrop_mean = xcropped_slc_burst.mean()
    xcrop_std  = xcropped_slc_burst.std()
    xcrop_power=sum(xcropped_slc_burst)
    pixels_gt_mean_f=xcropped_slc_burst[xcropped_slc_burst > mean_filtered]
    pixels_gt_mean_c=xcropped_slc_burst[xcropped_slc_burst > mean_cws]
    
    img_name = f"SLC_Burst__xCropped_around_ship_{ship_location}__{crop_size}_{crc_burst}"
    f=plot_2d_array(xcropped_slc_burst,img_name)
    img_name = f"SLC_Burst_Cropped_around_ship_{ship_location}__{crop_size}_{crc_burst}"
    f=plot_2d_array(cropped_slc_burst,img_name)
    #f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    
    #Set thresholds 
    mask = cropped_slc_burst > mean_cws + 3*std_cws
    masked_crop = cropped_slc_burst * mask.astype(int)
    img_name = f"SLC_Burst_masked_Cropped_around_ship_{ship_location}__{crop_size}_{crc_burst}"
    f=plot_2d_array(masked_crop,img_name)
    
    
