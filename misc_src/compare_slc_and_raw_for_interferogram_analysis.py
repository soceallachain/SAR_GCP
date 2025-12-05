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
    max_roi  = roi.max()
    # Get mean of 10 brightest pixels
    avg10_max = np.sort(roi)[-10:].mean()
    return mean, std, SNR, max_roi, avg10_max


def intensity(arr):
    return np.square(np.abs(arr))

Dataset= 'Train'
source = 'ext'

if source == 'ext':
    raw_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_data'
    slc_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_data'
    slc_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_burst_data'

    #Get ship locations in SLC file
    path = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}'
    img_path=f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m'


elif source == 'int':    
    raw_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/raw_data'
    slc_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/slc_data'
    slc_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/raw_burst_data'
    
    #Get ship locations in SLC file
    path = f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}'
    img_path = f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m'

    
elif source == 'Spain':
    raw_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/raw_data'
    slc_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/slc_data'
    slc_burst_dir =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/raw_burst_data'

    #Get ship locations in SLC file
    path = f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}'
    img_path = f'/mnt/smbshare/RAWSARAI_Data/Copernicus/Sandbox/test_slc_raw_interferogram_results/{Dataset}_gt100m'


else:
    print(f'Source errorL Exiting.....')
    sys.exit()


#plt.ion()   
csv_path = f'{path}/Reports/'
#img_path='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results'

#  Read CSV
csv_filename =f'{Dataset}_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m.csv'
csv_file = os.path.join(csv_path, csv_filename)

# Output file for statistics
csv_statistics_file = os.path.join(img_path, csv_filename.replace('.csv','_Statistics.csv'))

# Extract burst names and ship locations
df             = pd.read_csv(csv_file)    
burst_names    = df['Burst_Name'].to_list()
ship_locations = df[['ship_burst_row', 'ship_burst_col']].to_numpy()
ship_lengths   = df['ship_length'].to_numpy()




crop_size      = [512, 512]
ship_crop_size = [150, 150]
kernel1    = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])



#Set visual check to True to manually inspecting interferogram
do_visual_check = False

burst_crop_data=[]

start = time.perf_counter()

for i in range(891,892):
#for i in range(len(ship_locations)):
    ship_location = ship_locations[i]
    b            = burst_names[i].split('_#')[0]
    burst_number = b.split('#')[1]
    burst_id     = b.split('_RAW')[0]
    ship_length  = ship_lengths[i]

    crc_burst           = f"{burst_id.split('_')[9]}_burst#{burst_number}"
    slc_burst_file      = f'{burst_id}_RAW_IW2_slc_burst#{burst_number}.npy'
    raw_real_burst_file = f'{burst_id}_RAW_IW2_real_burst#{burst_number}.npy'
    raw_imag_burst_file = f'{burst_id}_RAW_IW2_imag_burst#{burst_number}.npy'

    try:
        slc_burst      = np.load(f"{slc_burst_dir}/{slc_burst_file}")
        raw_real_burst = np.load(f"{raw_burst_dir}/{raw_real_burst_file}")
        raw_imag_burst = np.load(f"{raw_burst_dir}/{raw_imag_burst_file}")
    except FileNotFoundError as e:
        print(f"Error: File(s) for {crc_burst} not found: {e}. Continue to next ship in list .")
        continue  # exit the loop if any of the files are missing

    slc_intensity       = intensity(slc_burst)
    slc_intensity_filtered = exclude_high_outliers(slc_intensity,10)
    
    
    
    r,c, ship_loc_in_crop = place_roi(raw_real_burst, crop_size, ship_location)    
    
    cropped_raw_real_burst = raw_real_burst[r:r + crop_size[0],
                                            c:c + crop_size[1]]
    cropped_raw_imag_burst = raw_imag_burst[r:r + crop_size[0],
                                            c:c + crop_size[1]]
    #SLC is complex, 
    cropped_slc_burst = slc_burst[r:r + crop_size[0],
                                 c:c + crop_size[1]]
    #Get crop around ship of 150x150, ship located in crop at ship_loc_in_crop
    #crop_centre = [crop_size[0]//2, crop_size[1]//2]
    r,c, pos_in_crop = place_roi(cropped_raw_real_burst, ship_crop_size, 
                                 ship_loc_in_crop
                                 )    
    
    ship_crop_raw_real = cropped_raw_real_burst[r:r + ship_crop_size[0],
                                 c:c + ship_crop_size[1]]
    ship_crop_raw_imag = cropped_raw_imag_burst[r:r + ship_crop_size[0],
                                 c:c + ship_crop_size[1]]
    ship_crop_slc      = cropped_slc_burst[r:r + ship_crop_size[0],
                                 c:c + ship_crop_size[1]]
    ship_crop_slc_I    = np.real(ship_crop_slc)
    ship_crop_slc_Q    = np.real(ship_crop_slc)
    ship_crop_slc_intensity = intensity(ship_crop_slc)
    
    
    #Get pixels of crop outside of ship crop
    #Create mask to exclude ship crop
    #Use of mask flattens array so crop_without_ship* arrays are 1D
    mask = np.ones((crop_size[0], crop_size[1]), dtype=bool)
    mask[r:r+ship_crop_size[0], c:c+crop_size[1]] = False
    
    crop_without_ship_slc      = cropped_slc_burst[mask]
    crop_without_ship_raw_real = cropped_raw_real_burst[mask]
    crop_without_ship_raw_imag = cropped_raw_imag_burst[mask]
    crop_without_ship_slc_intensity = intensity(crop_without_ship_slc)
    
    
    #Get pixels associated with ship
    # Set threshold to mean + X std dev of signal in area without ship
    ship_detection_threshold = ( crop_without_ship_slc_intensity.mean()
                             + 3*crop_without_ship_slc_intensity.std()
                                )
    #Create mask for identifying ship pixels
    ship_crop_slc_above_threshold = ship_crop_slc_intensity > ship_detection_threshold
    # Process mask map to remove isolated pixels using erosion/dilation
    processed_ship_crop_mask = dilation(erosion(ship_crop_slc_above_threshold))
    ship_pixels = ship_crop_slc_intensity[processed_ship_crop_mask]
    
    
    
    
    
    #burst & crop statistics            
    raw_burst_I_stats = get_statistics(raw_real_burst)
    raw_burst_Q_stats = get_statistics(raw_imag_burst)
    slc_burst_I_stats = get_statistics(np.real(slc_burst))
    slc_burst_Q_stats = get_statistics(np.imag(slc_burst))
    
    slc_burst_intensity_stats = get_statistics(slc_intensity)
    slc_burst_intensity_filtered_stats = get_statistics(slc_intensity_filtered)
    
    raw_crop_I_stats  = get_statistics(cropped_raw_real_burst)
    raw_crop_Q_stats  = get_statistics(cropped_raw_imag_burst)
    slc_crop_I_stats  = get_statistics(np.real(cropped_slc_burst))
    slc_crop_Q_stats  = get_statistics(np.imag(cropped_slc_burst))        
    crop_slc_intensity_stats = get_statistics(intensity(cropped_slc_burst))
    
    crop_without_ship_slc_intensity_stats = get_statistics(
                                              crop_without_ship_slc_intensity
                                                )
    raw_ship_crop_I_stats = get_statistics(ship_crop_raw_real)
    raw_ship_crop_Q_stats = get_statistics(ship_crop_raw_imag)    
    slc_ship_crop_I_stats = get_statistics(ship_crop_slc_I)
    slc_ship_crop_Q_stats = get_statistics(ship_crop_slc_Q)
    slc_ship_crop_intensity_stats = get_statistics(ship_crop_slc_intensity)
    
    #Get ship statistics, size and brightness
    ship_stats  = get_statistics(ship_pixels)
    ship_size   = len(ship_pixels)
    ship_rcs    = ship_size * ship_pixels.mean()
    ###
    
    ####
    
    #Plot graphs
    # Plot SLC and equivalent RAW
    """
    img_name = f'SLC_{crc_burst}'
    f=plot_2d_array(np.square(np.abs(slc_burst)), img_name)
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    # Plot RAW burst
    img_name = f'SLC_matched_Raw_Burst_Real_{crc_burst}'
    f=plot_2d_array(raw_real_burst,img_name,'linear')
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    """
    
    #Plot multiple graphs on single plot
    fig, axs  = plt.subplots(3, 6, figsize=(16,9),constrained_layout=True)
    num_bins  = 100
    threshold = 10
    # For example assign the function's plot to subplot 2
    plot_fig_name = f'{crc_burst}_ship_{ship_location}_RAW_SLC_Histograms'
    img_name  = f"Raw_Burst_Real_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[0, 0], raw_real_burst, num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"Raw_Imag_Real_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[0, 1], raw_imag_burst,num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"SLC_Burst_Real_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[0, 2], np.real(slc_burst),num_bins,
                             threshold, 'gaussian',img_name)
    img_name = f"SLC_Burst_Imag_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[0, 3], np.imag(slc_burst),num_bins,
                             threshold, 'gaussian',img_name)
    
    img_name = f"SLC_Burst_Amplitude_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[0, 4], np.abs(slc_burst),num_bins,
                             threshold, 'gamma',img_name)
    img_name =f"SLC_Burst_Intensity_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[0, 5], np.square(np.abs(slc_burst)),num_bins,
                             threshold, 'weibull',img_name)
    
    
    #Plot Crop data with ship at centre ; crop size = crop_size
    
    img_name  = f"Raw_Real_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[1, 0], cropped_raw_real_burst, num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"Raw_Imag_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[1, 1], cropped_raw_imag_burst,num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"SLC_Real_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[1, 2], np.real(cropped_slc_burst),num_bins,
                             threshold, 'gaussian',img_name)
    img_name = f"SLC_Imag_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[1, 3], np.imag(cropped_slc_burst),num_bins,
                             threshold, 'gaussian',img_name)
    
    img_name = f"SLC_Crop_Amplitude_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[1, 4], np.abs(cropped_slc_burst),num_bins,
                             threshold, 'gamma',img_name)
    img_name =f"SLC_Crop_Intensity_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[1, 5], np.square(np.abs(cropped_slc_burst)),num_bins,
                             threshold, 'weibull',img_name)
    
    #Plot data for ship only with crop size = ship_crop_size
    #In this data outliers are not excluded (threshold == 0)
    
    threshold = 0
    img_name  = f"Raw_Real_Ship_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[2, 0], ship_crop_raw_real, num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"Raw_Imag_Ship_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[2, 1], ship_crop_raw_imag,num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"SLC_Real_Ship_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[2, 2], np.real(ship_crop_slc),num_bins,
                             threshold, 'gaussian',img_name)
    img_name = f"SLC_Imag_Ship_Crop_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[2, 3], np.imag(ship_crop_slc),num_bins,
                             threshold, 'gaussian',img_name)
    
    img_name = f"SLC_Ship_Crop_Amp_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[2, 4], np.abs(ship_crop_slc),num_bins,
                             threshold, '',img_name)
    img_name =f"SLC_Ship_Crop_Intensity_{crc_burst}"
    plot_hist_with_fit_on_ax(axs[2, 5], intensity(ship_crop_slc),1000,
                             threshold, '',img_name)
    
    
    
    
    #plt.show()   
    
   

    
    
    # Plot burst, crop around ship, equivalent RAW (and focused/compressed RAW)?
    # #Plot multiple graphs on single plot
    fig1, axs1  = plt.subplots(2, 3, figsize=(16,9),constrained_layout=True)
    
    font_size =8
    plot_name = f'{crc_burst}_ship_{ship_location}_RAW_SLC'
    img_name  = f"SLC_{crc_burst}"
    plot_2d_array_on_ax(axs1[0, 0], slc_intensity,img_name,'log_normal',font_size)
    # Highlight ship:  Place circle around ship
    text_offset=[200,200]
    add_text_to_plot(axs1[0,0], f'Ship @ [{ship_location[0]},'
                     f'{ship_location[1]}], Length={np.round(ship_length)}', 
                     ship_location,text_offset
                     )
    circle = patches.Circle((ship_location[1], ship_location[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)    
    axs1[0,0].add_patch(circle)    
    img_name = f"SLC_crop_for_ship_{ship_location}_in_{crc_burst}"
    plot_2d_array_on_ax(axs1[0, 1], intensity(cropped_slc_burst),img_name,'log_normal',font_size)    
    img_name = f"RAW_crop_{crc_burst}"
    plot_2d_array_on_ax(axs1[1, 0], cropped_raw_real_burst,img_name,'linear',font_size)
    img_name = f"Range_compressed_Raw_crop_{crc_burst}"
    plot_2d_array_on_ax(axs1[1, 1], 
                        intensity(range_compress_burst(cropped_raw_real_burst)),
                        img_name,'log_normal',font_size
                        )
    
    img_name = f"Ship_crop_{crc_burst}"
    plot_2d_array_on_ax(axs1[0, 2], 
                       ship_crop_slc_intensity,
                        img_name,'log_normal',font_size
                        )
    img_name =f'Ship_only_pixels_after_masking_{crc_burst}'
    plot_2d_array_on_ax(axs1[1, 2], 
                       ship_crop_slc_intensity*processed_ship_crop_mask +1,
                        img_name,'log_normal',font_size
                        )
    
    
    
    big_crop_size = [1000,1000]
    r,c, pos_in_crop = place_roi(raw_real_burst, big_crop_size, ship_location)    
    
    cropped_raw_real_burst2 = raw_real_burst[r:r + big_crop_size[0], c:c + big_crop_size[1]]
    img_name = f"Large_Raw_crop_{crc_burst}"
    #plot_2d_array_on_ax(axs1[2, 0], cropped_raw_real_burst2,img_name,'linear')
   
    #plt.show()   
    
    
    #plt.pause(1)
    if do_visual_check:
        while True:
            try:
                response = int(input(f'What is strength of interferogram on'
                                     f' scale of 0 to 2; where 0 indicates'
                                     f' non-visibility, 1 weakly present and'
                                     f' 2 strongly present: '
                ))
                if response in (0, 1, 2):
                    signal_strength = response
                    break
                else:
                    print('Invalid response. Please enter 0, 1, or 2.')
            except ValueError:
                print('Input must be an integer. Please try again.')
    else:
        response=-1
        
    fig.savefig(f'{img_path}/{plot_fig_name}.jpg', format='jpg', 
             bbox_inches='tight', dpi=600
             )
 
    fig1.savefig(f'{img_path}/{plot_name}.jpg', format='jpg', 
            bbox_inches='tight', dpi=600
            )    
    
    #Assign rows to statistics dataframe
    burst_crop_data.append({
        
        "Ship Number"      : i,
        "Burst Name"       : crc_burst,
        "Ship Length"      : ship_length,
        "Ship Row"         : ship_location[0],
        "Ship Col"         : ship_location[1],
        "Signal Strength"  : response,
        "Burst Raw I Mean" : raw_burst_I_stats[0], 
        "Burst Raw Q Mean" : raw_burst_Q_stats[0], 
        "Burst Raw I StDev": raw_burst_I_stats[1], 
        "Burst Raw Q StDev": raw_burst_Q_stats[1],  
        "Burst SLC I Mean" : slc_burst_I_stats[0], 
        "Burst SLC Q Mean" : slc_burst_Q_stats[0], 
        "Burst SLC I StDev": slc_burst_I_stats[1], 
        "Burst SLC Q StDev": slc_burst_Q_stats[1],    
        "Burst SLC Intensity Mean"  : slc_burst_intensity_stats[0],         
        "Burst SLC Intensity StDev" : slc_burst_intensity_stats[1],         
        "Crop Raw I Mean"  : raw_crop_I_stats[0], 
        "Crop Raw Q Mean"  : raw_crop_Q_stats[0], 
        "Crop Raw I StDev" : raw_crop_I_stats[1], 
        "Crop Raw Q StDev" : raw_crop_Q_stats[1],  
        "Crop SLC I Mean"  : slc_crop_I_stats[0], 
        "Crop SLC Q Mean"  : slc_crop_Q_stats[0], 
        "Crop SLC I StDev" : slc_crop_I_stats[1], 
        "Crop SLC Q StDev" : slc_crop_Q_stats[1],        
        "Crop SLC Intensity Mean"  : crop_slc_intensity_stats[0],         
        "Crop SLC Intensity StDev" : crop_slc_intensity_stats[1], 
            
           
        "Ship Crop Raw I Mean"  : raw_ship_crop_I_stats[0], 
        "Ship Crop Raw Q Mean"  : raw_ship_crop_Q_stats[0], 
        "Ship Crop Raw I StDev" : raw_ship_crop_I_stats[1], 
        "Ship Crop Raw Q StDev" : raw_ship_crop_Q_stats[1],  
        "Ship Crop SLC I Mean"  : slc_ship_crop_I_stats[0], 
        "Ship Crop SLC Q Mean"  : slc_ship_crop_Q_stats[0], 
        "Ship Crop SLC I StDev" : slc_ship_crop_I_stats[1],
        "Ship Crop SLC Q StDev" : slc_ship_crop_Q_stats[1],        
        "Ship Crop SLC Intensity Mean"  : slc_ship_crop_intensity_stats[0],         
        "Ship Crop SLC Intensity StDev" : slc_ship_crop_intensity_stats[1], 
        "CWS Mean"      : crop_without_ship_slc_intensity_stats[0],
        "CWS StdDev"    : crop_without_ship_slc_intensity_stats[1],
        "Ship Mean RCS" : ship_stats[0],
        "Ship StDev"    : ship_stats[1],
        "Ship RCS"      : ship_rcs,
        "Ship Area"     : ship_size
        
    })
    #Convert to DataFrame
    df1 = pd.DataFrame(burst_crop_data)
    # Save to CSV with column header after every ship in case bombs
    # It's not efficient but it will work
    df1.to_csv(csv_statistics_file)
        
        
    
    #Close figures before start next iteration    
    fig.clf()
    fig1.clf()
    plt.close(fig)
    plt.close(fig1)
    print(f'Ship Number {i+1} completed for location {ship_location} in burst {crc_burst}')
    
    
    
    

end = time.perf_counter()
print(f"Execution time for {i+1} ships: {end - start:.6f} seconds")
#print(f"Execution time for {len(ship_locations)} ships: {end - start:.6f} seconds")
