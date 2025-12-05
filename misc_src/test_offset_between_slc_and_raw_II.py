#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 06:51:44 2025

@author: joe
"""

# Script to gauge degree of offset between positions in RAW, SLC and RDA
# focused RAW


import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from SAR_focusing_module import *
from blob_module import *

from SAR_code_module import *


def get_statistics(roi):    
    mean = roi.mean()
    std  = roi.std()
    SNR  = mean/std
    return mean, std, SNR


def intensity(arr):
    return np.square(np.abs(arr))

plt.style.use('dark_background')
source = 'ext'
Dataset = 'Train'

if source == 'ext':
    raw_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_data'
    slc_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_data'
    slc_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_burst_data'

    #Get ship locations in SLC file
    path = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}'
    #img_path=f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m'
    img_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results/Offset_tests_20Aug2025'


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
    print(f'Source error Exiting.....')
    sys.exit()


#plt.ion()   
csv_path = f'{path}/Reports/'
#img_path='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results'

#  Read CSV
#csv_filename =f'{Dataset}_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m_Statistics.csv'
csv_filename =f'{Dataset}_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m_Statistics_with_SLC_ID.csv'
test_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results'
#csv_filename =f'Bursts_Ships_To_Test_Offset.csv'
csv_results_file =f'{csv_filename.replace(".csv","_results.csv")}'
csv_file = os.path.join(test_dir, csv_filename)
df = pd.read_csv(csv_file)

# Get list of bursts to test

burst_names    = df['Burst_Name'].to_list()
ship_locations = df[['Ship Row', 'Ship Col']].to_numpy()
ship_lengths   = df['Ship Length'].to_numpy()
cws_means      = df['CWS Mean'].to_numpy()
cws_stdevs     = df['CWS StdDev'].to_numpy()


slc_detected_row      = []
slc_detected_col      = []
sm_focus_detected_row = []
sm_focus_detected_col = []
azimuth_sm_slc_offset        = []
range_sm_slc_offset          = []
   

crop_size      = [1024, 1024]
"""
#get scene_ids for bursts
scene_id=[]
for burst in burst_names:
    b            = burst.split('_#')[0]
    burst_number = b.split('#')[1]
    burst_id     = b.split('_RAW')[0]
    scene_id_data =get_scene_id_and_dataset(burst_id)
    scene_id.append(scene_id_data[0])
    scene_id.append(burst_id)
    print(f'Burst Name is: {burst}\n')
    
i=1
with open(f'{test_dir}/raw_files_to_get.txt', 'w') as f1, open(f'{test_dir}/slc_files_to_get.txt', 'w') as f2:

    for item in scene_id:
        if i==1:
            f1.write(f"{item}\n")
        else:
            f2.write(f"{item}\n")
        i=i*-1
"""
for i in range(0,1):
#for i in range(len(ship_locations)):
    ship_location = ship_locations[i]
    cws_mean      = cws_means[i]
    cws_stdev     = cws_stdevs[i]
    ship_length   = ship_lengths[i]
    b            = burst_names[i].split('_#')[0]
    burst_number = int(b.split('#')[1])
    burst_id     = b.split('_RAW')[0]
    
    
    crc_burst           = f"{burst_id.split('_')[9]}_burst#{burst_number}"
    slc_burst_file      = f'{burst_id}_RAW_IW2_slc_burst#{burst_number}.npy'
    raw_real_burst_file = f'{burst_id}_RAW_IW2_real_burst#{burst_number}.npy'
    raw_imag_burst_file = f'{burst_id}_RAW_IW2_imag_burst#{burst_number}.npy'

    try:
        slc_burst      = np.load(f"{slc_burst_dir}/{slc_burst_file}")
        raw_real_burst = np.load(f"{raw_burst_dir}/{raw_real_burst_file}")
        raw_imag_burst = np.load(f"{raw_burst_dir}/{raw_imag_burst_file}")
    except FileNotFoundError as e:
        print(f"Error: File(s) for {crc_burst} not found: {e}. Continurange_offsete to next ship in list .")
        continue  # exit the loop if any of the files are missing

    slc_intensity         = intensity(slc_burst)
    r,c, ship_loc_in_crop = place_roi(raw_real_burst, crop_size, ship_location)    
    
    raw_real_crop = raw_real_burst[r:r + crop_size[0],
                                   c:c + crop_size[1]]
    raw_imag_crop = raw_imag_burst[r:r + crop_size[0],
                                   c:c + crop_size[1]]
    #SLC is complex, 
    slc_crop = slc_burst[r:r + crop_size[0],
                         c:c + crop_size[1]]
    
    #Get RDA SM focused burst with no zero padding        
    data = get_raw_burst_with_offset_info(burst_id, burst_number, swath_of_interest=2, 
                                       polarization='vh', source='ext')
    
    #An slc-matched-raw-burst is one that has dimensions that match with the 
    #equivalent slc file ensuring de-burstment and offsets aligned    
    slc_matched_raw_burst = data[0]
    raw_burst    = data[1]
    az_offset    = data[2]
    az_end       = data[3]
    range_offset = data[4]
    range_end    = data[5]
    meta_data    = data[6]
    p_l0file     = data[7]
    
    range_length   = range_end-range_offset
    azimuth_length = az_end-az_offset
    #ROI for burst uses default range and azimuth offsets with r =0, c=0
    roi = np.array([range_offset, range_length, az_offset, azimuth_length])
    # Want to focus raw burst that matched SLC timings
    rzp = 0
    azp = 0
    
    
    sm_focused_burst = focus_SM_burst_with_zero_padding(meta_data, 
                                         slc_matched_raw_burst,
                                         roi, p_l0file, 'rma', rzp, azp
                                          )
    img_name = f'Full_focus_{crc_burst}_ship_at_{ship_location}_range_padding_{rzp}_az_padding_{azp}'
    """
    plot_2d_array(intensity(sm_focused_burst),img_name)     
    plot_2d_array(slc_intensity)
    plot_2d_array(intensity(slc_crop))
    plot_2d_array(raw_real_crop,'raw_crop','linear')
    """
   
    cropped_SM_focused_burst=sm_focused_burst[r:r + crop_size[0],
                                              c:c + crop_size[1]]
    # Want to see difference between cropping focused burst and focusing crop
    img_name = f'Crop_of_Full_focus_{crc_burst}_ship_at_{ship_location}_range_padding_{rzp}_az_padding_{azp}'
    #f=plot_2d_array(intensity(cropped_SM_focused_burst),img_name)     
    
    
    
    new_roi = np.array([range_offset + c, crop_size[1], 
                        az_offset    + r, crop_size[0]
                        ])
    raw_crop = slc_matched_raw_burst[r:r + crop_size[0],
                                     c:c + crop_size[1]]
    sm_focused_raw_crop = focus_SM_burst_with_zero_padding(meta_data, raw_crop,
                                            new_roi, p_l0file, 'rma', rzp, azp
                                            )
    img_name = f'SM Focused Crop of {crop_size} around {ship_location} in {crc_burst}'
    #f=plot_2d_array(intensity(sm_focused_raw_crop),img_name)     
    
    
   


    # Show slc ship crop
    crop_half_size=int(crop_size[0]/2)
    ship_crop_size =[crop_half_size, crop_half_size]
    ship_crop_half_size =int(ship_crop_size[0]/2)
    slc_ship_crop =slc_intensity[r+ship_crop_half_size:r+ship_crop_half_size+crop_half_size, 
                                 c+ship_crop_half_size:c+ship_crop_half_size+crop_half_size]
    #plot_2d_array(slc_ship_crop)
    #threshold slc cws_mean + 3*csw_Stdev
    bin_slc_crop = slc_ship_crop > (cws_mean + 3*cws_stdev)
    bin_slc_crop = dilation(erosion(bin_slc_crop))
    blob_data    = detect_blobs(bin_slc_crop, 10)
    centre_of_mass = blob_data[1][np.argmax(blob_data[3])]
    detected_slc_ship_coords_r_c = [r+ship_crop_half_size + centre_of_mass[0],
                                    c+ship_crop_half_size + centre_of_mass[1]]
    #plot___blobs(slc_ship_crop,blob_data[2])
                 
    # Show sm_focused_crop
    sm_intensity=intensity(sm_focused_burst)
    sm_ship_crop =sm_intensity[r+ship_crop_half_size:r+ship_crop_half_size+crop_half_size,
                               c+ship_crop_half_size:c+ship_crop_half_size+crop_half_size]
    #plot_2d_array(sm_ship_crop)
    #threshold sm_focused :mean + 3*Stdev
    bin_sm_crop = sm_ship_crop > (sm_intensity.mean() + 3*sm_intensity.std())
    bin_sm_crop = dilation(erosion(bin_sm_crop))
    blob_data   = detect_blobs(bin_sm_crop, 10)
    centre_of_mass = blob_data[1][np.argmax(blob_data[3])]
    detected_sm_ship_coords_r_c = [r+ship_crop_half_size + centre_of_mass[0],
                                    c+ship_crop_half_size + centre_of_mass[1]]
    #plot___blobs(sm_ship_crop,blob_data[2])            
    slc_sm_focused_offset = (np.array(detected_sm_ship_coords_r_c) 
                            - np.array(detected_slc_ship_coords_r_c)
                             ).astype(int)
    
    slc_detected_row.append(int(detected_slc_ship_coords_r_c[0]))
    slc_detected_col.append(int(detected_slc_ship_coords_r_c[1]))
    sm_focus_detected_row.append(int(detected_sm_ship_coords_r_c[0]))
    sm_focus_detected_col.append(int(detected_sm_ship_coords_r_c[1]))
    azimuth_sm_slc_offset.append(slc_sm_focused_offset[0])
    range_sm_slc_offset.append(slc_sm_focused_offset[1])
       
    #plot slc burst, raw burst, sm focused burst
    fig, axs  = plt.subplots(3, 1, figsize=(16,9),constrained_layout=True)
    plot_fig_name = f'{crc_burst}_{ship_location}_SLC_RAW_&_SM_Focused'
    fig.suptitle(f'{plot_fig_name}_Offset_{slc_sm_focused_offset}')
    font_size =8
   
    img_name=['SLC', 'RAW', 'SM_Focused']
    plot_type=['log_normal', 'linear', 'log_normal']
    sar_data_to_plot=[slc_intensity, raw_real_burst, intensity(sm_focused_burst)]
    az_sgn = -1 if ship_location[0] < azimuth_length / 2 else 1
    range_sgn = -1 if ship_location[1] > range_length / 2 else 1
       
    for i in range(0,3):
        
        grid_colour='gray' if i==0 or i==2 else 'red'
        plot_2d_array_on_ax(axs[i], sar_data_to_plot[i],img_name[i],
                            plot_type[i],font_size, grid_on=1, grid_colour=grid_colour
                            )
        if i==0:
            # Highlight ship:  Place circle around ship
            text_offset=[az_sgn*200,range_sgn*2000]
            add_text_to_plot(axs[i], f'Ship Labelled @ [{ship_location[0]},'
                         f'{ship_location[1]}]',
                         ship_location,text_offset, colour='white'
                         )
            circle = patches.Circle((ship_location[1], ship_location[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
            axs[i].add_patch(circle)
            # Add text for detected position of ship
            text_offset=[text_offset[0]*-1,text_offset[1]+100]
            add_text_to_plot(axs[i], f'Ship Detected @ [{int(detected_slc_ship_coords_r_c[0])},'
                         f'{int(detected_slc_ship_coords_r_c[1])}]',
                         detected_slc_ship_coords_r_c,text_offset, 
                         colour='white', arrow_colour='green'
                         )
            circle = patches.Circle((detected_slc_ship_coords_r_c[1], 
                                     detected_slc_ship_coords_r_c[0]), 
                                    radius=50, edgecolor='green', 
                                    facecolor='none', linewidth=1
                                    )
            axs[i].add_patch(circle)
        elif i==2:
            # Highlight ship:  Place circle around ship
            
            text_offset=[az_sgn*200,range_sgn*2000]
            add_text_to_plot(axs[i], f'Expected Ship @ [{ship_location[0]},'
                         f'{ship_location[1]}]',
                         ship_location,text_offset, colour = 'white'
                         )
            circle = patches.Circle((ship_location[1], ship_location[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
            axs[i].add_patch(circle)
            # Add text for detected position of ship
            text_offset=[text_offset[0]*-1,text_offset[1]+100]
            add_text_to_plot(axs[i], f'Ship Detected @ [{int(detected_sm_ship_coords_r_c[0])},'
                         f'{int(detected_sm_ship_coords_r_c[1])}]',
                         detected_sm_ship_coords_r_c,text_offset, 
                         colour='white', arrow_colour='green'
                         )
            circle = patches.Circle((detected_sm_ship_coords_r_c[1], 
                                     detected_sm_ship_coords_r_c[0]), 
                                    radius=50, edgecolor='green', 
                                    facecolor='none', linewidth=1
                                    )
            axs[i].add_patch(circle)
            
    fig.savefig(f'{img_path}/{plot_fig_name}.jpg', format='jpg',
             bbox_inches='tight', dpi=600
             )
    fig.clf()
    
    
    
    
    
    
    test_offsets_with_blobs =0
    if test_offsets_with_blobs:
        threshold = cws_mean + 3*cws_stdev
        detected_slc_ship_info = get_largest_blobs_centroid(intensity(slc_crop),
                                                            ship_loc_in_crop,
                                                            threshold
                                                            )
        detected_sm_focused_ship_info = get_largest_blobs_centroid(
                                                 intensity(sm_focused_raw_crop),
                                                 ship_loc_in_crop,
                                                 threshold
                                                 )
        detected_slc_loc = np.array(detected_slc_ship_info[0])
        detected_sm_focused_loc = np.array(detected_sm_focused_ship_info[0])
        slc_sm_focus_offset = detected_slc_loc - detected_sm_focused_loc
        #Overlay 
        
    
        contours = np.argwhere(blob)
        y, x = zip(*contours)
        axs[0].scatter(x+c+256, y+r+256, color='black', s=1)
    
    


    test_zero_padding = 0
    if test_zero_padding: 
        fig, axs  = plt.subplots(4, 4, figsize=(16,9),constrained_layout=True)
        font_size =8
        plot_fig_name = f'{crc_burst}_ship_{ship_location}_offset_w_zero_padding'
        plot_type=['log_normal', 'linear', 'log_normal','log_normal']
        img_name=['SLC', 'RAW', 'Crop of SM_Focused', 'SM_Focused_Crop -No Zero Padding']
        sar_data_to_plot=[intensity(slc_crop), np.real(raw_crop), 
                          intensity(cropped_SM_focused_burst), 
                          intensity(sm_focused_raw_crop)
                          ]
        for i in range(0,4):
            
            plot_2d_array_on_ax(axs[0,i], sar_data_to_plot[i],img_name[i],plot_type[i],font_size, grid_on=1)
            if i==0:
                # Highlight ship:  Place circle around ship
                text_offset=[200,200]
                add_text_to_plot(axs[0,i], f'Ship @ [{ship_loc_in_crop[0]},'
                             f'{ship_loc_in_crop[1]}]',
                             ship_loc_in_crop,text_offset, colour='white'
                             )
                circle = patches.Circle((ship_loc_in_crop[1], ship_loc_in_crop[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
                axs[0,i].add_patch(circle)
            elif i==2 or i==3:
                # Highlight ship:  Place circle around ship
                text_offset=[200,200]
                add_text_to_plot(axs[0,i], f'Expected Ship @ [{ship_loc_in_crop[0]},'
                             f'{ship_loc_in_crop[1]}]',
                             ship_loc_in_crop,text_offset, colour ='white'
                             )
                circle = patches.Circle((ship_loc_in_crop[1], ship_loc_in_crop[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
                axs[0,i].add_patch(circle)
        #Set zero padding sizes
        azp=[100,200,300]
        rzp=[500,1000,2000,4000]        
        plot_type=['log_normal', 'linear', 'log_normal','log_normal']
        img_name = 'SM_Focused_Crop'
        for j in range(1,4):
            for i in range(0,4):
                #get focused crop with azp and rzp levels of padding
                sm_focused_crop = focus_SM_burst_with_zero_padding(meta_data, 
                                                    raw_crop, new_roi, p_l0file, 
                                                    'rma',rzp[i]+raw_crop.shape[0],
                                                    azp[j-1] + raw_crop.shape[1]
                                                    )
                img_name =f'SM_focused_Crop_azp={azp[j-1]} rzp={rzp[i]}'
                plot_2d_array_on_ax(axs[j,i], intensity(sm_focused_crop),img_name,
                                    fontsize=8, grid_on=1)
                text_offset=[50,50]
                add_text_to_plot(axs[j,i], f'Expected Ship @ [{ship_loc_in_crop[0]},'
                             f'{ship_loc_in_crop[1]}]',
                             ship_loc_in_crop,text_offset, colour='white'
                             )
                circle = patches.Circle((ship_loc_in_crop[1], ship_loc_in_crop[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
                axs[j,i].add_patch(circle)
                
        fig.savefig(f'{img_path}/{plot_fig_name}.jpg', format='jpg',
                 bbox_inches='tight', dpi=600
                 ) 
        fig.clf()

if len(azimuth_sm_slc_offset) == df['Ship Number'].count():
    df['SLC Detected Row']         = slc_detected_row
    df['SLC Detected Column']      = slc_detected_col
    df['SM Focus Detected Row']    = sm_focus_detected_row
    df['SM Focus Detected Column'] = sm_focus_detected_col
    df['Azimuth Offset']           = azimuth_sm_slc_offset
    df['Range Offset']             = range_sm_slc_offset
    csv_outputfile = os.path.join(test_dir, csv_results_file)
    df.to_csv(csv_outputfile)
else:
    
    
    df1 = pd.DataFrame() 
    df1['SLC Detected Row']         = slc_detected_row
    df1['SLC Detected Column']      = slc_detected_col
    df1['SM Focus Detected Row']    = sm_focus_detected_row
    df1['SM Focus Detected Column'] = sm_focus_detected_col
    df1['Azimuth Offset']           = azimuth_sm_slc_offset
    df1['Range Offset']             = range_sm_slc_offset
    csv_outputfile = os.path.join(test_dir, csv_results_file)
    df1.to_csv(csv_outputfile)
           
            
