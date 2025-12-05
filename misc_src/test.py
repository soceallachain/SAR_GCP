#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 18:34:59 2025

@author: joe
"""
# Script to display raw, slc and rc_raw bursts that contain non-ships
# for the purpose of evaluating whether the moire pattern of stationary
# objects differs from non-stationary ones
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



# For element of list extract raw and slc burst name
def process_row(row, raw_burst_dir, slc_burst_dir, test_dir):
    # The 'row' is a Series, just like in iterrows()
    # You can extract the columns you need
    burst_name   = row['Burst_Name']
    location_col = row['ship_burst_col']
    location_row = row['ship_burst_row']
    length       = row['ship_length']
    location     = [int(location_row), int(location_col)]
    gps_lat      = row['gps_lat']
    gps_long     = row['gps_lon']
    index        = row.name
    
    
    real_raw_burst_name   = "_".join(burst_name.split('_')[:14]) +'.npy'
    imag_raw_burst_name   = real_raw_burst_name.replace('real','imag')
    slc_burst_name        = real_raw_burst_name.replace('real','slc')
    try:
        real_raw_burst        = np.load(f'{raw_burst_dir}/{real_raw_burst_name}')
        imag_raw_burst        = np.load(f'{raw_burst_dir}/{imag_raw_burst_name}')
        slc_burst             = np.load(f'{slc_burst_dir}/{slc_burst_name}')
    except:
        
        print(f'File {raw_burst_dir}/{real_raw_burst_name} not found. Continue to next file....')
        return
      
    
        
    slc_matched_raw_burst = real_raw_burst +  1j*imag_raw_burst
    
    rc_burst            = range_compress_padded_raw_data(slc_matched_raw_burst)           
    
    fig, axs  = plt.subplots(3, 1, figsize=(16,9),constrained_layout=True)
    fig1, axs1  = plt.subplots(1, 2, figsize=(16,9),constrained_layout=True)
    IDs = slc_burst_name.split('_')
    SLC_ID= f'{IDs[9]}_{IDs[11]}_{IDs[13]}'.split('.')[0]
    plot_fig_name = f'{SLC_ID}_{location}_SLC_RAW_&_RC_RAW'
    fig.suptitle(f'{plot_fig_name}')
    font_size =8
    
    img_name=['SLC', 'RAW', 'RC_RAW']
    plot_type=['log_normal', 'linear', 'log_normal']
    sar_data_to_plot=[intensity(slc_burst), np.real(slc_matched_raw_burst), 
                      intensity(rc_burst)
                      ]
    crop_size =[512,1024]
    r,c            = place_roi(slc_burst,  crop_size,location)
    slc_crop = slc_burst[r:r + crop_size[0], c:c + crop_size[1]]

    raw_crop = slc_matched_raw_burst[r:r + crop_size[0], 
                                     c:c + crop_size[1]
                                     ]
    rc_crop  = rc_burst[r:r + crop_size[0], c:c + crop_size[1]]
    
    crop_sar_data_to_plot = [intensity(slc_crop), np.real(raw_crop),
                             intensity(rc_crop)
                             ]
    
    az_sgn = -1 if location[0] < real_raw_burst.shape[0] / 2 else 1
    range_sgn = -1 if location[1] > real_raw_burst.shape[1] / 2 else 1
    
    for i in range(0,3):
    
        grid_colour='gray' if i==0 or i==2 else 'red'
        plot_2d_array_on_ax(axs[i], sar_data_to_plot[i],img_name[i],
                            plot_type[i],font_size, grid_on=1, 
                            grid_colour=grid_colour
                            )
        if i==0 or i==2:
            # Highlight ship:  Place circle around ship
            text_offset=[az_sgn*200,range_sgn*2000]
            #add_text_to_plot(axs[i], f'Burst based Object Labelled @ [{location[0]},'
            #             f'{location[1]}]',
            #             location,text_offset, colour='white'
            #             )
            #circle = patches.Circle((location1[1], location1[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
            #axs[i].add_patch(circle)
            # Add text for detected position of ship
            text_offset=[text_offset[0]*-1,text_offset[1]+100]
            add_text_to_plot(axs[i], f'Grid Based Object labelled @ [{location[0]},'
                         f'{location[1]}] Length ={int(length)}m [{gps_lat},{gps_long}]',
                         location,text_offset,
                         colour='white', arrow_colour='green'
                         )
            circle = patches.Circle((location[1],
                                     location[0]),
                                    radius=50, edgecolor='green',
                                    facecolor='none', linewidth=1
                                    )
            axs[i].add_patch(circle)
    
    
    
    for i in range(0,2):
    
        grid_colour='gray' if i==0  else 'red'
        plot_2d_array_on_ax(axs1[i], crop_sar_data_to_plot[i],img_name[i],
                            plot_type[i],font_size, grid_on=1, 
                            grid_colour=grid_colour
                            )
        if i==0:
            # Highlight ship:  Place circle around ship
            text_offset=[az_sgn*100,range_sgn*100]
            #add_text_to_plot(axs[i], f'Burst based Object Labelled @ [{location[0]},'
            #             f'{location[1]}]',
            #             location,text_offset, colour='white'
            #             )
            #circle = patches.Circle((location1[1], location1[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
            #axs[i].add_patch(circle)
            # Add text for detected position of ship
            text_offset=[text_offset[0]*-1,text_offset[1]+100]
            add_text_to_plot(axs1[i], f'Grid Based Object labelled @ [{location[0]},'
                         f'{location[1]}] Length ={int(length)}m [{gps_lat},{gps_long}]',
                         location,text_offset,
                         colour='white', arrow_colour='green'
                         )
            circle = patches.Circle((location[1],
                                     location[0]),
                                    radius=50, edgecolor='green',
                                    facecolor='none', linewidth=1
                                    )
            axs1[i].add_patch(circle)
    
    
    
    fig.savefig(f"{test_dir}/{burst_name.split('.')[0]}_{index}.jpg", format='jpg',
             bbox_inches='tight', dpi=600
             )
    
    fig1.savefig(f"{test_dir}/{burst_name.split('.')[0]}_zoom_{index}.jpg", format='jpg',
            bbox_inches='tight', dpi=600
            )
    
    plt.close(fig)
    plt.close(fig1)
    return 1

plt.style.use('dark_background')


#Set up directories

source = 'int'
Dataset = 'Train'
sar_mode = 'RAW'
swath_of_interest = 2

if source == 'ext':
    raw_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_data'
    slc_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_data'
    slc_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_burst_data'

    #Get ship locations in SLC file
    path = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}'
    #img_path=f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m'
    img_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results/range_compression_tests_Train_29Oct2025'
    dest_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/rc_raw_burst_data'


elif source == 'int':    
    raw_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/raw_data'
    slc_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/slc_data'
    #slc_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/slc_burst_data'
    #raw_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/raw_burst_data'
    slc_burst_dir =f'/home/joe/tmp/tmp_slc_data'
    raw_burst_dir =f'/home/joe/tmp/tmp_raw_data'
    
    
    #Get ship locations in SLC file
    path = f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}'
    img_path = f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_velocity_hypothesis/NonShip_gt50m_lt100m'

    
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

#Open csv file
csv_dir      =f'/home/joe/Work/Projects/RSAR/Test/SampleData/Train/Reports/'
csv_filename =f'SARFish_{sar_mode}_{Dataset}_IW{swath_of_interest}_BurstInfoShipsPerBurstWithNonShipLocations.csv'
df         = pd.read_csv(f'{csv_dir}{csv_filename}')

non_ship_size = 100
# Find all non-ships greater than 100m in length - save as list
df_gt_NonShipSize = df[df['ship_length']>non_ship_size]

print(f'Number of ships is {len(df_gt_NonShipSize)}')
