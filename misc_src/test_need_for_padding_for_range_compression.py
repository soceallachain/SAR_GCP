#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:54:13 2025

@author: joe
"""


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
    img_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results/range_compression_tests_16Oct2025'


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


#Open csv file for HM ships gt100m and store as dataframe

#plt.ion()   
csv_path = f'{path}/Reports/'
#  Read CSV
csv_filename =f'{Dataset}_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m_Statistics_with_SLC_ID.csv'
test_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results'
#csv_filename =f'Bursts_Ships_To_Test_Offset.csv'
csv_results_file =f'{csv_filename.replace(".csv","_results.csv")}'
csv_file = os.path.join(test_dir, csv_filename)
df = pd.read_csv(csv_file)

#Parse dataframe to find ships within 100 of azimuth edges or 1000 of range edges
# Corrected code
df_edge_ships = df[(df['Ship Row'] < 100) | 
                   (df['Ship Row'] > 1200) | 
                   (df['Ship Col'] < 1000) | 
                   (df['Ship Col'] > 20000)]


#get list of bursts
burst_names    = df['Burst_Name'].to_list()
ship_locations = df[['Ship Row', 'Ship Col']].to_numpy()
ship_lengths   = df['Ship Length'].to_numpy()
cws_means      = df['CWS Mean'].to_numpy()
cws_stdevs     = df['CWS StdDev'].to_numpy()


#For each burst load real raw, imag raw and slc
for i in range(83,84):
#for i in range(len(ship_locations)):
    ship_location = ship_locations[i]
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
    #Create raw burst as real + i*imag
    raw_burst = raw_real_burst + 1j * raw_imag_burst
    #Pad raw burst in azimuth and  range
    range_padding = 1000
    az_padding    = 200
    padded_raw_burst = np.pad(raw_burst, ((az_padding,az_padding),
                              (range_padding,range_padding)),
                              mode='constant', constant_values=0
                              )

    #Range compress raw and padded_raw
    rc_raw_burst = range_compress_burst(raw_burst)
    rc_padded_burst = range_compress_burst(padded_raw_burst)
    resized_rc_padded_burst=rc_padded_burst[
                                 az_padding:az_padding+raw_burst.shape[0],
                                 range_padding:range_padding+raw_burst.shape[1]
                                 ]
    del rc_padded_burst
    rc_padded_burst = resized_rc_padded_burst
    
    plot_graphs = 0
    if plot_graphs: 
        # Create 2x2 figure grid
        fig, axs  = plt.subplots(4, 1, figsize=(16,9),constrained_layout=True)
        plot_fig_name = f'{crc_burst}_{ship_location}_SLC_RAW_&_RC_Focused'
        fig.suptitle(f'{plot_fig_name}')
        font_size =8
       
        img_name=['RAW','SLC', 'RC_RAW','RC_PADDED_RAW']
        plot_type=['linear', 'log_normal', 'log_normal', 'log_normal']
        sar_data_to_plot=[raw_real_burst, slc_intensity,  
                          intensity(rc_raw_burst), intensity(rc_padded_burst)
                          ]
       
           
        for ii in range(0,4):
            
            grid_colour='gray' if ii==0 or ii==2 else 'red'
            plot_2d_array_on_ax(axs[ii], sar_data_to_plot[ii],img_name[ii],
                                plot_type[ii],font_size, grid_on=1, grid_colour=grid_colour
                                )
            
        
        fig.savefig(f'{img_path}/{plot_fig_name}.jpg', format='jpg',
                 bbox_inches='tight', dpi=600
                 )
        fig.clf()
        plot.close('all')
    
    
    if save_rc_data:
        np.save(f'{dest_dir}')
        
    
    
#plot Raw, SLC, rc_Raw, and rc_padded_Raw

#Label ships in SLC and rc_RAW and rc_padded_Raw

#?? Do blob detection on chips of 1024x1024 centered on expected ship position
# and then label with detected position


