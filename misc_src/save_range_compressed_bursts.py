#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 17:21:16 2025

@author: joe
"""

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
Dataset = 'Public'

if source == 'ext':
    raw_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_data'
    slc_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_data'
    slc_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_burst_data'
    raw_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_burst_data'

    #Get ship locations in SLC file
    path = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}'
    #img_path=f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m'
    #img_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results/range_compression_tests_16Oct2025'
    img_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results/range_compression_tests_Public_29Oct2025'
    dest_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/rc_raw_burst_data'


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


files_already_done = os.listdir(dest_dir)

burst_list = os.listdir(raw_burst_dir)
save_rc_data = 1

#read chip file
for file in burst_list:
    if '_real_' not in file:
        continue
    if file.replace('_real_','_rc_') in files_already_done:
        print(f'{file} already processed. Continuing to next file.')
        continue    
    real_file_path = f'{raw_burst_dir}/{file}'
    imag_file_path = real_file_path.replace('_real_','_imag_')
    if not os.path.isfile(real_file_path) or not os.path.isfile(imag_file_path):
        print(f'Error: File not found : Either  {real_file_path} or {imag_file_path}')
        continue
    try:
        raw_real_burst = np.load(f'{real_file_path}')
        raw_imag_burst = np.load(f'{imag_file_path}')
            
    except:
        print(f'File Error with {file}')
        continue  # exit the loop if any of the files are missing
    
    
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
    #rc_raw_burst = range_compress_burst(raw_burst)
    rc_padded_burst = range_compress_burst(padded_raw_burst)
    resized_rc_padded_burst=rc_padded_burst[
                                 az_padding:az_padding+raw_burst.shape[0],
                                 range_padding:range_padding+raw_burst.shape[1]
                                 ]
    del rc_padded_burst
    rc_padded_burst = resized_rc_padded_burst
    
    
    
    if save_rc_data:
        rc_raw_burst = file.replace('_real_','_rc_')
        np.save(f'{dest_dir}/{rc_raw_burst}', rc_padded_burst)
        
    
    
#plot Raw, SLC, rc_Raw, and rc_padded_Raw

#Label ships in SLC and rc_RAW and rc_padded_Raw

#?? Do blob detection on chips of 1024x1024 centered on expected ship position
# and then label with detected position
