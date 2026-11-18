#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:51:42 2026

@author: joe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:37:46 2026

@author: joe
"""

import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d


from modules.SAR_code_module import *
#from modules.chunk_processing_module import *
from modules.SAR_focusing_module import *
from modules.image_processing_module import *
from scipy.ndimage import zoom

#Set DEBUG mode if do not want to save files
DEBUG = True
# Get list of SLC_IDs

#test_dir   = f'Houston_SM_2019_20'
test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/jc_data/{test_dir}/'
#root_path  = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'
#test_dir   = f'SaoPaulo_SM_2019_20'
#root_path  = f'/mnt/SAR_bucket/data/SaoPaulo_SM/{test_dir}/'

#test_dir   = f'S1A_D_S1_S6_Samples'
root_path     = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'
#save_path     = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN_burst_data/'

# Vm cannot write to GCP for some reason
save_path     = f'/home/joe/SAR_GCP/tests/slc_raw_matching/'
raw_burst_dir = f'raw_burst'
slc_burst_dir = f'slc_burst'
rc_burst_dir  = f'rc_burst'

pol1       = 'hv'
pol2       = 'vh'
pol3       = 'vv'
pol4       = 'hh'
pols       = [pol1, pol2, pol3, pol4]
SLC_IDs    = os.listdir(root_path)
swaths     = ['_S2_', '_S4_','_S5_','_S6_','_S3_']


# Define how big a region to search to cross correlate over for matching
slice_size = [5000, 5000] 

plot_plots = False
# Test if FFT in azimuth needed for Range COmpression

#Load raw file
#slc_ids= SLC_IDs[495:505]  # S3 files
#slc_ids= SLC_IDs[299:306]  # S6 files
#slc_ids= SLC_IDs[603:623]  # Both S3 abnd S6
slc_ids= SLC_IDs[0:9]  # S1 files


dd=[]
for slc_id in slc_ids:
    slc_id_path = root_path + slc_id +'/'
    for pol in pols:  
       files       = get_SM_files_from_slc_id(slc_id_path, pol)
       if files != -1:
           break  # Found files
    if files == -1:
        print(f'Warning: {slc_id}:  Files not found ')      
    else:
        # Get block raw data and metadata, and time-based-offsets
        scene_files = files
        chunk_info  = process_file_for_chunks(scene_files)
        meta_data= chunk_info[1]
        for chunk_number, chunk in enumerate(meta_data):
            RGDEC = chunk["Range Decimation"].unique()[0]
            range_sample_freq    = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
            print(range_sample_freq)
            tx_pulse_info = chunk[['Tx Pulse Length', 
                                  'Tx Ramp Rate', 
                                  'Tx Pulse Start Frequency'
                                  ]]
            tx_info = tx_pulse_info.iloc[0]
            tx_info['Range Sampling Freq']=range_sample_freq
            tx_info['SLC_ID']=slc_id
            swath_number =  int(chunk['Swath Number'].unique()[0].astype(int))
            if swath_number !=6 and swath_number!=5:
                tx_info['Swath']= f'S{int(swath_number + 1)}'
            else:
                tx_info['Swath']= f'S{int(swath_number)}'
    
            tx_info['Chunk Number'] = chunk_number
            dd.append(tx_info)

tx_df = pd.DataFrame(dd)


            
            
            
            
        