

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
from modules.SAR_focusing_module import *
from scipy.ndimage import zoom
#filepath = "../data/sao_paulo/"
#filename = "s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
import mpld3


import psutil



def print_mem_usage():
    # Get virtual memory details
    mem = psutil.virtual_memory()

    # Convert bytes to Gigabytes for readability
    gb = 1024**3

    print(f"Total RAM:     {mem.total / gb:.2f} GB")
    print(f"Used RAM:      {mem.used / gb:.2f} GB")
    print(f"Available RAM: {mem.available / gb:.2f} GB")  # RAM available without swapping
    print(f"RAM Usage:     {mem.percent}%")
    return



use_GCP = True

if use_GCP:

    
    #root_path   = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN/'
    root_path    = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_OCN/'
    #SLC_ID      = f'S1C_S3_SLC__1SDH_20250520T001736_20250520T001802_002407_0050D2_838C'
    #SLC_ID      = f'S1C_S3_SLC__1SDH_20250601T001738_20250601T001804_002582_0055B8_7A92'
    SLC_ID       = f'S1C_S3_SLC__1SDH_20250613T001738_20250613T001805_002757_005ACE_F1C4'
    
    #raw_filepath = f'{root_path}{SLC_ID}/RAW/S1C_S3_RAW__0SDH_20250520T001735_20250520T001803_002407_0050D2_7ECF.SAFE/'
    #raw_filepath = f'{root_path}{SLC_ID}/RAW/S1C_S3_RAW__0SDH_20250601T001737_20250601T001805_002582_0055B8_75F6.SAFE/'
    raw_filepath = f'{root_path}{SLC_ID}/RAW/S1C_S3_RAW__0SDH_20250613T001738_20250613T001805_002757_005ACE_E85D.SAFE/'
    

    
    
    #root_path    = f'/mnt/GCP_ubot_sar_data/data/Houston_SM/Houston_SM_No_OCN/'
    #SLC_ID       = f'S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A'
    #SLC_ID       = f'S1B_S3_SLC__1SDH_20180109T001740_20180109T001803_009087_010402_E6D9'
    #SLC_ID       = f'S1B_S3_SLC__1SDH_20180214T001739_20180214T001802_009612_011540_E777'
    #SLC_ID       = f'S1B_S3_SLC__1SDH_20180226T001739_20180226T001802_009787_011B00_D63C'
    
    #raw_filepath = f'{root_path}{SLC_ID}/RAW/S1B_S3_RAW__0SDH_20180801T001745_20180801T001811_012062_01635C_9872.SAFE/'
    #raw_filepath = f'{root_path}{SLC_ID}/RAW/S1B_S3_RAW__0SDH_20180109T001739_20180109T001804_009087_010402_196D.SAFE/'
    #raw_filepath = f'{root_path}{SLC_ID}/RAW/S1B_S3_RAW__0SDH_20180214T001738_20180214T001804_009612_011540_0081.SAFE/'
    #raw_filepath = f'{root_path}{SLC_ID}/RAW/S1B_S3_RAW__0SDH_20180226T001738_20180226T001803_009787_011B00_5F63.SAFE/'
    
    slc_filepath = f'{root_path}{SLC_ID}/SLC/S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A.SAFE/measurement/'
    annot_filepath = f'{root_path}{SLC_ID}/SLC/S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A.SAFE/annotation/'

else:
    raw_filepath   = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1C_S6_RAW__0SDH_20260601T002551_20260601T002618_007905_010136_80A6.SAFE/"
    #raw_filepath   = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_RAW__0SDH_20180801T001745_20180801T001811_012062_01635C_9872.SAFE/"
    annot_filepath = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A.SAFE/annotation/"
    slc_filepath   = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A.SAFE/measurement/"




#raw_filename    = 's1c-s3-raw-s-hv-20250520t001735-20250520t001803-002407-0050d2.dat'
#raw_filename   = 's1c-s3-raw-s-hv-20250601t001737-20250601t001805-002582-0055b8.dat'
raw_filename   = 's1c-s3-raw-s-hv-20250613t001738-20250613t001805-002757-005ace.dat'

#raw_filename    = "s1c-s6-raw-s-hh-20260601t002551-20260601t002618-007905-010136.dat"
#raw_filename    = "s1b-s3-raw-s-hv-20180801t001745-20180801t001811-012062-01635c.dat"
#raw_filename    = 's1b-s3-raw-s-hv-20180109t001739-20180109t001804-009087-010402.dat'
#raw_filename    = 's1b-s3-raw-s-hv-20180214t001738-20180214t001804-009612-011540.dat'
#raw_filename    = 's1b-s3-raw-s-hv-20180226t001738-20180226t001803-009787-011b00.dat'

slc_filename    = 's1b-s3-slc-hv-20180801t001746-20180801t001810-012062-01635c-002.tiff'
annot_filename  = "s1b-s3-slc-hv-20180801t001746-20180801t001810-012062-01635c-002.xml"
raw_file        = raw_filepath + raw_filename
slc_file        = slc_filepath + slc_filename
annotation_file = annot_filepath + annot_filename


print(datetime.now())
print(f' executing get_SM_raw_burst_info')
raw_burst_info, l0file = get_SM_raw_burst_info_v2(raw_file)
print(datetime.now())
    
print_mem_usage()
selected_burst = raw_burst_info[0].index[0]
meta_data      = l0file.get_burst_metadata(selected_burst)
print(f' **********  assigned meta data *********')
print_mem_usage()
#exit(0)
radar_data     = l0file.get_burst_data(selected_burst)
        

print(datetime.now())

print_mem_usage()
print(f' Radar data assigned   -----')
