#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 16:01:48 2025

@author: joe
"""

import platform
import psutil
import os

from SAR_code_module import *
from SAR_focusing_module import *
from pypapi import events, papi_high
import time
import pickle


def list_computer_info():
    
    print("=== CPU Information ===")
    print("Processor:", platform.processor())
    print("CPU architecture:", platform.machine())
    print("CPU cores (physical):", psutil.cpu_count(logical=False))
    print("CPU cores (logical):", psutil.cpu_count(logical=True))
    
    if os.name == "posix":
        try:
            with open("/proc/cpuinfo") as f:
                lines = f.readlines()
                model_name = [line for line in lines if "model name" in line]
                if model_name:
                    print("CPU Model:", model_name[0].split(":")[1].strip())
        except Exception:
            pass
    
    print("\n=== RAM Information ===")
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    print(f"Total RAM: {total_gb:.2f} GB")


"""
# Test functions
raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data/f9fd0c2000d1a632p/RAW/S1B_IW_RAW__0SDV_20200421T054807_20200421T054840_021238_0284D5_1452.SAFE/"
raw_file_name='s1b-iw-raw-s-vh-20200421t054807-20200421t054840-021238-0284d5.dat'
raw_file=f'{raw_path}{raw_file_name}'
annotation_file='/home/joe/Misc/tempTest/Data/SLC_Data/S1B_IW_SLC_1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96/SLC/S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96.SAFE/annotation/s1b-iw2-slc-vh-20200421t054811-20200421t054837-021238-0284d5-002.xml'
slc_id = 'S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96'

burst_to_get =3
raw_burst_data = get_single_raw_burst_from_files(raw_file, annotation_file, burst_to_get, swath_of_interest=2, polarization='vh')
raw_burst = raw_burst_data[0]
raw_burst_metadata = raw_burst_data[1]
l0file  = raw_burst_data[2]



with open('data.pkl','wb') as f:
    pickle.dump(l0file, f)
    
with open('burst_matadata.pkl', 'wb') as f:
    pickle.dump(raw_burst_metadata,f)

with open('raw_burst.pkl', 'wb') as f:
    pickle.dump(raw_burst,f)
    
"""

slc_id = 'S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96'
burst_to_get = 3

with open('data.pkl','rb') as f:
    p_l0file = pickle.load(f)
    
with open('burst_matadata.pkl', 'rb') as f:
    p_burst_meta_data =pickle.load(f)

with open('raw_burst.pkl', 'rb') as f:
    p_raw_burst = pickle.load(f)
    



#Set Offsets to access sub-images
range_offset  = 0
range_length  = p_raw_burst.shape[1]-range_offset
azimuth_offset= 0
azimuth_length= p_raw_burst.shape[0]-azimuth_offset

roi=np.array([range_offset, range_length, azimuth_offset, azimuth_length])

# Do full focusing
rma='rma'



start_time = time.process_time()  # Start CPU time measurement (or time.time() for wall-clock)

# Call your function here
#result = your_function(data)
#radar_data = focus_SM_burst(raw_burst_metadata, raw_burst, roi, l0file, 'rma')
radar_data = focus_SM_burst(p_burst_meta_data, p_raw_burst, roi, p_l0file, 'rma')
#print(f'Hello')

cpu_time = time.process_time() - start_time  # Calculate CPU time
list_computer_info()



print(f"CPU time spent: {cpu_time:.4f} seconds")


"""
plot_raw_burst(radar_data, 'None',3, 'real')
plot_raw_burst(radar_data, 'None',3, 'imag')
plot_raw_burst(radar_data, 'None',3, 'amplitude')
"""
