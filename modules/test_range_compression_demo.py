#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:55:39 2026

@author: joe
"""

import logging
import math
import cmath
import struct
import os
import io
import re
import sys
import glob
import random
import shutil
import shapely
import cv2
import pandas as pd
import numpy as np
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.stats import norm, gamma
import geopandas as gpd

from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from matplotlib.lines import Line2D




# Focus a burst or ROI with Range Compression Only
# Assign Sentinel 1 chirp parameters: Unique for each IW swath
# Can be obtained from decoded RAW data OR from SLC annotation file
# samplingFrequencyAfterDecimation, txPulseLength, txPulseStartFrequency, txPulseRampRate
S1 = np.zeros((3, 4)) 
S1[0,:]=[6.434523812571429e+07, 5.240481033595628e-05, -2.825153419637256e+07, 1.078230321255894e+12 ]
S1[1,:]=[5.459595962181818e+07, 6.199592966536363e-05, -2.415503791125222e+07, 7.792817275120481e+11]
S1[2,:]=[4.691840280000000e+07, 5.339056426703426e-05, -2.139431040091632e+07, 8.014509490705804e+11  ]

range_sampling_freq_index = 0
pulse_length_index        = 1
pulse_start_freq_index    = 2
pulse_ramp_rate_index     = 3



def range_compress_burst(rd:np.array, satellite='S1', iw_mode=2)->np.array: 
     
    #radar_data     = rd 
    range_padded_size    = 4096
    azimuth_padded_size  = 1296
    range_compressed_data = range_compress_burst_with_zero_padding(rd,
                                                  range_padded_size,
                                                  azimuth_padded_size,
                                                  satellite,
                                                  iw_mode
                                                  )
    return range_compressed_data



# Function to combine padding and range compression
# Returns range compressed data with padding removed
def range_compress_padded_raw_data(raw_data,satellite='S1', iw_mode=2,
                                   range_padding=1000,
                                   az_padding=200
                                   ):
    
    padded_raw_burst = np.pad(raw_data, ((az_padding,az_padding),
                               (range_padding,range_padding)),
                               mode='constant', constant_values=0
                               )
    rc_padded_burst = range_compress_burst(padded_raw_burst, satellite, 
                                           iw_mode
                                           )
    resized_rc_padded_burst=rc_padded_burst[
                    az_padding:az_padding+raw_data.shape[0],
                    range_padding:range_padding+raw_data.shape[1]
                    ]
    del rc_padded_burst
    rc_padded_burst = resized_rc_padded_burst
    return rc_padded_burst

def range_compress_burst_with_zero_padding(rd:np.array, rzp, azp, 
                                           satellite='S1', iw_mode=2
                                           )->np.array: 
     
    #radar_data     = rd 
    range_padded_size    = rzp
    azimuth_padded_size  = azp
    
    
    if satellite =='S1':
        range_sample_freq = S1[iw_mode-1, range_sampling_freq_index]
        TXPL              = S1[iw_mode-1, pulse_length_index]
        TXPSF             = S1[iw_mode-1, pulse_start_freq_index]
        TXPRR             = S1[iw_mode-1, pulse_ramp_rate_index]
        
    elif satellite =='ICEYE':
        range_sample_freq = 0
        TXPL              = 0
        TXPSF             = 0
        TXPRR             = 0

    else:
        print('Error: Satellite not known. Exiting ....')
        sys.exit()
    
    #Generate chirp replica pulse for convolution with matched filter
    num_tx_vals           = int(TXPL*range_sample_freq) 
    tx_replica_time_vals  = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals) 
    phi1                  = TXPSF + TXPRR*TXPL/2 
    phi2                  = TXPRR/2 
    tx_replica            = np.exp(2j * np.pi 
                                      * (phi1*tx_replica_time_vals 
                                         + phi2*tx_replica_time_vals**2
                                         )
                                   )     
 
    # Check if radar data needs padding to reduce risk of folding
    # NOTE: Check if azimuth padding causes timing issues!!!!!!
    # Ensure padded size is smaller than complete burst size which is:
    # total_burst_size = len(selection)
    az_num_zeroes   = 0
    az_num_zeroes_t = 0
    num_zeroes      = 0
    num_zeroes_r    = 0
    if rd.shape[1] < num_tx_vals or rd.shape[0] < azimuth_padded_size:
        if rd.shape[1] < num_tx_vals:
            num_zeroes   = int((range_padded_size - rd.shape[1])/2)
            num_zeroes_r = num_zeroes + (rd.shape[1] % 2)

        if rd.shape[0] < azimuth_padded_size:
            az_num_zeroes = int((azimuth_padded_size - rd.shape[0])/2)
            az_num_zeroes_t = az_num_zeroes + (rd.shape[0] % 2)


        radar_data = np.pad(rd, pad_width=((az_num_zeroes, az_num_zeroes_t),
                                          (num_zeroes,num_zeroes_r)
                                          ),
                           mode='constant', constant_values=0
                           )

    else:
        radar_data = rd
    
    # Image sizes
    len_range_line = radar_data.shape[1]
    len_az_line    = radar_data.shape[0]
    
    
    # Sample rates
    range_sample_period  = 1/range_sample_freq
    
    # Fast time vector - defines the time axis along the fast time direction
    sample_num_along_range_line = np.arange(0, len_range_line, 1)
    
    # Axes - defines the frequency axes in each direction after FFT
    SWL                  = len_range_line/range_sample_freq
    range_freq_vals      = np.arange(-range_sample_freq/2, range_sample_freq/2, 1/SWL)

     
    #print(range_sample_freq) 
     
    #Get posiiton and velocity vectors 
     
    
    #FFT in range direction 
    radar_data = np.fft.fft(radar_data, axis=1) 
     
    # FFT each azimuth line - is this needed for range compression?
    radar_data = np.fft.fftshift(np.fft.fft(radar_data, axis=0), axes=0) 

    # Create range filter from replica pulse 
    range_filter         = np.zeros(len_range_line, dtype=complex) 
    index_start          = np.ceil((len_range_line-num_tx_vals)/2)-1 
    index_end            = num_tx_vals+np.ceil((len_range_line-num_tx_vals)/2)-2 
    range_filter[int(index_start):int(index_end+1)] = tx_replica 
    range_filter         = np.conjugate(np.fft.fft(range_filter)) 
     
    # Apply filter 
    radar_data           = np.multiply(radar_data, range_filter) 
     
    #print(f'TXPL, (Pulse Length) = {TXPL} ')
    #print(f' Number of  Pulse samples   = {num_tx_vals}')
    del range_filter 
    del tx_replica 

    radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1)
    radar_data = np.fft.ifft(radar_data, axis=0) 

    return radar_data[az_num_zeroes:az_num_zeroes + rd.shape[0], 
                      num_zeroes:num_zeroes + rd.shape[1]
                      ] 



# Get SLC and RAW files
test_dir   = f'Houston_SM_No_OCN'
root_path  = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'

pol1        = 'hv'
pol2        = 'vh'
pol         = [pol1, pol2]
SLC_IDs     = os.listdir(root_path)
#swaths      = ['_S1_', '_S3_', '_S6_']
swaths      =['_S3_']

