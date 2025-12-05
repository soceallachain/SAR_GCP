
from lxml import etree
# system modules
import sentinel1decoder

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
# data manipulation
import shapely
import cv2
import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.measure as skm
from skimage.morphology import erosion, dilation
from skimage import exposure
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import colors
from datetime import datetime, timedelta
from shapely import LineString, Point, Polygon
from scipy.interpolate import interp1d
from scipy.stats import norm, gamma
#from shapely.geometry import Polygon, Point
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings
from blob_module import *

from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from matplotlib.lines import Line2D




def place_roi(A: np.ndarray, roi_shape: tuple, p: tuple):
    """
    Place a region of interest (ROI) of given shape centrally at location p in 2D numpy array A.
    If ROI extends beyond array edges, position it to include the edge fully.

    Args:
        A (np.ndarray): 2D numpy array
        roi_shape (tuple): (num_cols, num_rows) size of region of interest
        p (tuple): (row, col) coordinate where ROI is to be centered

    Returns:
        starting_row (int): starting row index of ROI within A
        starting_col (int): starting column index of ROI within A
        new_centre        : location in crop of where desired point ends up
    """

    num_rows, num_cols = roi_shape  # note: roi_shape is (num_cols, num_rows) as per your problem
    total_rows, total_cols = A.shape

    row_c, col_c = p  # The location p is (row, col)

    # Calculate half sizes (use integer division)
    half_rows = num_rows // 2
    half_cols = num_cols // 2

    # Starting row and col with +1 as in your example
    starting_row = row_c - half_rows + 1
    starting_col = col_c - half_cols + 1
    new_centre_r = half_rows
    new_centre_c = half_cols

    # Correct start if ROI extends beyond top/left edges
    if starting_row < 0:
        starting_row = 0
        new_centre_r = row_c
    if starting_col < 0:
        starting_col = 0
        new_centre_c = col_c

    # Correct start if ROI extends beyond bottom/right edges
    if starting_row + num_rows > total_rows:
        starting_row = total_rows - num_rows
        new_centre_r = row_c - starting_row
    if starting_col + num_cols > total_cols:
        starting_col = total_cols - num_cols
        new_centre_c = col_c - starting_col

    # Ensure no negative indices after correction (in case ROI bigger than array)
    starting_row = max(0, starting_row)
    starting_col = max(0, starting_col)

    return int(starting_row), int(starting_col), [new_centre_r, new_centre_c]



def focus_SM_burst(burst_number: pd.DataFrame, rd:np.array,
                   region_of_interest:np.array, _l0file, rma, 
                   )->np.array: 


    range_padded_size    = 4096
    azimuth_padded_size  = 1296
    focused_data         = focus_SM_burst_with_zero_padding(burst_number, rd,
                                       region_of_interest, _l0file, rma,
                                       range_padded_size, azimuth_padded_size
                                       )
    return focused_data



# Focus a burst with Strip Mode RDA algorithm
# Can chose to do only range compression ("rma == 'r'"), both it and RCMC ("rma=='rm'") or 
# full focusing ("rma=='rma'") _ useful for checking flop count! 
# NOTE: <>_padded_size is size after padding!
def focus_SM_burst_with_zero_padding(burst_number: pd.DataFrame, rd:np.array,
                                     region_of_interest:np.array, _l0file, rma, 
                                     range_padded_size, azimuth_padded_size
                                     )->np.array: 
     
    l0file         = _l0file 
    selection      = burst_number 
    #radar_data     = rd 
    #range_padded_size    = 4096
    #azimuth_padded_size  = 1296
     
    range_offset   = region_of_interest[0] #offset into matched raw burst :range_offset + col
    range_length   = region_of_interest[1] 
    azimuth_offset = region_of_interest[2] 
    azimuth_length = region_of_interest[3] 
 
    # Need to check if size of data in range is greater than range filter
    # based on pulse width and sampling. If not need to symmetrically pad with
    # zeroes in range direction
    RGDEC                = selection["Range Decimation"].unique()[0]
    range_sample_freq    = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    TXPL                  = selection["Tx Pulse Length"].unique()[0] 
    num_tx_vals           = int(TXPL*range_sample_freq) 
    
    # Check if radar data needs padding to reduce risk of folding
    # NOTE: Check if azimuth padding causes timing issues!!!!!!
    # Ensure padded size is smaller than complete burst size which is:
    # total_burst_size = len(selection)
    az_num_zeroes   = 0
    az_num_zeroes_t = 0
    num_zeroes      = 0
    num_zeroes_r    = 0
    total_azimuth_lines = len(selection)
    total_range_samples = int(selection['Number of Quads'].unique())*2
    if (rd.shape[1] < num_tx_vals and range_padded_size !=0) or rd.shape[0] < azimuth_padded_size:
        if rd.shape[1] < num_tx_vals:
            #int effectively floors 
            # Note: range_offset gives index into raw burst of first azimuth
            #       line of roi 
            # If padded roi extends past valid range values does not matter as 
            # those values are zero and are not using any positional data, 
            # unlike processing in azimuth when using positions 
            num_zeroes   = min(int((range_padded_size - rd.shape[1])/2), 
                                range_offset
                               )
            num_zeroes_r = min(num_zeroes + (rd.shape[1] % 2),
                               total_range_samples-rd.shape[1] - num_zeroes
                               )
            """
            num_zeroes = int((range_padded_size - rd.shape[1])/2)
            #Check if can pad by num_zeroes, if not set to max can pad by
            #while retaining timing information
            if num_zeroes > range_offset:
                num_zeroes = range_offset
            # Check if range length is even, if not inc num_zeroes_r
            num_zeroes_r = num_zeroes + (rd.shape[1] % 2)
            if num_zeroes + rd.shape[1] + num_zeroes_r > total_range_samples:
                num_zeroes_r = total_range_samples-rd.shape[1] - num_zeroes
            # Place rd data in middle of padded array
            """
            
           
            
            
        if rd.shape[0] < azimuth_padded_size:
            
            az_num_zeroes = min(int((azimuth_padded_size - rd.shape[0])/2), 
                                azimuth_offset
                                )
            space_at_end = total_azimuth_lines - (azimuth_offset + rd.shape[0])
            az_num_zeroes_t = min(az_num_zeroes + (rd.shape[0] % 2),
                                  space_at_end)
            """
            az_num_zeroes = int((azimuth_padded_size - rd.shape[0])/2)
            #Check if can pad by az_num_zeroes, if not set to max can pad by
            #while retaining timing information
            if az_num_zeroes > azimuth_offset:
                az_num_zeroes = azimuth_offset
            # Check if azimuth length is even, if not inc num_zeroes_r
            az_num_zeroes_t = az_num_zeroes + (rd.shape[0] % 2)
            if az_num_zeroes + rd.shape[0] + az_num_zeroes_t > total_azimuth_lines:
                az_num_zeroes_t = total_azimuth_lines - rd.shape[0] - az_num_zeroes
            # Place rd data in middle of padded array
            """
        
        #Note - Remove this after testing!!!!!!    
        #num_zeroes = range_padded_size
        #num_zeroes_r = num_zeroes
        radar_data = np.pad(rd, pad_width=((az_num_zeroes, az_num_zeroes_t),
                                           (num_zeroes,num_zeroes_r)
                                           ), 
                            mode='constant', constant_values=0
                            )    
    else:
        radar_data = rd
    
    P_PRI=selection["PRI"] 
    print(P_PRI.shape) 
    print(np.max(P_PRI)) 
    print(np.min(P_PRI)) 
    S_SWST= selection["SWST"] 
    print(S_SWST.shape) 
    print(np.max(S_SWST)) 
    print(np.min(S_SWST)) 
    T_TXPL = selection["Tx Pulse Length"] 
    print(T_TXPL.shape) 
    print(np.max(T_TXPL)) 
    print(np.min(T_TXPL))                                                                                              
     
    # Image sizes
    len_range_line = radar_data.shape[1]
    len_az_line    = radar_data.shape[0]
    
    # Tx pulse parameters
    c                    = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
    RGDEC                = selection["Range Decimation"].unique()[0]
    PRI                  = selection["PRI"].unique()[0]
    rank                 = selection["Rank"].unique()[0]
    suppressed_data_time = 320/(8*sentinel1decoder.constants.F_REF)
    range_start_time     = selection["SWST"].unique()[0] + suppressed_data_time
    wavelength           = sentinel1decoder.constants.TX_WAVELENGTH_M
    
    # Sample rates
    range_sample_freq    = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    range_sample_period  = 1/range_sample_freq
    az_sample_freq       = 1 / PRI
    az_sample_period     = PRI
    
    # Fast time vector - defines the time axis along the fast time direction
    sample_num_along_range_line = np.arange(0, len_range_line, 1)
    #Add offset time due to range_offset
    #fast_time_vec = range_start_time + (range_sample_period * sample_num_along_range_line)
    fast_time_vec = range_start_time + (range_sample_period * range_offset) + (range_sample_period * sample_num_along_range_line)
    
    # Slant range vector - defines R0, the range of closest approach, for each range cell
    slant_range_vec      = ((rank * PRI) + fast_time_vec) * c/2
        
    # Axes - defines the frequency axes in each direction after FFT
    SWL                  = len_range_line/range_sample_freq
    #az_freq_vals         = np.arange(-az_sample_freq/2, az_sample_freq/2, 1/(PRI*len_az_line))
    az_freq_vals         = np.linspace(-az_sample_freq/2, az_sample_freq/2, len_az_line)
    range_freq_vals      = np.arange(-range_sample_freq/2, range_sample_freq/2, 1/SWL)

     
    print(range_sample_freq) 
     
    #Get posiiton and velocity vectors 
    ecef_vels = l0file.ephemeris.apply(lambda x: math.sqrt(x["X-axis velocity ECEF"]**2 + x["Y-axis velocity ECEF"]**2 +x["Z-axis velocity ECEF"]**2), axis=1) 
    velocity_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), ecef_vels.unique(), fill_value="extrapolate") 
    x_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["X-axis position ECEF"].unique(), fill_value="extrapolate") 
    y_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["Y-axis position ECEF"].unique(), fill_value="extrapolate") 
    z_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["Z-axis position ECEF"].unique(), fill_value="extrapolate") 
     
    space_velocities = selection.apply(lambda x: velocity_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float) 
     
    x_positions = selection.apply(lambda x: x_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float) 
    y_positions = selection.apply(lambda x: y_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float) 
    z_positions = selection.apply(lambda x: z_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float) 
     
    # Only want positions for azimuth range being viewed 
    #space_vels  = space_velocities[azimuth_offset:azimuth_offset + azimuth_length] 
    """
    space_vels  = space_velocities[azimuth_offset - az_num_zeroes:
                                    azimuth_offset - az_num_zeroes  
                                   + azimuth_length + az_num_zeroes_t
                                   ] 
    """    
    space_vels  = space_velocities[azimuth_offset - az_num_zeroes:
                                    azimuth_offset + rd.shape[0]  
                                    + az_num_zeroes_t
                                   ] 
        
        
    del space_velocities 
    space_velocities = space_vels 
    #x_pos      = x_positions[azimuth_offset:azimuth_offset + azimuth_length] 
    #y_pos      = y_positions[azimuth_offset:azimuth_offset + azimuth_length] 
    #z_pos      = z_positions[azimuth_offset:azimuth_offset + azimuth_length] 
    x_pos = x_positions[azimuth_offset - az_num_zeroes:
                        azimuth_offset + rd.shape[0]  + az_num_zeroes_t
                        ]
    y_pos = y_positions[azimuth_offset - az_num_zeroes:
                        azimuth_offset + rd.shape[0]  + az_num_zeroes_t
                        ]
    z_pos = z_positions[azimuth_offset - az_num_zeroes:
                        azimuth_offset + rd.shape[0]  + az_num_zeroes_t
                        ]
     
    #position_array = np.transpose(np.vstack((x_positions, y_positions, z_positions))) 
    position_array  = np.transpose(np.vstack((x_pos, y_pos, z_pos))) 
     
     
    space_velocities.shape 
     
    a = sentinel1decoder.constants.WGS84_SEMI_MAJOR_AXIS_M 
    b = sentinel1decoder.constants.WGS84_SEMI_MINOR_AXIS_M 
    H = np.linalg.norm(position_array, axis=1) 
    W = np.divide(space_velocities, H) 
    lat = np.arctan(np.divide(position_array[:, 2], position_array[:, 0])) 
    local_earth_rad = np.sqrt( 
        np.divide( 
            (np.square(a**2 * np.cos(lat)) + np.square(b**2 * np.sin(lat))), 
            (np.square(a * np.cos(lat)) + np.square(b * np.sin(lat))) 
        ) 
    ) 
     
    local_earth_rad.shape 
     
    cos_beta = (np.divide(np.square(local_earth_rad) + np.square(H) - np.square(slant_range_vec[:, np.newaxis]) , 2 * local_earth_rad * H)) 
     
    ground_velocities    = local_earth_rad * W * cos_beta 
    effective_velocities = np.sqrt(space_velocities * ground_velocities) 
     
    del ground_velocities 
    del cos_beta 
    del local_earth_rad 
    del H 
    del W 
    del lat 
     
    D = np.sqrt( 
        1 - np.divide( 
            wavelength**2 * np.square(az_freq_vals), 
            4 * np.square(effective_velocities) 
        ) 
    ).T 
     
    radar_data = np.fft.fft(radar_data, axis=1) 
     
    # FFT each azimuth line 
    radar_data = np.fft.fftshift(np.fft.fft(radar_data, axis=0), axes=0) 
     
    # Create replica pulse 
    TXPSF                 = selection["Tx Pulse Start Frequency"].unique()[0] 
    TXPRR                 = selection["Tx Ramp Rate"].unique()[0] 
    TXPL                  = selection["Tx Pulse Length"].unique()[0] 
    num_tx_vals           = int(TXPL*range_sample_freq) 
    tx_replica_time_vals  = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals) 
    phi1                  = TXPSF + TXPRR*TXPL/2 
    phi2                  = TXPRR/2 
    tx_replica            = np.exp(2j * np.pi * (phi1*tx_replica_time_vals 
                                                + phi2*tx_replica_time_vals**2)
                                   ) 
     
    # Create range filter from replica pulse 
    range_filter         = np.zeros(len_range_line, dtype=complex) 
   
    # Need to deal with case where no zero padding and size of range data
    # is smaller than extent of pulselength at the sampling frequency
    # Position range filter at mid-point of pulse
    if len_range_line < num_tx_vals:
        range_filter = tx_replica[int ((tx_replica.size - 1)/2 - 
                                       len_range_line/2) : 
                                  int( (tx_replica.size -1)/2  
                                      + len_range_line/2
                                      )
                                  ]
    else:
        index_start          = np.ceil((len_range_line-num_tx_vals)/2)-1 
        index_end            = num_tx_vals+np.ceil((len_range_line-num_tx_vals)/2)-2 
        range_filter[int(index_start):int(index_end+1)] = tx_replica
        
                
    range_filter         = np.conjugate(np.fft.fft(range_filter)) 
     
    # Apply filter 
    radar_data           = np.multiply(radar_data, range_filter) 
     
    print(f'TXPL, (Pulse Length) = {TXPL} ')
    print(f' Number of  Pulse samples   = {num_tx_vals}')
    del range_filter 
    del tx_replica 
    

    # If only doing range compression
    if rma =='r':
        print(f' rma = r - Exiting')
        radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1)
        radar_data = np.fft.ifft(radar_data, axis=0) 
        return radar_data[az_num_zeroes:az_num_zeroes + rd.shape[0], 
                          num_zeroes:num_zeroes + rd.shape[1]
                          ] 
        sys.exit(0)
    # Create RCMC filter 
    range_freq_vals    = np.linspace(-range_sample_freq/2, range_sample_freq/2, num=len_range_line) 
    rcmc_shift         = slant_range_vec[0] * (np.divide(1, D) - 1) 
    rcmc_filter        = np.exp(4j * np.pi * range_freq_vals * rcmc_shift / c) 
   
    # Apply filter 
    radar_data        = np.multiply(radar_data, rcmc_filter) 
 
    del rcmc_shift 
    del rcmc_filter 
    del range_freq_vals 
     
    # If doing range compression and RCMC filtering
    if rma =='rm':
        print(f'rma = rm .... Exiting. ')
        radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1) 
        return radar_data[az_num_zeroes:az_num_zeroes + rd.shape[0], 
                          num_zeroes:num_zeroes + rd.shape[1]
                          ] 
        sys.exit(0)
    
    # Else - if doing full focusing 
    # Create Azimuth filter 
    print(f' rma = rma - Doing full focusing.')
    radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1) 
    az_filter = np.exp(4j * np.pi * slant_range_vec * D / wavelength) 
    # Apply filter 
    radar_data = np.multiply(radar_data, az_filter) 
    del az_filter 
     
    radar_data = np.fft.ifft(radar_data, axis=0) 
    return radar_data[az_num_zeroes:az_num_zeroes + rd.shape[0], 
                      num_zeroes:num_zeroes + rd.shape[1]
                      ] 






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

