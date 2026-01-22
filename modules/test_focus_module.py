
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




# Focus a burst with Strip Mode RDA algorithm
# Can chose to do only range compression ("rma == 'r'"), both it and RCMC ("rma=='rm'") or 
# full focusing ("rma=='rma'") _ useful for checking flop count! 
def focus_SM_burst(burst_number: pd.DataFrame, rd:np.array,
                   region_of_interest:np.array, _l0file, rma)->np.array: 
     
    l0file         = _l0file 
    selection      = burst_number 
    radar_data     = rd 
     
    range_offset   = region_of_interest[0] 
    range_length   = region_of_interest[1] 
    azimuth_offset = region_of_interest[2] 
    azimuth_length = region_of_interest[3] 
 
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
    fast_time_vec = ( range_start_time + (range_sample_period * range_offset) 
                      + (range_sample_period * sample_num_along_range_line)
                    )
    
    # Slant range vector - defines R0, the range of closest approach, for each range cell
    slant_range_vec      = ((rank * PRI) + fast_time_vec) * c/2
        
    # Axes - defines the frequency axes in each direction after FFT
    SWL                  = len_range_line/range_sample_freq
    #az_freq_vals         = np.arange(-az_sample_freq/2, az_sample_freq/2, 1/(PRI*len_az_line))
    az_freq_vals         = np.linspace(-az_sample_freq/2, az_sample_freq/2, len_az_line)
    range_freq_vals      = np.arange(-range_sample_freq/2, range_sample_freq/2, 1/SWL)

     
    print(range_sample_freq) 
     
    #Get posiiton and velocity vectors 
    ecef_vels = l0file.ephemeris.apply(lambda x: math.sqrt(x["X-axis velocity ECEF"]**2 
                                                 + x["Y-axis velocity ECEF"]**2 
                                                 + x["Z-axis velocity ECEF"]**2), axis=1) 

    velocity_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), 
                                                ecef_vels.unique(), fill_value="extrapolate") 
    x_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), 
                                          l0file.ephemeris["X-axis position ECEF"].unique(), 
                                          fill_value="extrapolate") 
    y_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), 
                                          l0file.ephemeris["Y-axis position ECEF"].unique(), 
                                          fill_value="extrapolate") 
    z_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), 
                                          l0file.ephemeris["Z-axis position ECEF"].unique(), 
                                          fill_value="extrapolate") 
     
    space_velocities = selection.apply(lambda x: velocity_interp(x["Coarse Time"] 
                                            + x["Fine Time"]), axis=1
                                      ).to_numpy().astype(float) 
     
    x_positions = selection.apply(lambda x: x_interp(x["Coarse Time"] 
                                  + x["Fine Time"]), axis=1
                                 ).to_numpy().astype(float) 
    y_positions = selection.apply(lambda x: y_interp(x["Coarse Time"] 
                                  + x["Fine Time"]), axis=1
                                 ).to_numpy().astype(float) 
    z_positions = selection.apply(lambda x: z_interp(x["Coarse Time"] 
                                  + x["Fine Time"]), axis=1
                                 ).to_numpy().astype(float) 
     
    # Only want positions for azimuth range being viewed 
    space_vels  = space_velocities[azimuth_offset:azimuth_offset + azimuth_length] 
    del space_velocities 
    space_velocities = space_vels 
    x_pos      = x_positions[azimuth_offset:azimuth_offset + azimuth_length] 
    y_pos      = y_positions[azimuth_offset:azimuth_offset + azimuth_length] 
    z_pos      = z_positions[azimuth_offset:azimuth_offset + azimuth_length] 
     
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
     
    cos_beta = (np.divide(np.square(local_earth_rad) + np.square(H) 
                - np.square(slant_range_vec[:, np.newaxis]
                           ), 2 * local_earth_rad * H)
               ) 
     
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
    tx_replica            = np.exp(2j * np.pi * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2)) 
     
    # Create range filter from replica pulse 
    range_filter         = np.zeros(len_range_line, dtype=complex) 
    index_start          = np.ceil((len_range_line-num_tx_vals)/2)-1 
    index_end            = num_tx_vals+np.ceil((len_range_line-num_tx_vals)/2)-2 
    range_filter[int(index_start):int(index_end+1)] = tx_replica 
    #range_filter        = tx_replica[int ((tx_replica.size - 1)/2 - len_range_line/2) : int( (tx_replica.size -1)/2  + len_range_line/2)] 
    range_filter         = np.conjugate(np.fft.fft(range_filter)) 
     
    # Apply filter 
    radar_data           = np.multiply(radar_data, range_filter) 
    
    
    print(f'TXPL: - pulse length = {TXPL} ')
    print(f'NUmber of pulse samples = {num_tx_vals}')
    del range_filter 
    del tx_replica 
    


    # If only doing range compression
    if rma =='r':
        print(f' rma = r - Exiting')
        radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1)
        return radar_data
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
        return radar_data
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
    return radar_data 

