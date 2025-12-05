###############################################################################
# test_Burst_Focus_IW_Burst.py
# Focuses 1 burst in given file.
###############################################################################


import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d





#  Function to read in a burst and focus it
#  Assumes is IW data so not too large
def focus_burst(burst_number: pd.DataFrame, rd:np.array,region_of_interest:np.array)->np.array:
    
    
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
    
    del range_filter
    del tx_replica
    
    # Create RCMC filter
    range_freq_vals    = np.linspace(-range_sample_freq/2, range_sample_freq/2, num=len_range_line)
    rcmc_shift         = slant_range_vec[0] * (np.divide(1, D) - 1)
    rcmc_filter        = np.exp(4j * np.pi * range_freq_vals * rcmc_shift / c)
    
    # Apply filter
    radar_data        = np.multiply(radar_data, rcmc_filter)
    
    del rcmc_shift
    del rcmc_filter
    del range_freq_vals
    
    
    radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1)
    
    # Create filter
    az_filter = np.exp(4j * np.pi * slant_range_vec * D / wavelength)
    # Apply filter
    radar_data = np.multiply(radar_data, az_filter)
    del az_filter
    
    radar_data = np.fft.ifft(radar_data, axis=0)
    return radar_data
    
# End function focus_SM_burst


###############################################################################################################


filepath="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/Dublin/RAW/S1A_IW_RAW__0SDV_20240107T063924_20240107T063956_051997_06488E_5DA4.SAFE/" 

filename="s1a-iw-raw-s-vv-20240107t063924-20240107t063956-051997-06488e.dat"


#Start here
#filepath = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/Dublin/SLC/S1A_IW_SLC__1SDV_20240105T180705_20240105T180733_051975_0647CF_89FB.SAFE/measurement/s1a-iw1-slc-vh-20240105t180706-20240105t180731-051975-0647cf-001.tiff

    
#filepath="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S3_RAW_0SDV_20230518T213602_20230518T213627_048593_05D835_4B93/RAW/S1A_S3_RAW__0SDV_20230518T213602_20230518T213627_048593_05D835_F012.SAFE/"
#filepath="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S3_RAW_0SDV_20230518T213602_20230518T213627_048953_05D835_4B93/RAW/S1A_S3_RAW__0SDH_20230518T213602_20230518T213627_048593_05D835_F012.SAFE/"
 
#filename="s1a-s6-raw-s-vh-20210226t214354-20210226t214424-036766-045287.dat"
#filename="s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
 
 
inputfile = filepath+filename
if os.path.exists(inputfile):
   print(f"{inputfile} found")
else:
   print(f"{inputfile} not found")

l0file   = sentinel1decoder.Level0File(inputfile)
 
#l0file.packet_metadata
#l0file.ephemeris



#### Get Number of bursts in file
index= l0file.packet_metadata.index
index_size = index.size
n_bursts   = index[index_size -1][0]
print(f"Number of bursts in {inputfile} = {n_bursts}")





selected_burst = 45
selection = l0file.get_burst_metadata(selected_burst)
#selection
import datetime
 
radar_data = l0file.get_burst_data(selected_burst)
 
#Set Offsets to access sub-images
range_offset  = 0
range_length  = radar_data.shape[1]-range_offset
azimuth_offset= 0
azimuth_length= radar_data.shape[0]-azimuth_offset

roi=np.array([range_offset, range_length, azimuth_offset, azimuth_length])


radar_data.shape
# Define sub-image
rd=radar_data[azimuth_offset:azimuth_offset+azimuth_length,range_offset:range_offset+range_length]

l0file.save_burst_data(selected_burst)
print(datetime.datetime.now())
 
del radar_data
radar_data=rd





rdd=focus_burst(selection,rd, roi)
rdd =np.flipud(rdd)


plt.figure(figsize=(20, 5))
plt.title("Abs of iFFT of (I,Q) Along Range & Azimuth Direction")
plt.imshow(abs(np.flip(rdd,1)), origin='lower', norm=colors.LogNorm(vmin=300, vmax=4000))
#plt.imshow(abs(rdd), origin='lower', norm=colors.LogNorm(vmin=300, vmax=4000))
plt.xlabel("Fast Time (down range)")
plt.ylabel("Slow Time (cross range)")
plt.show()




print(f"**************************************************** Finished focus_SM_burst ********************************************")

