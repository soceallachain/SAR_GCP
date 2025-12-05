#test_Focus_IW_Burst.py


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






def focus_SM_burst(burst_number: pd.DataFrame, rd:np.array,region_of_interest:np.array)->np.array:
    
    
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
    az_freq_vals         = np.arange(-az_sample_freq/2, az_sample_freq/2, 1/(PRI*len_az_line))
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

#Start here
#filepath = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/Dublin/SLC/S1A_IW_SLC__1SDV_20240105T180705_20240105T180733_051975_0647CF_89FB.SAFE/measurement/s1a-iw1-slc-vh-20240105t180706-20240105t180731-051975-0647cf-001.tiff

    
#filepath="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S3_RAW_0SDV_20230518T213602_20230518T213627_048593_05D835_4B93/RAW/S1A_S3_RAW__0SDV_20230518T213602_20230518T213627_048593_05D835_F012.SAFE/"
filepath="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S3_RAW_0SDV_20230518T213602_20230518T213627_048953_05D835_4B93/RAW/S1A_S3_RAW__0SDH_20230518T213602_20230518T213627_048593_05D835_F012.SAFE/"
 
 
#filename="s1a-s6-raw-s-vh-20210226t214354-20210226t214424-036766-045287.dat"
filename="s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
 
 
inputfile = filepath+filename
if os.path.exists(inputfile):
   print(f"{inputfile} found")
else:
   print(f"{inputfile} not found")

l0file   = sentinel1decoder.Level0File(inputfile)
 
#l0file.packet_metadata
#l0file.ephemeris
 
selected_burst = 8
selection = l0file.get_burst_metadata(selected_burst)
#selection
import datetime
 
radar_data = l0file.get_burst_data(selected_burst)
 
#Set Offsets to access sub-images
range_offset  = 5000
range_length  = 10500-range_offset
azimuth_offset= 100
azimuth_length= 6000-azimuth_offset

roi=np.array([range_offset, range_length, azimuth_offset, azimuth_length])


radar_data.shape
# Define sub-image
rd=radar_data[azimuth_offset:azimuth_offset+azimuth_length,range_offset:range_offset+range_length]

l0file.save_burst_data(selected_burst)
print(datetime.datetime.now())
 
del radar_data
radar_data=rd





rdd=focus_SM_burst(selection,rd, roi)



"""
plt.figure(figsize=(12, 12))
plt.title("Abs of iFFT of (I,Q) Along Range & Azimuth Direction")
plt.imshow(abs(np.flip(rdd,0)), origin='lower', norm=colors.LogNorm(vmin=300, vmax=4000))
plt.xlabel("Fast Time (down range)")
plt.ylabel("Slow Time (cross range)")
plt.show()

"""



print(f"**************************************************** Finished focus_SM_burst ********************************************")


new_rd = rd



# Ablate with num_ablation_lines
# If within num_ablation_lines/2 of start or end then ablate first or last num_ablation_lines
# if not then ablate num_ablation_lines/2 either side of offset
# If offset = 0 then ablating from end of range or azimuth; if offset = 1 then ablating from start
# If offset = 0.5 then ablating num_ablation_lines/2 either side of middle or range or azimuth 
#
#Update: 14 May 2024
#   Added funcstionality to mask either side of offset to provide windowing ability
#   Select 'BOTH' for ablation_axis, and set ablation_offset to 0.5 if want to centre
# Due to folding get ambiguities at edges which extend significantly into data
#ablation_axis         = 'BOTH'
#ablation_axis         = 'RANGE'
ablation_axis         = 'AZIMUTH'
ablation_offset       = 0.5
num_ablation_lines    = 2500

start_ra_ablation_pos = int(range_length   - range_length*ablation_offset)
start_az_ablation_pos = int(azimuth_length - azimuth_length*ablation_offset)

half_na_lines         = num_ablation_lines/2
f_half_na_lines       = int(np.floor(half_na_lines))
mask_value            = 1

if ablation_axis == 'RANGE':

   if (start_ra_ablation_pos >= f_half_na_lines) and ((range_length - start_ra_ablation_pos) >= f_half_na_lines):
      # Ensure range_length > num_ablation_lines!!  Might need to check use of np.floor command as well!
      new_rd[:, start_ra_ablation_pos - f_half_na_lines:start_ra_ablation_pos + f_half_na_lines]= mask_value
   elif start_ra_ablation_pos < half_na_lines:
       new_rd[:, 0:num_ablation_lines]= mask_value
   elif (range_length - start_ra_ablation_pos) < half_na_lines:
       new_rd[:, range_length - num_ablation_lines : range_length]= mask_value

elif ablation_axis == 'AZIMUTH':
   if (start_az_ablation_pos >= f_half_na_lines) and ((azimuth_length - start_ra_ablation_pos) >= f_half_na_lines):
      # Ensure azimuth_length > num_ablation_lines!!
      new_rd[start_az_ablation_pos-f_half_na_lines:start_ra_ablation_pos+ f_half_na_lines,:]= mask_value
   elif start_az_ablation_pos < half_na_lines:
       new_rd[0:num_ablation_lines, :]= mask_value
   elif azimuth_length - start_az_ablation_pos < half_na_lines:
       new_rd[azimuth_length - num_ablation_lines : azimuth_length, :]= mask_value

else:


   assert (start_ra_ablation_pos >= f_half_na_lines) and ((range_length - start_ra_ablation_pos) >= f_half_na_lines), "Centre position of window too close to range edge"+ \
           "for number of ablation lines chosen. Must change offset or reduce number of range ablation lines or both"
   
  # new_rd[:, 0:f_half_na_lines] = mask_value
  # new_rd[:, range_length - f_half_na_lines:range_length]= mask_value
   new_rd[:, 0: start_ra_ablation_pos - f_half_na_lines] = mask_value
   new_rd[:, start_ra_ablation_pos + f_half_na_lines:range_length]= mask_value


   assert (start_az_ablation_pos >= f_half_na_lines) and ((azimuth_length - start_az_ablation_pos) >= f_half_na_lines), "Centre position of window too close to azimuth edge"+ \
           "for number of ablation lines chosen. Must change offset or reduce number of azimuth ablation lines or both"
   new_rd[0:start_az_ablation_pos-f_half_na_lines,:] = mask_value
   new_rd[start_az_ablation_pos+ f_half_na_lines:azimuth_length,:]= mask_value
   #new_rd[0:start_az_ablation_pos-f_half_na_lines,:] = mask_value
   #new_rd[start_az_ablation_pos+ f_half_na_lines:azimuth_length,:]= mask_value
"""    
#Mask both RANGE and AZIMUTH directons. 
   #Mask RANGE direction 
   if (start_ra_ablation_pos >= f_half_na_lines) and ((range_length - start_ra_ablation_pos) >= f_half_na_lines):
      # Ensure range_length > num_ablation_lines!!  Might need to check use of np.floor command as well!
      new_rd[:, 0: start_ra_ablation_pos - f_half_na_lines] = mask_value
      new_rd[:, start_ra_ablation_pos + f_half_na_lines:range_length]= mask_value
   elif start_ra_ablation_pos < half_na_lines:
       new_rd[:, 0:num_ablation_lines]= mask_value
   elif (range_length - start_ra_ablation_pos) < half_na_lines:
       new_rd[:, range_length - num_ablation_lines : range_length]= mask_value

   #Mask AZIMUTH direction 
   if (start_az_ablation_pos >= f_half_na_lines) and ((azimuth_length - start_az_ablation_pos) >= f_half_na_lines):
      # Ensure azimuth_length > num_ablation_lines!!
      new_rd[0:start_az_ablation_pos-f_half_na_lines,:] = mask_value
      new_rd[start_az_ablation_pos+ f_half_na_lines:azimuth_length,:]= mask_value
   elif start_az_ablation_pos < half_na_lines:
       new_rd[0:num_ablation_lines, :]= mask_value
   elif azimuth_length - start_az_ablation_pos < half_na_lines:
       new_rd[azimuth_length - num_ablation_lines : azimuth_length, :]= mask_value

"""
rddd  = focus_SM_burst(selection, new_rd, roi)

##########
## Get metrics


if ablation_axis == 'RANGE' or 'BOTH':
   axis = 0
else:
   axis = 1 


range_sigma     = np.std(abs(np.flip(rddd,0)),axis)
range_mean      = np.mean(abs(np.flip(rddd,0)),axis)
ref_range_sigma = np.std(abs(np.flip(rdd,0)),axis)
ref_range_mean  = np.mean(abs(np.flip(rdd,0)),axis)

normalised_sigma      = np.divide(range_sigma, range_mean)
normalised_ref_sigma  = ref_range_sigma/ref_range_mean

metric = normalised_sigma - normalised_ref_sigma




fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2)

fig.suptitle('Ablation Studies', fontsize=12,fontweight="bold")

ax1.set_title('Focused Burst - 0 lines ablated ')
ax1.imshow(abs(np.flip(rdd,0)), origin='lower', norm=colors.LogNorm(vmin=300, vmax=4000))
ax1.set_xlabel('Fast Time (down range)')
ax1.set_ylabel('Slow Time (cross range)')


t2=f"Unfocused Burst -{num_ablation_lines} lines ablated in {ablation_axis} direction(s)"
if ablation_axis == "BOTH":
    t2=f"Unfocused Burst - Non-masked window = {num_ablation_lines} x {num_ablation_lines} "
ax2.set_title(t2)
ax2.imshow(abs(np.flip(new_rd,0)), origin='lower', norm=colors.LogNorm(vmin=1, vmax=30))
ax2.set_xlabel('Fast Time (down range)')
ax2.set_ylabel('Slow Time (cross range)')


t3 = f"Focused Burst -{num_ablation_lines} lines ablated in {ablation_axis} direction(s) "
if ablation_axis == "BOTH":
    t3=f"Focused Burst - Non-masked window = {num_ablation_lines} x {num_ablation_lines} "
ax3.set_title(t3)
ax3.imshow(abs(np.flip(rddd,0)), origin='lower', norm=colors.LogNorm(vmin=300, vmax=4000))
ax3.set_xlabel('Fast Time (down range)')
ax3.set_ylabel('Slow Time (cross range)')

t4 = f" Std Dev of Data"
ax4.set_title(t4)
#ax4.plot(metric)
ax4.plot(range_sigma, color='r', label = r'$\sigma$')
ax4.plot(ref_range_sigma, color='b', label = r'$\sigma_0$')
ax4.plot(range_sigma - ref_range_sigma, color='g', label = r'$\sigma$' + '-' + r'$\sigma_0$')
if ablation_axis == 'BOTH':
    ax4.set_xlabel(f"Range Line")
else:
    ax4.set_xlabel(f"{ablation_axis} Lines")
ax4.set_ylabel(r'$\sigma$')
#ax4.set_ylabel('Normalised Difference to Ref')
ax4.legend(loc=1)
ax4.grid(color='grey', linewidth = "0.4", ls = "-.")
plt.show()

"""
ax1=fig.add_subplot(2,1,1)
ax1.title("Focused Burst - 0 lines ablated in range")
ax1.imshow(abs(np.flip(rdd,0)), origin='lower', norm=colors.LogNorm(vmin=300, vmax=4000))
ax1.xlabel("Fast Time (down range)")
ax1.ylabel("Slow Time (cross range)")



ax2=fig.add_subplot(2,1,2)
ax2.title("Focused Burst - 500 lines ablated in range")
ax2.imshow(abs(np.flip(rddd,0)), origin='lower', norm=colors.LogNorm(vmin=300, vmax=4000))
ax2.xlabel("Fast Time (down range)")
ax2.ylabel("Slow Time (cross range)")
"""

"""
plt.figure(figsize=(12, 12))
plt.title("Sentinel-1 Raw I/Q Sensor Output")
#plt.imshow(abs(rd[:,:]), origin='lower',norm=colors.LogNorm(vmin=1, vmax=30))
#plt.imshow(abs(radar_data[:,:]), vmin=0, vmax=15, origin='lower')
#plt.imshow(abs(radar_data[:,:]), vmin=0, vmax=15, origin='lower')
plt.xlabel("Fast Time (down range)")
plt.ylabel("Slow Time (cross range)")
plt.show()
print(datetime.datetime.now())

plt.figure(figsize=(12, 12))
plt.title("Sentinel-1 Raw I/Q Sensor Output")
#plt.imshow(abs(rd[:,:]), origin='lower',norm=colors.LogNorm(vmin=1, vmax=30))
#plt.imshow(abs(radar_data[:,:]), vmin=0, vmax=15, origin='lower')
plt.imshow(abs(rd[:,:]), vmin=0, vmax=15, origin='lower')
plt.xlabel("Fast Time (down range)")
plt.ylabel("Slow Time (cross range)")
plt.show()
print(datetime.datetime.now())

"""
