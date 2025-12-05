import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import matplotlib.pyplot as plt
import os
from matplotlib import colors

from scipy.interpolate import interp1d





filepath="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S3_RAW_0SDV_20230518T213602_20230518T213627_048953_05D835_4B93/RAW/S1A_S3_RAW__0SDH_20230518T213602_20230518T213627_048593_05D835_F012.SAFE/"
filename = "s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
#filepath = "../data/russia/"
#filename = "s1a-iw-raw-s-vh-20230307t152137-20230307t152158-047540-05b562.dat"
inputfile = filepath+filename
if os.path.exists(inputfile):
   print(f"{inputfile} found")
else:
   print(f"{inputfile} not found")
  


l0file = sentinel1decoder.Level0File(inputfile)
l0file.packet_metadata
l0file.ephemeris
selected_burst = 8
selection = l0file.get_burst_metadata(selected_burst)
print(selection.iloc[0:3])



# Decode the IQ data
radar_data = l0file.get_burst_data(selected_burst)

# Cache this data so we can retreive it more quickly next time we want it
l0file.save_burst_data(selected_burst)

# Plot the raw IQ data extracted from the data file
plt.figure(figsize=(12, 12))
plt.title("Sentinel-1 Raw I/Q Sensor Output")
plt.imshow(abs(radar_data[:,:]), vmin=0, vmax=15, origin='lower')
plt.xlabel("Fast Time (down range)")
plt.ylabel("Slow Time (cross range)")
plt.show()



P_PRI = selection["PRI"]

print(P_PRI.size)

os._exit(os.EX_OK)




"""

# Image sizes
len_range_line = radar_data.shape[1]
len_az_line = radar_data.shape[0]

# Tx pulse parameters
c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
RGDEC = selection["Range Decimation"].unique()[0]
PRI = selection["PRI"].unique()[0]
rank = selection["Rank"].unique()[0]
suppressed_data_time = 320/(8*sentinel1decoder.constants.F_REF)
range_start_time = selection["SWST"].unique()[0] + suppressed_data_time
wavelength = sentinel1decoder.constants.TX_WAVELENGTH_M

"""


