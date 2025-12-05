#!/usr/bin/python3


# HTTP requests
import requests
# JSON parser
import json

# XML parser
import xml.etree.ElementTree as ET

# system modules
import os
import io
import re
import sys
import random
import shutil
# data manipulation
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import colors


filepath="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S3_RAW_0SDV_20230518T213602_20230518T213627_048953_05D835_4B93/SLC/product/S1A_S3_SLC__1SDH_20230518T213602_20230518T213627_048593_05D835_8BAA.SAFE/measurement/"
path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
GRD_dir   ="S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"
#SAFE_dir  ="S1B_IW_SLC__1SDV_20200709T054004_20200709T054031_022390_02A7EC_3C97.SAFE"
SAFE_dir  ="S1B_IW_SLC__1SDV_20200709T054004_20200709T054031_022390_02A7EC_3C97.SAFE"
SLC_fname ="s1b-iw1-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff"
SM_SLC_fname ="s1a-s3-slc-hh-20230518t213602-20230518t213627-048593-05d835-001.tiff"
#src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fname
src_file = filepath + SM_SLC_fname

if os.path.exists(src_file):
    print(f"{SLC_fname} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")

sub_array=array[6000:14000,4000:12000]
del array
amplitude = np.real(np.sqrt(sub_array * np.conjugate(sub_array)))

print(np.max(abs(sub_array)))
print(np.mean(abs(sub_array)))
print(np.min(abs(sub_array)))

#amplitude[2000:2010, 2000:2010]
plt.imshow(abs(np.flip(sub_array,0)), origin='lower', norm=colors.LogNorm(vmin=30, vmax=700))

#plt.imshow(np.log(amplitude), cmap ='pink')
plt.show()
print('Press any key to exit')
input()
sys.exit(1)






SLC_fname ="s1b-iw2-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff"
src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fname


if os.path.exists(src_file):
    print(f"{SLC_fname} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")

amplitude_iw2 = np.real(np.sqrt(array * np.conjugate(array)))

print(amplitude_iw2[2000:2010, 2000:2010])
#amplitude[2000:2010, 2000:2010]

plt.imshow(np.log(amplitude_iw2), cmap ='pink')
plt.show()


SLC_fname ="s1b-iw3-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff"
src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fname


if os.path.exists(src_file):
    print(f"{SLC_fname} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")

amplitude_iw3 = np.real(np.sqrt(array * np.conjugate(array)))

print(amplitude_iw3[2000:2010, 2000:2010])
#amplitude[2000:2010, 2000:2010]

plt.imshow(np.log(amplitude_iw3), cmap ='pink')
plt.show()



