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
import skimage.measure as skm


path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
#GRD_dir   ="S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"
#SAFE_dir  ="S1B_IW_SLC__1SDV_20200709T054004_20200709T054031_022390_02A7EC_3C97.SAFE"
#SLC_fname ="s1b-iw1-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff"
#GRD_dir    ="Cairo_1"
#SAFE_dir   ="product/S1A_IW_SLC__1SDV_20240817T155704_20240817T155731_055255_06BC64_C952.SAFE"
#SLC_fnames = ["s1a-iw1-slc-vh-20240817t155705-20240817t155731-055255-06bc64-001.tiff",
#              "s1a-iw2-slc-vh-20240817t155704-20240817t155729-055255-06bc64-002.tiff",
#              "s1a-iw3-slc-vh-20240817t155705-20240817t155730-055255-06bc64-003.tiff"]

GRD_dir    ="Alexandria"
SAFE_dir   ="product/S1A_IW_SLC__1SDV_20240805T155728_20240805T155755_055080_06B602_DF20.SAFE"
SLC_fnames = ["s1a-iw1-slc-vh-20240805t155730-20240805t155755-055080-06b602-001.tiff",
              "s1a-iw2-slc-vh-20240805t155728-20240805t155754-055080-06b602-002.tiff",
              "s1a-iw3-slc-vh-20240805t155729-20240805t155755-055080-06b602-003.tiff"]

src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fnames[0]


if os.path.exists(src_file):
    print(f"{src_file} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")

amplitude = np.real(np.sqrt(array * np.conjugate(array)))

print(amplitude[2000:2010, 2000:2010])
#amplitude[2000:2010, 2000:2010]
amplitude[amplitude <=0]=1
mean_image = skm.block_reduce(amplitude, (10,10), np.mean)
#plt.imshow(np.log(amplitude), cmap ='pink')
plt.imshow(np.log(mean_image), cmap='viridis')
plt.show(block=False)


input("Press any key to continue......")

#exit(0)



#SLC_fname ="s1b-iw2-slc-vv-20200709t054006-20200709t054031-022390-02a7ec-005.tiff"
src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fnames[1]


if os.path.exists(src_file):
    print(f"{src_file} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")

amplitude_iw2 = np.real(np.sqrt(array * np.conjugate(array)))

amplitude_iw2[amplitude_iw2 <=0]=1
print(amplitude_iw2[2000:2010, 2000:2010])
#amplitude[2000:2010, 2000:2010]

plt.imshow(np.log(amplitude_iw2), norm=colors.LogNorm(vmin=2, vmax=300), aspect ='1')
plt.show(block=False)


#SLC_fname ="s1b-iw3-slc-vv-20200709t054004-20200709t054029-022390-02a7ec-006.tiff"
src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fnames[2]


if os.path.exists(src_file):
    print(f"{src_file} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    exit(0)


src = rasterio.open(src_file)
array = src.read(1)
print(f"Array size = {array.shape}")

amplitude_iw3 = np.real(np.sqrt(array * np.conjugate(array)))

amplitude_iw3[amplitude_iw3 <=0]=1
print(amplitude_iw3[2000:2010, 2000:2010])
#amplitude[2000:2010, 2000:2010]

plt.imshow(np.log(amplitude_iw3),norm=colors.LogNorm(vmin=2, vmax=300), aspect ='1')
#plt.show(block=False)
plt.show()



