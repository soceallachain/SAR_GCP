##viewIWSLCBursts.py
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
import matplotlib.gridspec as gridspec
from matplotlib import colors

path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
#GRD_dir   ="Dublin"
#SAFE_dir  ="S1A_IW_SLC__1SDV_20240107T063927_20240107T063954_051997_06488E_177F.SAFE"


#SLC_fnames =["s1a-iw1-slc-vh-20240107t063928-20240107t063953-051997-06488e-001.tiff",
#             "s1a-iw2-slc-vh-20240107t063929-20240107t063954-051997-06488e-002.tiff",
#             "s1a-iw3-slc-vh-20240107t063927-20240107t063952-051997-06488e-003.tiff"]

## Cairo
GRD_dir    ="Cairo_1"
SAFE_dir   ="product/S1A_IW_SLC__1SDV_20240817T155704_20240817T155731_055255_06BC64_C952.SAFE"
SLC_fnames = ["s1a-iw1-slc-vh-20240817t155705-20240817t155731-055255-06bc64-001.tiff"]
#              "s1a-iw2-slc-vh-20240817t155704-20240817t155729-055255-06bc64-002.tiff",
#              "s1a-iw3-slc-vh-20240817t155705-20240817t155730-055255-06bc64-003.tiff"]

#SLC_fnames =["s1a-iw2-slc-vh-20240107t063929-20240107t063954-051997-06488e-002.tiff"]
#GRD_dir   ="S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53"
#SAFE_dir  ="S1A_IW_SLC__1SDV_20200421T051411_20200421T051438_032221_03BA16_C5BE.SAFE"

#GRD_dir    ="S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"
#SAFE_dir   ="S1B_IW_SLC__1SDV_20200709T054004_20200709T054031_022390_02A7EC_3C97.SAFE"

#SLC_fnames =["s1b-iw1-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff", 
#             "s1b-iw2-slc-vv-20200709t054006-20200709t054031-022390-02a7ec-005.tiff",
#             "s1b-iw3-slc-vv-20200709t054004-20200709t054029-022390-02a7ec-006.tiff"]
#SLC_fnames =["s1a-iw1-slc-vh-20200421t051411-20200421t051438-032221-03ba16-001.tiff", 
#             "s1a-iw2-slc-vh-20200421t051412-20200421t051437-032221-03ba16-002.tiff",
#             "s1a-iw3-slc-vh-20200421t051412-20200421t051438-032221-03ba16-003.tiff"]
#SLC_fnames =["s1b-iw1-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff"] 
#SLC_fnames =["s1b-iw2-slc-vv-20200709t054006-20200709t054031-022390-02a7ec-005.tiff"]
#SLC_fnames= ["s1b-iw3-slc-vv-20200709t054004-20200709t054029-022390-02a7ec-006.tiff"]

i=0
#fig, axis = plt.subplots(1,3, constrained_layout=False, gridspec_kw = {'wspace':0, 'hspace':0})
#fig, axis = plt.subplots(1,3, constrained_layout=True)
#fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
#fig.tight_layout(pad=-5)
fig, axis = plt.subplots(1,3)
fig.subplots_adjust(wspace=-0.8, hspace=-20)
fig.set(facecolor="black")
fig.suptitle(f"SLC IW  {SAFE_dir}", color='white')


#gs1 = gridspec.GridSpec(1, 3)
#gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes.


image_list = []
for SLC_fname in SLC_fnames:
    src_file  = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fname
    if os.path.exists(src_file):
        print(f"{SLC_fname} found")
    else:
        print(f"ERROR: {src_file} Not found. Exiting")
        exit(0)
    src = rasterio.open(src_file)
    array = src.read(1)
    print(f"Array size = {array.shape}")
    amplitude = np.real(np.sqrt(array * np.conjugate(array)))
    #print(amplitude[2000:2010, 2000:2010])
    amplitude[amplitude <=0]=1
    mean_image = np.flipud(skm.block_reduce(amplitude, (10,10), np.mean))
    image_list.append(mean_image)
    axis[i].imshow((image_list[i]),norm=colors.LogNorm(vmin=2, vmax=300), aspect ='6')
    axis[i].set(xticks=[], yticks=[], facecolor="black")
    plt.show(block=False)
    i=i+1
    

input("Press any key to continue......")


exit(0)




