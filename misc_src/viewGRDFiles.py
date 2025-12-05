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

#GRD_dir   ="S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"        
#SAFE_dir  ="S1B_IW_SLC__1SDV_20200709T054004_20200709T054031_022390_02A7EC_3C97.SAFE"
#SAFE_dir  ="S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE.SAFE"
path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
GRD_dir   ="S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53"
SAFE_dir  ="S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53.SAFE"


#SLC_fnames =["s1b-iw1-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff", 
#             "s1b-iw2-slc-vv-20200709t054006-20200709t054031-022390-02a7ec-005.tiff",
#             "s1b-iw3-slc-vv-20200709t054004-20200709t054029-022390-02a7ec-006.tiff"]
#SLC_fnames =["s1b-iw1-slc-vh-20200709t054005-20200709t054030-022390-02a7ec-001.tiff", 
#             "s1b-iw2-slc-vh-20200709t054006-20200709t054031-022390-02a7ec-002.tiff",
#             "s1b-iw3-slc-vh-20200709t054004-20200709t054029-022390-02a7ec-003.tiff"]
#SLC_fnames =["s1b-iw1-slc-vv-20200709t054005-20200709t054030-022390-02a7ec-004.tiff"] 
#SLC_fnames =["s1b-iw2-slc-vv-20200709t054006-20200709t054031-022390-02a7ec-005.tiff"]
#SLC_fnames= ["s1b-iw3-slc-vv-20200709t054004-20200709t054029-022390-02a7ec-006.tiff"]

#GRD_fnames =["s1b-iw-grd-vv-20200709t054005-20200709t054030-022390-02a7ec-001.tiff", 
#             "s1b-iw-grd-vh-20200709t054005-20200709t054030-022390-02a7ec-002.tiff"]
GRD_fnames =["s1a-iw-grd-vh-20200421t051412-20200421t051437-032221-03ba16-002.tiff",
             "s1a-iw-grd-vv-20200421t051412-20200421t051437-032221-03ba16-001.tiff"]

i=0

polarization=['V-V', 'V-H']
#fig, axis = plt.subplots(1,3, constrained_layout=False, gridspec_kw = {'wspace':0, 'hspace':0})
fig, axis = plt.subplots(1,2, constrained_layout=True)
fig.set(facecolor="black")
fig.suptitle('GRD IW', color = 'white')
fig.tight_layout(pad=-5)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.001, hspace=0.001)

#gs1 = gridspec.GridSpec(1, 3)
#gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes.


image_list = []
for GRD_fname in GRD_fnames:
    src_file  = path + GRD_dir +'/GRD/' + SAFE_dir + '/measurement/' + GRD_fname
    if os.path.exists(src_file):
        print(f"{GRD_fname} found")
    else:
        print(f"ERROR: {src_file} Not found. Exiting")
        exit(0)
    src = rasterio.open(src_file)
    image = src.read(1)
    print(f"Array size = {image.shape}")
    print(image[2000:2010, 2000:2010])
    #image[image <=0]=1
    mean_image = skm.block_reduce(image, (10,10), np.mean)
    #plt.imshow(np.log(amplitude), cmap ='pink')
    image_list.append(mean_image)
#    plt.subplot(i)
    axis[i].imshow(np.flip((image_list[i]),1),norm=colors.LogNorm(vmin=10, vmax=512))
    #axis[i].imshow(np.flip(np.flip((image_list[i]),1),0),norm=colors.LogNorm(vmin=10, vmax=512))
    #axis[i].imshow(np.transpose(image_list[i]),norm=colors.LogNorm(vmin=10, vmax=512))
    axis[i].set(xticks=[], yticks=[], facecolor="black")
    axis[i].set_title(f"{polarization[i]}", color ='green')
    #plt.show()
    plt.show(block=False)

    i=i+1
    
input("Press any key to continue......")

exit(0)




