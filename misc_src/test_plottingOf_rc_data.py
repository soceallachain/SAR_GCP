#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:04:57 2025

@author: joe
"""



import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from SAR_focusing_module import *
from blob_module import *

from SAR_code_module import *

def get_statistics(roi):    
    mean = roi.mean()
    std  = roi.std()
    SNR  = mean/std
    return mean, std, SNR


def plot_2d_array(array, title='Title', _type ='log_normal'):
    fig, ax = plt.subplots()
    if _type == 'log_normal':
        im = ax.imshow(array,norm=colors.LogNorm(vmin=10,vmax = 1000000), aspect = 'auto')
        #plt.colorbar(im)
    elif _type == 'linear':
       im = ax.imshow(array,vmin=array.min(), vmax=array.max(), aspect = 'auto')
    elif _type == 'binary':
       im = ax.imshow(array, cmap='Greys_r' ,aspect = 'auto')

    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()
    return fig

def intensity(arr):
    return np.square(np.abs(arr))

plt.style.use('dark_background')
source = 'ext'
Dataset = 'Train'

test_dir = (f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/'
            f'test_velocity_hypothesis/')


rc_file = f'S1A_IW_SLC__1SDV_20200602T054837_20200602T054904_032834_03CD99_1929_RAW_IW2_rc_burst#6.npy'

rc_data = np.load(f'{test_dir}{rc_file}')

plot_2d_array(intensity(rc_data),rc_file)

plot_2d_array(np.square(np.abs(rc_data)), rc_file)