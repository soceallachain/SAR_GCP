

from lxml import etree
# system modules
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
import sys
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
pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.precision', 12)
sys.path.insert(0, '/home/joe/Misc')

from  SAR_code_module import *


SLC_PRODUCT_ID ="S1B_IW_SLC_1SDV_20200619T051035_20200619T051102_022098_029EF7_F467"
burst_to_plot  =8

burst_data = get_single_burst_image_data(SLC_PRODUCT_ID, burst_to_plot)


amplitude = np.abs(burst_data)
intensity = amplitude*amplitude +1
I_dB      = 10*np.log10(intensity)
I_min = intensity.min()
I_max = intensity.max()

fig1, ax1 = plt.subplots(1,2)
ax1[0].imshow(amplitude, norm=colors.LogNorm(vmin=1,vmax = 300), aspect = 'auto')
ax1[1].imshow(I_dB, vmin=10, vmax = 40, aspect = 'auto')
ax1[0].set_title(f'Amplitude of Burst #{burst_to_plot}')
ax1[1].set_title(f'Intensity of Burst #{burst_to_plot}')
fig1.suptitle(f'{SLC_PRODUCT_ID}', y=0.95)
plt.show()

from skimage import exposure
I_dB_normalised = (I_dB - I_dB.min())/(I_dB.max() - I_dB.min())
I_dB_equalised  = exposure.equalize_hist(I_dB_normalised)
fig2, ax2 = plt.subplots(1,2)
ax2[0].imshow(I_dB_normalised, cmap='gray', aspect ='auto')
ax2[1].imshow(I_dB_equalised,  cmap='gray', aspect ='auto')
ax2[0].set_title(f'Hist Normalised Decibel Scale of Burst #{burst_to_plot}')
ax2[1].set_title(f'Hist Equalised and Normalised Decibel Scale of Burst #{burst_to_plot}')

fig2.suptitle(f'{SLC_PRODUCT_ID}', y=0.95)
plt.show()



