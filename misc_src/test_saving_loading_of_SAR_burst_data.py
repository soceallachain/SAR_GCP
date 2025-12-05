import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2
import time

from matplotlib import colors

burst_of_interest = 5
root_fname        = "S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E"


real_data = np.load(f'{root_fname}_real_burst#{burst_of_interest}.npy')
imag_data = np.load(f'{root_fname}_imag_burst#{burst_of_interest}.npy')
bin_data  = np.load(f'{root_fname}_binary_mask_burst#{burst_of_interest}.npy')

fig, ax = plt.subplots(2,1)
ax[0].imshow(np.square(real_data) + np.square(imag_data),norm=colors.LogNorm(vmin=700,vmax = 20000), aspect = '10')
ax[1].imshow(bin_data, cmap='gray', aspect = '10')
fig.tight_layout()
plt.show()

