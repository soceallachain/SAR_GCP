#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:42:47 2025

@author: joe
"""
from  SAR_code_module import *
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.precision', 12)
sys.path.insert(0, '/home/joe/Misc')


SLC_Product_ID='S1B_IW_SLC__1SDV_20201117T064109_20201117T064136_024301_02E347_3F94'

#SLC_Product_ID ='S1B_IW_SLC__1SDV_20200619T051035_20200619T051102_022098_029EF7_F467'
#SLC_Product_ID ='S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B'
burst_num_to_plot = 5
burst_image, burst_ship_tagged_image, burst_binary_image = get_slc_burst_data_with_ships(SLC_Product_ID, burst_num_to_plot,polarization='vh')
fig, ax = plot_output_of_get_slc_burst_data(SLC_Product_ID, burst_num_to_plot, 
                                  burst_image, burst_ship_tagged_image,
                                  burst_binary_image)

chip_size=[512,3072]
burst_size = burst_image.shape
chip_number = 12
draw_chip(burst_image, chip_number, chip_size, ax)


#chip = get_raw_chip_data(burst_image, chip_number, chip_size)
chip        = get_chip_data(burst_image, chip_number, chip_size)
binary_chip = get_chip_data(burst_binary_image, chip_number, chip_size)


slc_data = chip
num_train = 10
num_guard = 3
pfa = 1e-12

detection_result = cfar_detector(slc_data, num_train, num_guard, pfa)


fig2, ax2 = plt.subplots(2,2)
ax2[0,0].imshow(np.abs(burst_ship_tagged_image), norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
ax2[0,1].imshow((slc_data), norm=colors.LogNorm(vmin=100,vmax = 30000), aspect = 'auto')
ax2[1,0].imshow(binary_chip, cmap='Greys_r', aspect = 'auto')
ax2[1,1].imshow(detection_result, cmap='Greys_r', aspect = 'auto')
ax2[0,0].set_title(f'SLC Amplitude Burst#{burst_num_to_plot}')
ax2[0,1].set_title(f'SLC Amplitude Chip#{chip_number}')
ax2[1,1].set_title(f'CFAR Mask for  Burst #{burst_num_to_plot}')
ax2[1,0].set_title(f'Binary Mask of Burst#{burst_num_to_plot} Chip#{chip_number}')
fig2.suptitle(f'{SLC_Product_ID} g{num_guard} t{num_train} pfa{pfa}', y=0.95)
draw_chip(burst_image, chip_number, chip_size, ax2[0,0])

kernel1 = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])

kernel2 = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])

detection_result_er1 = erosion(detection_result,kernel1)
detection_result_er1 = dilation(detection_result_er1,kernel1)

detection_result_er2 = erosion(detection_result, kernel2)
#detection_result_er2 = erosion(detection_result_er2,kernel2)
detection_result_er2 = dilation(detection_result_er2, kernel2)
detection_result_er3 = dilation(detection_result, kernel1)
detection_result_er3 = erosion(detection_result_er1,kernel1)

fig3, ax3 = plt.subplots(4,1)
ax3[0].imshow(detection_result,cmap='Greys_r', aspect = 'auto')
ax3[1].imshow(detection_result_er1,cmap='Greys_r', aspect = 'auto')
ax3[2].imshow(detection_result_er2,cmap='Greys_r', aspect = 'auto') 
ax3[3].imshow(detection_result_er3,cmap='Greys_r', aspect = 'auto')
ax3[0].set_title(f'CFAR Mask Erosion/Dilation x 0 for  Burst #{burst_num_to_plot} Chip#{chip_number}')
ax3[1].set_title(f'CFAR Mask Erosion/Dilation x 1 k1 for  Burst #{burst_num_to_plot} Chip#{chip_number}')
ax3[2].set_title(f'CFAR Mask Erosion/Dilation x 1 k2 for  Burst #{burst_num_to_plot} Chip#{chip_number}') 
ax3[3].set_title(f'CFAR Mask Dilation/Erosion x 1 k1 for  Burst #{burst_num_to_plot} Chip#{chip_number}')
fig3.suptitle(f'{SLC_Product_ID} :g{num_guard} t{num_train} Pfa = {pfa}', y=0.95)




# Flatten image to 1D array for histogram
#pixel_values = 10*np.log10(slc_data.flatten() + 1)
pixel_values = slc_data.flatten()

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(pixel_values, bins=256, range=(pixel_values.min(), pixel_values.max()), color='gray', alpha=0.7)
#plt.hist(pixel_values, bins=100, range=(pixel_values.min(), pixel_values.max()/25), color='gray', alpha=0.7)

# Labels and title
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Histogram of Image Pixel Values")

# Show plot
plt.show()

