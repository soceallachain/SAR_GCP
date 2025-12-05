#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:04:26 2025

@author: joe
"""


from SAR_code_module import *
from blob_module import *
from datetime import datetime
from scipy.ndimage import median_filter
import numpy as np
import time, os 
def get_positions_in_roi(roi, positions):
    
    # Calculate ROI boundaries - Assume roi defined in clockwise order from
    # from bottom left
    roi_row_start = roi[0][0]
    roi_col_start = roi[0][1]
    roi_row_end   = roi[2][0]
    roi_col_end   = roi[2][1]
    
    # Convert to NumPy array
    pos_np = np.array(positions)
    # Handle empty positions list to avoid errors
    if pos_np.size == 0:
       print(f'No positions in position list. Exiting....')
       return []
    
    # Boolean mask for positions within ROI
    mask = (
        (pos_np[:, 0] >= roi_row_start) & (pos_np[:, 0] <= roi_row_end) &
        (pos_np[:, 1] >= roi_col_start) & (pos_np[:, 1] <= roi_col_end)
    )
    
    # Filtered positions
    roi_positions = pos_np[mask]
    
    # Convert back to list if needed
    roi_pos_list = roi_positions.tolist()
    return roi_pos_list


plt.ion()

# Set Train and Guard paramaters

train_band = 10
guard_band = 50
pfa        = 1e-2



dest_dir = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/Reports/CFAR_Results'
slc_dir  = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/slc_chip_data/'
csv_dir  = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/Reports/'

chip_list = os.listdir(slc_dir)



for file in chip_list[:2]:
    
    file_path = f'{slc_dir}{file}'
    if not os.path.isfile(file_path):
        print(f'Error: File not found : {file_path}')
        continue
    
    detected_slc_chip = np.load(file_path)
    if not detected_slc_chip.shape[0] ==512 and not detected_slc_chip.shape[1]== 512:
        print(f'Error: Chip size error: {file_path}')
        continue
    

    #plot slc
    plot_2d_array(detected_slc_chip, f'SLC: {file}')
    #Apply Median filter
    medf_detected_chip_data = median_filter(detected_slc_chip, size=3)
    plot_2d_array(medf_detected_chip_data, f'Median Filtered  SLC: {file}')

chip_rows =  512
chip_cols = 3072
chip_size = [chip_rows, chip_cols]
#chip_size =np.array(slc_burst.shape)
row_start = 0
col_start = 10000
row_end   = row_start + chip_size[0] -1
col_end   = col_start + chip_size[1] -1 
#chip_data = slc_burst[row_start:row_end, col_start:col_end]
chip_aoi = [[row_start, col_start], [row_end, col_start], 
            [row_end, col_end], [row_start, col_end]]
chip_data = create_roi(detected_burst_data, row_start, row_end, col_start, col_end)
#chip_data = slc_burst

#Set parameters for CFAR & Morphological operations
train_cells_size = 5
guard_ring_size = 20
pfa   = 1e-2
kernel1 = np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]])

start = time.time()
# Run CFAR
cfar_gamma_map2=optimized_cfar_detector(chip_data, train_cells_size, guard_ring_size, pfa)
end = time.time()
print(f'Time taken for CFAR is :{end-start} s')
# Perform morphological operations for removing noise / opening -closing
cfar_gamma_map2_eroded=erosion(cfar_gamma_map2,kernel1)
cfar_gamma_map2_eroded_dilated=dilation(cfar_gamma_map2_eroded,kernel1)              
cfar_gamma_map2_eroded_dilatedx2=dilation(cfar_gamma_map2_eroded_dilated,kernel1)


plot_2d_array(cfar_gamma_map2,f"{slc_burst_file.replace('.npy','')}",'linear')
plot_2d_array(cfar_gamma_map2_eroded,f"{slc_burst_file.replace('.npy','_eroded')}",'linear')
plot_2d_array(cfar_gamma_map2_eroded_dilated,f"{slc_burst_file.replace('.npy','_eroded_dilated')}",'linear')
plot_2d_array(cfar_gamma_map2_eroded_dilatedx2,f"{slc_burst_file.replace('.npy','_eroded_dilatedx2')}",'linear')

nearness_threshold = 5
blob_data=detect_blobs(cfar_gamma_map2_eroded_dilatedx2, nearness_threshold)

number_of_blobs = blob_data[0]
blob_centroids  = blob_data[1] 
blobs           = blob_data[2]

#Add offset into chip to get centroids in burst
blob_centroids = (np.array(blob_centroids) + np.array([row_start, col_start])).astype(int)

#Do blobs match ship locations
#Get ships in chip
burst_ship_locations = burst_ship_locations.iloc[:,2:4].to_numpy().astype(int)
chip_ship_locations  = get_positions_in_roi(chip_aoi, burst_ship_locations)
num_ships_in_chip    = len(chip_ship_locations)
distances            = distance.cdist(blob_centroids, chip_ship_locations)
ship_min_distances   = np.min(distances, axis=0)
blob_min_distances   = np.min(distances, axis=1)
ship_min_indices     = np.argmin(distances, axis = 0)
blob_min_indices     = np.argmin(distances, axis = 1)
#Get ship detection statistics
if number_of_ships <= number_of_blobs:
    # Determine how many ships are within threshold distance of blobs
    #First check that unique mapping between blobs and ships exists
    if len(set(ship_min_indices)) == len(ship_min_indices):
        number_true_positives  = sum(ship_min_distances 
                               < distance_threshold)
    else:
        #Have at least one instance of more than 1 ship per blob 
        #For each instance find which ship is closest to blob
        #Fnd how many repeated indices exist
        
        valid_blob_min_distances = blob_min_distances[list(set(ship_min_indices))]
        number_true_positives  = sum(valid_blob_min_distances 
                               < distance_threshold)
        
    number_false_positives = number_of_blobs - number_true_positives
    number_false_negatives = number_of_ships - number_true_positives 
else:
    if len(set(blob_min_indices)) == len(blob_min_indices):
        number_true_positives = sum(blob_min_distances < distance_threshold)
    else:
        valid_ship_min_distances = ship_min_distances[list(set(blob_min_indices))]
        number_true_positives  = sum(valid_ship_min_distances 
                               < distance_threshold)
        
    number_false_positives = number_of_blobs - number_true_positives
    number_false_negatives = number_of_ships - number_true_positives
    
