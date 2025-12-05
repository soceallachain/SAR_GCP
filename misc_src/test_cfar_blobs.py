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
import time
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

raw_burst_dir = '/home/joe/Work/Projects/RSAR/Test/SampleData/Public/raw_burst_data'
slc_burst_dir = '/home/joe/Work/Projects/RSAR/Test/SampleData/Public/slc_burst_data'
"""
raw_burst_dir = '/media/joe/WD_12TB_HD1/jc_work/Copernicus/Public/raw_burst_data'
slc_burst_dir = '/media/joe/WD_12TB_HD1/jc_work/Copernicus/Public/slc_burst_data'

raw_burst_file = 'S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794_RAW_IW2_real_burst#1.npy'
slc_burst_file = 'S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794_RAW_IW2_slc_burst#1.npy'
binary_mask    = 'S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794_RAW_IW2_binary_mask_burst#1.npy'
"""
path = '/home/joe/Work/Projects/RSAR/Test/SampleData/Public'
raw_burst_path = f'{path}/raw_burst_data/'
slc_burst_path = f'{path}/slc_burst_data/'
csv_path = f'{path}/Reports/'
img_path='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/t2'

# 2. Read CSV
csv_file = os.path.join(csv_path, 'Public_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships.csv')

burst_name='S1A_IW_SLC__1SDV_20200310T054859_20200310T054927_031609_03A47D_6794'
burst_number = 4
raw_burst_file = f'{burst_name}_RAW_IW2_real_burst#{burst_number}.npy'
slc_burst_file = f'{burst_name}_RAW_IW2_slc_burst#{burst_number}.npy'

slc_burst =np.load(f'{slc_burst_dir}/{slc_burst_file}')

burst_info = get_slc_id_burst_and_chip_numbers(slc_burst_file)
burst_ship_locations = get_ship_locations_in_bursts(burst_info, csv_file)

#plot raw burst using plot_2d_array
#plot_2d_array(raw_burst, f'Raw: {raw_burst_file}', 'linear')

#plot slc
detected_burst_data = np.square(np.abs(slc_burst))
plot_2d_array(detected_burst_data, f'SLC: {slc_burst_file}')
#Apply Median filter
medf_detected_burst_data = median_filter(detected_burst_data, size=3)
plot_2d_array(medf_detected_burst_data, f'Median Filtered  SLC: {slc_burst_file}')

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
    
