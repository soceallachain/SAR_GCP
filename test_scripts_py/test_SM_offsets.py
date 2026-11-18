#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11th August 2026
 ******************   test_SM_offsets.py  *************************************
Code to:
   - Open SM files, ensure that v2.0 of sentinel1decoder can read all raw chunks
   - get azimuth offset and range mid point
   - identify specific location in slc
   - focus raw data
   - idenitfy same location in raw
   - obtain error in range location

   Will have to:
   - merge chunks
   - get SM slice (we're calling 'bursts' slices now!) where location is
   - get GPS and row/col co-ordinates of slice
   - plot slc and compressed raw


@author: joe
"""



import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d
from modules.SAR_code_module import *
from modules.SAR_focusing_module import *
from scipy.ndimage import zoom
#filepath = "../data/sao_paulo/"
#filename = "s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
import mpld3





# This file tests the operation of get_SM_files_from_slc_id to verify that
# can plot S1 SLC and raw data and zoom dynamically and then use this 
# functionality to test whether the offsets in range between SM RAW and SLC
# are fixed  (for swath or node) or are random.
# 

# Do this by running test_get_SM_GridGPSBoundaries 

#Set DEBUG mode if do not want to save files
DEBUG = True
# Get list of SLC_IDs

#test_dir   = f'Houston_SM_2019_20'
#test_dir   = f'Houston_SM_No_OCN'
#root_path  = f'/mnt/SAR_bucket/data/jc_data/{test_dir}/'
#root_path  = f'/mnt/SAR_bucket/data/S1A_D_S1_S6_Samples/'

test_dir   = f'Houston_SM_No_OCN'
root_path  = f'/mnt/SAR_bucket/data/Houston_SM/{test_dir}/'

#test_dir   = f'SaoPaulo_SM_2019_20'
#root_path  = f'/mnt/SAR_bucket/data/SaoPaulo_SM/{test_dir}/'

save_fig_path = f'./tests/test_offsets'

pol         = 'hv'
pol2        = 'vh'
SLC_IDs     = os.listdir(root_path)

number_of_files_with_one_chunk = 0
number_of_files_with_more_than_one_chunk = 0
number_failed_SM_files = 0
number_failed_chunks   = 0
# Check how many of the SM files have > 1 acquisition chunk


#Define location of interest
location_lon_lat = [-94.766951, 29.366783] ## Bolivar Point Lighthouse
#location_lon_lat = [-95.070124, 29.765044] #
#location_lon_lat = [-95.098663, 30.327437] # Freeway Intersection at top[]

data_list        = []
swath ='_S6_'
number_of_swath_files = 0
for slc_id in SLC_IDs:

   #slc_id_path = root_path + SLC_IDs[14] +'/'
   #if swath not in slc_id:
   #if '_CC83' not in slc_id:
   if '_8D99' not in slc_id:
       continue
   
   if number_of_swath_files == 20:
       break
   slc_id_path = root_path + slc_id +'/'
   files       = get_SM_files_from_slc_id(slc_id_path, pol)   
   parts = slc_id.split('_')
   slc_id_name =f'{parts[0]}_{parts[1]}_{parts[-1]}'

   if files == -1:
       print(f'Warning: {slc_id} missing files with pol = {pol}. Try {pol2}')
       files = get_SM_files_from_slc_id(slc_id_path, pol2)        
       if files  == -1:
          print(f'Warning: {slc_id} missing files with pol = {pol2} as well. Definitely missing! ')
          continue
   raw_file    = files[0]
   slc_file    = files[1]
   annotation_file  = files[2]


   raw_burst_info, l0file = get_SM_raw_burst_info_v2(raw_file)
   # If > 1 acq chunks then raw_burst_info[0] has >1 rows
   number_of_chunks = len(raw_burst_info[0])
   if number_of_chunks ==1:
       number_of_files_with_one_chunk +=1
   else:
       number_of_files_with_more_than_one_chunk +=1
   
   print(f'{slc_id} has {number_of_chunks} chunks : {raw_burst_info[0].index[:].to_list()}')
   chunk_failed = False
   
   # Get chunk data
   #  place in slice array
   chunk_list     = raw_burst_info[0].index[:].to_list()
   chunk_data     = []
   chunk_metadata = []
   for chunk in chunk_list:
       try:
           chunk_metadata.append(l0file.get_acquisition_chunk_metadata(chunk))
           chunk_data.append(l0file.get_acquisition_chunk_data(chunk))
           
           print(f'{slc_id} Chunk#{chunk} successfully read')
       except:
           print(f'{slc_id} Chunk#{chunk} failed!')
           chunk_metadata.append([])
           chunk_data.append([])
           number_failed_chunks+=1
           chunk_failed = True
    
   if number_failed_chunks == 0:
       #process chunks into single array - is this really necessary?
       raw_data     = np.concatenate(chunk_data, axis = 0)
       raw_metadata = pd.concat(chunk_metadata, ignore_index=True)
   else:
       print(f'Warning : {slc_id_name} has {number_failed_chunks} failed chunk(s):'
             f' Ignoring and moving to next file'
             )
       continue
   
   # Need to fix get_SM_raw_slc_offsets to for chunk concatenation
   raw_slc_offsets = get_SM_raw_slc_offsets_v2(raw_file, annotation_file)
   
   #If have successfully read and appended all chunks
   number_of_swath_files +=1
   #   Get grid for slc gps to [row,col] mapping
   #   SM files have a grid spaced ~980 in aximuth 
   #   - these referred to as quasi-bursts
   #   Within each quasi burst have grids in range direction
   #   - these referred to as sub-grids
   slc_grid_info = get_SM_GridGPSBoundaries(annotation_file)
   quasi_burst_gps_boundaries = slc_grid_info[2]
   quasi_burst_index = getGrid(location_lon_lat, quasi_burst_gps_boundaries)
   
   
   # Get sub-grid in which location_lon_lat is located
   quasi_burst_sub_grid_gps_boundaries = slc_grid_info[4][quasi_burst_index]
   quasi_burst_sub_grid_index = getGrid(location_lon_lat, 
                                        quasi_burst_sub_grid_gps_boundaries
                                        )
   
   # Check if location not found in sub-grids
   # This can occur when location is within cartesian grid defined by the
   # burst_gps_boundaries but the sub-grids are contiguous along a curved path
   # so location may be between boundary of a sub-grid and the burst grid.
   # In this instance the location will be found in a sub-grid of one of the
   # adjacent 'bursts'. So check burst_sub_grids for quasi_burst_index +/- 1
   #
   # Check preceeding burst
   if quasi_burst_sub_grid_index == -1:
       quasi_burst_index -=1
       quasi_burst_sub_grid_gps_boundaries = slc_grid_info[4][quasi_burst_index]
       quasi_burst_sub_grid_index = getGrid(location_lon_lat, 
                                            quasi_burst_sub_grid_gps_boundaries
                                            )
       if quasi_burst_sub_grid_index == -1:
           quasi_burst_index +=2
           quasi_burst_sub_grid_gps_boundaries = slc_grid_info[4][quasi_burst_index]
           quasi_burst_sub_grid_index = getGrid(location_lon_lat, 
                                                quasi_burst_sub_grid_gps_boundaries
                                                )
           
           
       
   gps_grid = np.array(quasi_burst_sub_grid_gps_boundaries[quasi_burst_sub_grid_index])
   row_col_grid = np.array(slc_grid_info[3][quasi_burst_index][quasi_burst_sub_grid_index])
   
   #   Using sub grid index and gps & [row,col] grid data, interpolate for 
   #   [row,col] position in slc file
   #   
   location_row_col = extrapolate(location_lon_lat, gps_grid, row_col_grid)
   location_row_col = np.round(location_row_col).astype(int)
   
   
   # Extract region around location and plot slc values
   # Open SLC file
   slc_data = rasterio.open(slc_file)
   slc_img  =slc_data.read(1)
   # Centre roi on location
   n_roi_rows = 2000
   n_roi_cols = 6000
   roi_size   = [n_roi_rows, n_roi_cols]
   roi_pos = place_roi(slc_img, roi_size, location_row_col)
   slc_roi = create_roi(slc_img, roi_pos[0], roi_pos[0] + roi_size[0],
                                 roi_pos[1], roi_pos[1] + roi_size[1]
                                 )
   
   del slc_img, slc_data
   
   #Plot SLC data
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))
   im = ax1.imshow(intensity(slc_roi),
                  norm=colors.LogNorm(vmin=5,
                                      vmax = intensity(slc_roi).max()
                                      ),
                  aspect = 'auto',
                  origin = 'lower'
                  )
   ax1.set_title(f' SLC Plot for {slc_id_name}')
   
   # Draw outline of gps/row_col_grid
   x_min = row_col_grid[0][1] - roi_pos[1]
   y_min = row_col_grid[0][0] - roi_pos[0]
   width = row_col_grid[2][1] - row_col_grid[0][1]
   height = row_col_grid[2][0] - row_col_grid[0][0]
   rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
   ax1.add_patch(rect)
   
   # Label corner of gps grid
   rect = patches.Rectangle((x_min+width-5, y_min-5), 10, 10, linewidth=2, edgecolor='g', facecolor='none')
   ax1.add_patch(rect)
   
   # Place boundary around location_lon_lat - is in centre of roi
   rect2 = patches.Rectangle((n_roi_cols//2 -5, n_roi_rows//2 -5), 10, 10, edgecolor='b', facecolor='none')
   ax1.add_patch(rect2)
   ax1.annotate(
       f'[{gps_grid[3][0]:.4f}, {gps_grid[3][1]:.4f}]',    # The text label
       xy=(x_min+width -5, y_min -5),                 # Arrow point: (x, y) data coordinates
       xytext=(x_min+width -5 - 200, y_min -5 -200),  # Text position: (x, y) data coordinates
       arrowprops=dict(
           facecolor='black',       # Arrow color
           shrink=0.05,             # Pulls the arrow back slightly from text/point
           width=1,                 # Width of the arrow tail
           headwidth=8              # Width of the arrow head
       ),
       fontsize=7,
       color='black'
   )
   
   # Define center coordinates
   center_x = n_roi_cols // 2
   center_y = n_roi_rows // 2
   zoom_radius = 260  # +/- 200 pixels from center
 
   if not DEBUG:
   # Set view limits
      az_range_ratio = 1.6   # ratio of azimuth to range resolution to get square pixels
      ax1.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
      ax1.set_ylim(center_y - zoom_radius//az_range_ratio, center_y + zoom_radius//az_range_ratio)
   
   #plt.show()
   
   #Extract RAW data
   # Create roi for raw data matched with slc
   radar_data = raw_data
   meta_data  = raw_metadata
   raw_roi = create_roi(radar_data, raw_slc_offsets[0] + roi_pos[0],
                                    raw_slc_offsets[0] + roi_pos[0] + roi_size[0],
                                    raw_slc_offsets[2] + roi_pos[1],
                                    raw_slc_offsets[2] + roi_pos[1]+ roi_size[1])
   
   
   
   #Set Offsets to access sub-images
   range_offset   = raw_slc_offsets[2] + roi_pos[1]
   azimuth_offset = raw_slc_offsets[0] + roi_pos[0]
   range_length   = raw_roi.shape[1]
   azimuth_length = raw_roi.shape[0]  
   
   # Define offsets of roi in radar data so access correct meta data
   roi_offsets=np.array([range_offset, range_length, azimuth_offset, azimuth_length])
   # Focus raw data
   # 
   sm_focused_roi = focus_SM_burst(meta_data,raw_roi, roi_offsets, l0file,'rma')
   # Plot focused RAW data and add labels for location of interest
   #fig, ax = plt.subplots()
   im = ax2.imshow(intensity(sm_focused_roi),
                  norm=colors.LogNorm(vmin=50,
                                      #vmax = 1000000
                                      vmax = intensity(sm_focused_roi).max()
                                      ),
                  aspect = 'auto',
                  origin = 'lower'
                  )
   ax2.set_title(f"Focused Raw Plot for {slc_id_name}")
   x_min = row_col_grid[0][1] - roi_pos[1]
   y_min = row_col_grid[0][0] - roi_pos[0]
   width = row_col_grid[2][1] - row_col_grid[0][1]
   height = row_col_grid[2][0] - row_col_grid[0][0]
   rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
   ax2.add_patch(rect)
   
   # Place boundary around location_lon_lat
   rect = patches.Rectangle((x_min+width-5, y_min-5), 10, 10, linewidth=2, edgecolor='g', facecolor='none')
   ax2.add_patch(rect)
   
   rect2 = patches.Rectangle((n_roi_cols//2 -5, n_roi_rows//2 -5), 10, 10, edgecolor='b', facecolor='none')
   ax2.add_patch(rect2)
   ax2.annotate(
       f'[{gps_grid[3][0]:.4f}, {gps_grid[3][1]:.4f}]',    # The text label
       xy=(x_min+width -5, y_min -5),                 # Arrow point: (x, y) data coordinates
       xytext=(x_min+width -5 - 200, y_min -5 -200),  # Text position: (x, y) data coordinates
       arrowprops=dict(
           facecolor='black',       # Arrow color
           shrink=0.05,             # Pulls the arrow back slightly from text/point
           width=1,                 # Width of the arrow tail
           headwidth=8              # Width of the arrow head
       ),
       fontsize=7,
       color='black'
   )
   # Draw guide line of N pixels length
   # Extends N//2 units left and N//2 units right (total length = N)
   half_line_length = 50
   tick_size        = 5
   v_offset         = 10   # Displace line vertically
   ax2.plot([n_roi_cols//2 - half_line_length, n_roi_cols//2 + half_line_length],
           [n_roi_rows//2 + v_offset, n_roi_rows//2 + v_offset], color='y', linewidth=1
           )
   
   for step in range(0, half_line_length*2 + 1, 5):
       ax2.plot([n_roi_cols//2 - half_line_length + step, 
                 n_roi_cols//2 - half_line_length + step],
                [n_roi_rows//2 + v_offset - tick_size,
                 n_roi_rows//2 + v_offset + tick_size], color='y', linewidth =1
               )

   if not DEBUG:
   # Set view limits
      ax2.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
      ax2.set_ylim(center_y - zoom_radius//az_range_ratio, center_y + zoom_radius//az_range_ratio)
   
   plt.show(block=False)
      
   
   
   # Capture offset between SLC location and Focused Raw location using mouse
   # Wait for 1 mouse click (timeout=-1 means wait indefinitely)
   print("Click on the target location in the plot window...")
   points = plt.ginput(n=1, timeout=-1)
   
   # 4. Extract the clicked x, y coordinates
   if points:
       clicked_x, clicked_y = points[0]
       print(
           f"Recorded Position -> Row (Y): {int(np.round(clicked_y))}, Col (X):"
           f" {int(np.round(clicked_x))}"
       )
       offset = int(np.round(clicked_x)) - n_roi_cols//2
       print(f'Recorded offset for {slc_id_name}: {offset}')
       data_list.append(
               {
                   "slc_id_name": str(slc_id_name),
                   "Row"        : int(clicked_y),
                   "Col"        : int(clicked_x),
                   "Offset"     : offset
                   }
               )

   
   if not DEBUG:
       fig.savefig(f'{save_fig_path}/Offset_{slc_id_name}.png', dpi=300, bbox_inches='tight')
       plt.close(fig)

if not DEBUG:
    df = pd.DataFrame(data_list)
    df.to_csv(f'{save_fig_path}/Offsets_{test_dir}_{swath}.csv', index=False) 
       
       
       
       
    print(f'Out of {len(SLC_IDs)} SM files {number_of_files_with_one_chunk} have 1 '
          f'chunk - {100*number_of_files_with_one_chunk/len(SLC_IDs)}% ')
    
    print(f'Out of {len(SLC_IDs)} SM files Number of failed chunks is {number_failed_chunks}')
    print(f'Out of {len(SLC_IDs)} SM files Number of failed files is {number_failed_SM_files}')
          
    
    if not df.empty and "Offset" in df.columns:
       valid_files = number_of_swath_files - number_failed_SM_files
       print(f"Out of {valid_files} valid {swath} SM files --> "
             f"Mean Offset is: {df['Offset'].mean():.2f} | "
             f"Std Dev is: {df['Offset'].std():.2f} | "
             f"Max Offset is: {df['Offset'].max()} | "
             f"Min Offset is: {df['Offset'].min()}"
             )
    else:
       print("No valid offset data collected to compute statistics.")
    


"""
#print(f"Out of {len(SLC_IDs)-number_failed_SM_files} SM files --> "
print(f"Out of {number_of_S6_files-number_failed_SM_files} valid S6 SM files --> "
      f"Mean Offset is: {df['Offset'].mean()} | Std Dev is: {df['Offset'].std()} |"
      f"Max Offset is: {df['Offset'].max()} | Min Offset is: {df['Offset'].min()}"
      )
"""
f_slc = intensity(slc_roi[700:1300,2550:3450])
g_raw = intensity(sm_focused_roi[700:1300,2550:3450])

r_off, c_off = compute_image_offset(g_raw, f_slc, plot_surface=True)
print(f"Calculated Row Offset: {r_off:.2f} pixels")
print(f"Calculated Col Offset: {c_off:.2f} pixels")
input('Press ENTER to exit .....')

