#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:41:29 2026

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
import mpld3

#filepath = "../data/sao_paulo/"
#filename = "s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
#raw_filepath = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_RAW__0SDH_20180801T001745_20180801T001811_012062_01635C_9872.SAFE/"
root_path    = f'/mnt/SAR_bucket/data/Houston_SM/Houston_SM_No_OCN/'
SLC_ID       = f'S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A'

pol         = 'hv'

SLC_IDs=[SLC_ID]

for slc_id in SLC_IDs:

   #slc_id_path = root_path + SLC_IDs[14] +'/'
   slc_id_path = root_path + slc_id +'/'
   files       = get_SM_files_from_slc_id(slc_id_path, pol)   
   if files == -1:
       print(f'Warning: {slc_id} missing files')
       continue
   raw_file    = files[0]
   slc_file    = files[1]
   annot_file  = files[2]


   raw_burst_info, l0file = get_SM_raw_burst_info_v2(raw_file)
   # If > 1 acq chunks then raw_burst_info[0] has >1 rows
   number_of_chunks = len(raw_burst_info[0])






annotation_file=annot_file
print(datetime.now())
# This uses a lot of memory while running - move to early in code
raw_slc_offsets = get_SM_raw_slc_offsets(raw_file, annotation_file)
print(datetime.now())

# This uses a lot of memory while running - move to early in code
#raw_slc_offsets = get_SM_raw_slc_offsets(raw_file, annotation_file)


# Want to verify that can use gps co-ordinates on a known location to determine
# the [row,col] co-ordinates of the location in a SLC file and then extract a 
# region around the co-ordinates and plot them verifying accuracy


#1. Get slc, annotation and raw files

#2. Get location of interest
#location_lon_lat = [-94.783632, 29.546587]
#location_lon_lat = [-94.909698, 29.497012] ## Eagle Point Jetty 
#location_lon_lat = [-94.896680, 29.518980] ## Red Fish Island
location_lon_lat = [-94.766951, 29.366783] ## Bolivar Point Lighthouse


#3. Get grid for slc gps to [row,col] mapping
#   SM files have a grid spaced ~980 in aximuth 
#   - these referred to as quasi-bursts
#   Within each quasi burst have grids in range direction
#   - these referred to as sub-grids
slc_grid_info = get_SM_GridGPSBoundaries(annotation_file)
quasi_burst_gps_boundaries = slc_grid_info[2]
quasi_burst_index = getGrid(location_lon_lat, quasi_burst_gps_boundaries)

#4. Get sub-grid in which location_lon_lat is located
quasi_burst_sub_grid_gps_boundaries = slc_grid_info[4][quasi_burst_index]
quasi_burst_sub_grid_index = getGrid(location_lon_lat, quasi_burst_sub_grid_gps_boundaries)
gps_grid = np.array(quasi_burst_sub_grid_gps_boundaries[quasi_burst_sub_grid_index])
row_col_grid = np.array(slc_grid_info[3][quasi_burst_index][quasi_burst_sub_grid_index])




#5. Using sub grid index and gps & [row,col] grid data, interpolate for 
#   [row,col] position in slc file
#   
location_row_col = extrapolate(location_lon_lat, gps_grid, row_col_grid)


#6. Extract region around location and plot slc values
#   Open SLC file
slc_data = rasterio.open(slc_file)
slc_img  =slc_data.read(1)

n_roi_rows = 2000
n_roi_cols = 6000
roi_size   = [n_roi_rows, n_roi_cols]
roi_pos = place_roi(slc_img, roi_size, location_row_col)  
slc_roi = create_roi(slc_img, roi_pos[0], roi_pos[0] + roi_size[0],
                              roi_pos[1], roi_pos[1] + roi_size[1]
                              )

del slc_img, slc_data

fig, ax = plt.subplots()
im = ax.imshow(intensity(slc_roi),
               norm=colors.LogNorm(vmin=10,
                                   vmax = intensity(slc_roi).max()*0.5
                                   ), 
               aspect = 'auto', 
               origin = 'lower'
               )

# Draw outline of gps/row_col_grid
x_min = row_col_grid[0][1] - roi_pos[1]
y_min = row_col_grid[0][0] - roi_pos[0]
width = row_col_grid[2][1] - row_col_grid[0][1]
height = row_col_grid[2][0] - row_col_grid[0][0]
rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Label corner of gps grid
rect = patches.Rectangle((x_min+width-5, y_min-5), 10, 10, linewidth=2, edgecolor='g', facecolor='none')
ax.add_patch(rect)

# Place boundary around location_lon_lat - is in centre of roi
rect2 = patches.Rectangle((n_roi_cols//2 -5, n_roi_rows//2 -5), 10, 10, edgecolor='b', facecolor='none')
ax.add_patch(rect2)
ax.annotate(
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



#Get matching AOI in raw data


"""
l0file = sentinel1decoder.Level0File(raw_file)

raw_burst_info=get_SM_raw_burst_info(raw_file)
if len(raw_burst_info)>1:
    print(f'Warning: More than 1 SM Azimuth Block - need to process'
          f'differently!')

else:
    selected_burst = raw_burst_info[0].index[0]
    selection      = l0file.get_burst_metadata(selected_burst)
    radar_data     = l0file.get_burst_data(selected_burst)


slc_burst_info=get_SM_GridPoints(annot_file)
tree = etree.parse(annot_file)
root = tree.getroot()

# SLC parameters
n_slc_az_lines      = int(root.find('.//numberOfLines').text)
n_slc_range_samples = int(root.find('.//numberOfSamples').text)
azTimeInterval      = float(root.find('.//azimuthTimeInterval').text)
az_pri              = float(root.find('.//pri').text)
range_sampling_rate = float(root.find('.//rangeSamplingRate').text)

                                                                               

# Raw parameters 
n_raw_az_lines      = raw_burst_info[0]['Num Azimuth Lines'].iloc[0]
n_raw_range_samples = raw_burst_info[0]['Number of Quads'].iloc[0] *2


raw_start_time = raw_burst_info[0]['Burst Start Time']
slc_start_time = slc_burst_info.query("line == '0' and pixel == '0'"
                                      )['azimuth_time']

delta = (slc_start_time.iloc[0] - raw_start_time.iloc[0]).total_seconds()
# Get slc offset from first line of raw data
slc_az_start_offset_from_raw = int(np.ceil(delta / az_pri))
slc_az_end_offset_from_raw   = (n_raw_az_lines 
                             -  (slc_az_start_offset_from_raw + n_slc_az_lines)
                               )



# Check from range offset 
range_sample_diff = n_raw_range_samples - n_slc_range_samples
slc_range_start_offset = int(np.round(range_sample_diff / 2 ))
slc_range_end_offset   = n_raw_range_samples - (slc_range_start_offset + n_slc_range_samples)
"""
raw_data        = get_SM_data(raw_file)
radar_data      = raw_data[1]
meta_data       = raw_data[0]
l0file          = raw_data[2]

# This uses a lot of memory while running - move to early in code
#raw_slc_offsets = get_SM_raw_slc_offsets(raw_file, annotation_file)
# Create roi for raw data matched with slc
raw_roi = create_roi(radar_data, raw_slc_offsets[0] + roi_pos[0],
                                 raw_slc_offsets[0] + roi_pos[0] + roi_size[0],
                                 raw_slc_offsets[2] + roi_pos[1], 
                                 raw_slc_offsets[2] + roi_pos[1]+ roi_size[1])


             
#Set Offsets to access sub-images
range_offset  = 0
range_length  = raw_roi.shape[1]-range_offset
azimuth_offset= 0
azimuth_length= raw_roi.shape[0] - azimuth_offset

roi=np.array([range_offset, range_length, azimuth_offset, azimuth_length])


sm_focused_roi = focus_SM_burst(meta_data,raw_roi, roi, l0file,'rma')

#Plot focused SM roi and highlight location_lon_lat

                                  
fig, ax = plt.subplots()
im = ax.imshow(intensity(sm_focused_roi),
               norm=colors.LogNorm(vmin=5000,
                                   vmax = 1000000
                                   #vmax = intensity(sm_focused_roi).max()*0.5
                                   ), 
               aspect = 'auto', 
               origin = 'lower'
               )
x_min = row_col_grid[0][1] - roi_pos[1]
y_min = row_col_grid[0][0] - roi_pos[0]
width = row_col_grid[2][1] - row_col_grid[0][1]
height = row_col_grid[2][0] - row_col_grid[0][0]
rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Place boundary around location_lon_lat
rect = patches.Rectangle((x_min+width-5, y_min-5), 10, 10, linewidth=2, edgecolor='g', facecolor='none')
ax.add_patch(rect)

rect2 = patches.Rectangle((n_roi_cols//2 -5, n_roi_rows//2 -5), 10, 10, edgecolor='b', facecolor='none')
ax.add_patch(rect2)
ax.annotate(
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
print(datetime.now())


#
# Function to return grid points and gps grid points for Stripmap file
# Usually only 1 Azimuth Block of echo data in Stripmap - but might be more
# Here it is assumed that there is only 1
# References to bursts are a relic of IW mode processiing
#          
# Given point in a burst determine its gps location
# Point's offset from starting azimuth and range needs to be determined based
# on offset of burst from start of file 
# Not working with suffcient accuracy - why?
def get_gps_of_point_in_SM_data(point,burst, annotation_file):
    #Get all grid and gps boundaries for all bursts
    sm_info = get_SM_GridGPSBoundaries(annotation_file)
    #Get offset for burst of interest:  1st grid point for burst
    burst_offset = burst_info[1][burst_number-1][0]
    #Correct point position with offset
    offset_point = [point[0] + burst_offset[0], point[1] + burst_offset[1]]
    #Get grid points in burst
    burst_grid_points = burst_info[3][burst_number - 1]
    #Find which grid in burst the offset point is located in
    #Note: grid numbers start at 0 (bursts start at 1!!)
    grid_number = getGrid(offset_point,burst_grid_points)
    #Get gps & cartesian co-ordinates of grid in burst
    grid_number_gps = np.array(burst_info[4][burst_number-1][grid_number])
    grid_number_boundaries = np.array(burst_info[3][burst_number-1][grid_number])


    return cartesian_to_gps(offset_point, grid_number_boundaries,
                     grid_number_gps)


                
def get_SM_GridGPSBoundaries(annotation_file):

    
    tree = etree.parse(annotation_file)
    root = tree.getroot()         
   
    nLines         = int(root.find('.//numberOfLines').text)
    nRangeSamples  = int(root.find('.//numberOfSamples').text)
    
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
    
    print(f'Number of azimuth lines in each burst is {nLines}')
    print(f'Number of lines  is {nLines}')
    print(f'Number of Samples is {nRangeSamples}')
    burst_number    = 0
    burstGridPoints = []
    for burstGridPoint in root.findall('.//geolocationGridPoint'):
        azimuth_time = burstGridPoint.find('.//azimuthTime').text
        longitude    = burstGridPoint.find('.//longitude').text
        latitude     = burstGridPoint.find('.//latitude').text
        line         = burstGridPoint.find('.//line').text
        pixel        = burstGridPoint.find('.//pixel').text

        if pixel =='0':
            burst_number +=1

        azimuth_burst_end_time = datetime.strptime(azimuth_time,
                                                   '%Y-%m-%dT%H:%M:%S.%f') \
                                  + timedelta(seconds
                                  = (nBurstLines * azTimeInterval))

        burstGridPoints.append({
            'azimuth_time': azimuth_time,
             'longitude': longitude,
             'latitude': latitude,
             'line': line,
             'pixel': pixel,
             'burst_number' : str(burst_number),
             'azimuth_burst_end_time': azimuth_burst_end_time

        })
    #Get Ground Control Points from GeoLocationGridPoints
    df                = pd.DataFrame(burstGridPoints)
    gridPoints        = df[['burst_number','line','pixel',
                            'longitude','latitude']]
    gridControlPoints = gridPoints.to_numpy().astype(float)

    pixel_value = '0'
    burst_start_times = df[df['pixel'] == pixel_value]['azimuth_time']
    burst_end_time = []
    number_overlap_lines = []
    for i in range(0,len(burst_start_times)-1):
        # Convert string to datetime object
        t = datetime.strptime(burst_start_times.iloc[i],
                              '%Y-%m-%dT%H:%M:%S.%f')
        # Add burst interval time to start time
        t = t + timedelta(seconds=(nBurstLines*azTimeInterval))
        # Subtract end time of burst i from start of next burst i+1
        overlap_time = t- datetime.strptime(burst_start_times.iloc[i+1],
                                            '%Y-%m-%dT%H:%M:%S.%f')
        overlap_lines = 0   # Kludge for SM mode where no overlap ... or burstys
        #overlap_lines = round(overlap_time.total_seconds() / azTimeInterval)
        # Convert back to string if necessary (optional)
        burst_end_time.append(t)
        number_overlap_lines.append(overlap_lines)

    overlap_offset_lines = [x//2 for x in number_overlap_lines]
    merged_burst_start_line = []
    merged_burst_end_line   = []
    burst_corners           = []
    burst_gps_corners       = []
    gridPoints['merged_burst_start_line']= 0
    gridPoints['merged_burst_end_line']  = 0

    burst_ends           = gridPoints[(gridPoints['pixel']
                                 == str(nRangeSamples-1))]
    burst_starts         = gridPoints[(gridPoints['pixel'] == str(0))]
    burst_end_long_lat   = burst_ends[['longitude',
                                       'latitude']].to_numpy().astype(float)
    burst_start_long_lat = burst_starts[['longitude',
                                         'latitude']].to_numpy().astype(float)

    burst_grid_corners  = []
    burst_grid_gps      = []
    #for i in range(0,nBursts):
    for i in range (0,len(burst_start_times)-1):
        burst_start_line  = gridPoints[gridPoints['burst_number']
                                 == str(i+1)]['line'].to_numpy().astype(int)[0]
        burst_end_line    = gridPoints[gridPoints['burst_number']
                                 == str(i+2)]['line'].to_numpy().astype(int)[0]
        if i==0:
            #+ round(np.mean(overlap_offset_lines))  
            # NOT SURE ABOUT THIS, but may be source of offset 
            merged_start_line = burst_start_line
            merged_end_line   = (burst_end_line
                                - round(overlap_offset_lines[i]))
        else:
            merged_start_line = (burst_start_line
                                + round(overlap_offset_lines[i-1]))
            merged_end_line   = (burst_end_line
                                - round(overlap_offset_lines[i]))

        gridPoints.loc[gridPoints['burst_number'] == str(i+1),
                                  'merged_burst_start_line'] = merged_start_line

        gridPoints.loc[gridPoints['burst_number'] == str(i+1),
                                  'merged_burst_end_line']   = merged_end_line

        burst_corners.append([(merged_start_line,0),
                              (merged_end_line,0),
                              (merged_end_line,nRangeSamples-1),
                              (merged_start_line,nRangeSamples-1)]
                              )

        burst_gps_corners.append([burst_start_long_lat[i,:],
                                  burst_start_long_lat[i+1,:],
                                  burst_end_long_lat[i+1,:],
                                  burst_end_long_lat[i,:]]
                                  )

        ##### Add code for sub-grids
        boi1           = gridPoints[(gridPoints['burst_number'] == str(i+1))]
        boi2           = gridPoints[(gridPoints['burst_number'] == str(i+2))]
        # Check why burst coords work with getGrid but burst_grid coords don't
        long_lat_start = boi1[['longitude','latitude']].to_numpy().astype(float)
        long_lat_end   = boi2[['longitude','latitude']].to_numpy().astype(float)
        range_col      = boi1['pixel'].to_numpy().astype(int)
        grid_corners   = []
        grid_gps       = []
        for j in range(0,len(boi1)-1):
            grid_corners.append([(merged_start_line,range_col[j]),
                                 (merged_end_line, range_col[j]),
                                 (merged_end_line, range_col[j+1]),
                                 (merged_start_line, range_col[j+1])]
                                 )
            grid_gps.append([long_lat_start[j,:],
                             long_lat_end[j,:],
                             long_lat_end[j+1,:],
                             long_lat_start[j+1,:]]
                             )
        burst_grid_corners.append(grid_corners)
        burst_grid_gps.append(grid_gps)

    #Define boundaries of SLC file 
    file_corners = [(0,0),(nLines-1,0),
                   (nLines-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')

    sm_grid_gps_info=[]
    sm_grid_gps_info.append(gps_corners)
    sm_grid_gps_info.append(burst_corners)
    sm_grid_gps_info.append(burst_gps_corners)
    sm_grid_gps_info.append(burst_grid_corners)
    sm_grid_gps_info.append(burst_grid_gps)
    #return gps_corners, burst_corners, burst_gps_corners, burst_grid_corners, burst_grid_gps, burst_info
    return sm_grid_gps_info

