#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# November 4th 2026
# Script to:
#            1. Accept SLC file and gps location
#            2. Define AOI size in pixels e.g. 2048x2048
#            3. Define AOI in gps SW/NE using place_roi()
#            4. Plot AOI slc data
#            5. Overlay gps location
#


from modules.Houston_SM_Config import *


scenes = {
    'S3': {
        'A': [(-94.9837, 29.0670), (-94.8470, 29.1525)],
        'B': [(-94.8306, 29.0862), (-94.6690, 29.2512)],
        'C': [(-94.6358, 29.1167), (-94.5778, 29.3966)],
        'Sea': Sea_scene
    },
    'S6': {
        'A': [(-94.9837, 29.0670), (-94.8470, 29.1525)],
        'B': [(-94.8306, 29.0862), (-94.6690, 29.2512)],
        'C': [(-94.6358, 29.1167), (-94.4837, 29.4108)],
        'Sea': Sea_scene
    }
}

slc_id = 'S1A_S3_SLC__1SDH_20160712T001814_20160712T001838_012108_012BED_7BED'
gps_point_lon_lat= [-94.7090323728814,29.2460655932203]
aoi_size = [2048,2048]
buffer   = 1000
 
files      = get_raw_slc_files(slc_id)
slc_file   = files[1]
annot_file = files[2]

slc_parameters = get_slc_parameters(annot_file)
n_slc_rows = slc_parameters[0]
n_slc_cols = slc_parameters[1] 

# Get row col of gps_lon_lat
slc_point_row_col = np.round(get_SM_cartesian_point_using_annotation(
                                    gps_point_lon_lat, 
                                    annot_file)
                             ).astype(int)


# Define slc aoi with buffer 
slc_aoi_start_row = slc_point_row_col[0] - (aoi_size[0]+buffer)//2
slc_aoi_start_col = slc_point_row_col[1] - (aoi_size[1]+buffer)//2

if slc_aoi_start_row <0:
    slc_aoi_start_row = 0
if slc_aoi_start_col <0:
    slc_aoi_start_col =0

if slc_aoi_start_row + aoi_size[0] + buffer > n_slc_rows:
    slc_aoi_start_row = n_slc_rows - aoi_size[0]- buffer//2
if slc_aoi_start_col + aoi_size[1] + buffer > n_slc_cols:
    slc_aoi_start_col = n_slc_cols - aoi_size[1] - buffer//2


slc_aoi_end_row = slc_aoi_start_row + aoi_size[0] + buffer
slc_aoi_end_col = slc_aoi_start_col + aoi_size[1] + buffer
buffered_slc_aoi = [slc_aoi_start_row, 
                    slc_aoi_end_row, 
                    slc_aoi_start_col,
                    slc_aoi_end_col
                    ]

slc_aoi =[slc_aoi_start_row + buffer//2, 
          slc_aoi_end_row   - buffer//2, 
          slc_aoi_start_col + buffer//2,
          slc_aoi_end_col   - buffer//2
          ]

# Load slc file
slc_data = get_SM_slc_aoi(buffered_slc_aoi, slc_file)
raw_data = get_SM_raw_aoi_data(buffered_slc_aoi,files)

# plot aoi and buffer region
fig,ax = plot_2d_array(intensity(slc_data))

# Add cicles for ship co-ordinates
# Correct co-ordinates for AOI as 
aoi_offset = [buffered_slc_aoi[0], buffered_slc_aoi[2]]
corrected_point = slc_point_row_col - aoi_offset
#add_circles_to_figure(ax, slc_point_row_col)
add_circles_to_figure(ax, [corrected_point])

# Draw rectangle for aoi

# Label with aoi corner gps co-ordinates

# Plot raw aoi

# plot aoi and buffer region
fig1,ax1 = plot_2d_array(np.real(raw_data))
add_circles_to_figure(ax1, [corrected_point])



###############################################################################
# Plot an AOI and then buffer surrounding it. Label all ships in AOI
# The find ships in AOI + buffer and label those

# Choose AOI
swath = slc_id[4:6]
aoi_name = 'C'


aoi_gps_lon_lat = scenes[swath][aoi_name]
aoi_SW_row_col = np.round(get_SM_cartesian_point_using_annotation(
                                    aoi_gps_lon_lat[0], 
                                    annot_file)
                             ).astype(int)
aoi_NE_row_col = np.round(get_SM_cartesian_point_using_annotation(
                                    aoi_gps_lon_lat[1], 
                                    annot_file)
                             ).astype(int)


aoi_centre = (aoi_SW_row_col + aoi_NE_row_col)//2


# Re-use code above for centre of aoi
slc_point_row_col = aoi_centre

# Define slc aoi with buffer 
buffered_slc_aoi_start_row = slc_point_row_col[0] - (aoi_size[0]+buffer)//2
buffered_slc_aoi_start_col = slc_point_row_col[1] - (aoi_size[1]+buffer)//2

if buffered_slc_aoi_start_row <0:
    buffered_slc_aoi_start_row = 0
if buffered_slc_aoi_start_col <0:
    buffered_slc_aoi_start_col =0

if buffered_slc_aoi_start_row + aoi_size[0] + buffer > n_slc_rows:
    buffered_slc_aoi_start_row = n_slc_rows - aoi_size[0]- buffer//2
if buffered_slc_aoi_start_col + aoi_size[1] + buffer > n_slc_cols:
    buffered_slc_aoi_start_col = n_slc_cols - aoi_size[1] - buffer//2


buffered_slc_aoi_end_row = buffered_slc_aoi_start_row + aoi_size[0] + buffer
buffered_slc_aoi_end_col = buffered_slc_aoi_start_col + aoi_size[1] + buffer
buffered_slc_aoi = [buffered_slc_aoi_start_row, 
                    buffered_slc_aoi_end_row, 
                    buffered_slc_aoi_start_col,
                    buffered_slc_aoi_end_col
                    ]

slc_aoi =[buffered_slc_aoi_start_row + buffer//2, 
          buffered_slc_aoi_end_row   - buffer//2, 
          buffered_slc_aoi_start_col + buffer//2,
          buffered_slc_aoi_end_col   - buffer//2
          ]
# Load slc file
slc_data = get_SM_slc_aoi(buffered_slc_aoi, slc_file)


# Get ship locations 
df = pd.read_csv(csv_file)

# Assuming ROI defines bounding limits: [row_min, row_max, col_min, col_max]
# or ROI = [(col_min, row_min), (col_max, row_min), ...]
subset = df.loc[df['scene_name'] == slc_id]

roi = buffered_slc_aoi
row_vals = subset['slc_row'].to_numpy()
col_vals = subset['slc_col'].to_numpy()
row_min=roi[0]
row_max=roi[1]
col_min=roi[2]
col_max=roi[3]

mask = (
    (row_vals >= row_min) & (row_vals <= row_max) &
    (col_vals >= col_min) & (col_vals <= col_max)
)


points_in_roi = subset[mask]

aoi_offset = [buffered_slc_aoi[0],buffered_slc_aoi[2]]
ship_locations_in_aoi = points_in_roi[
                       ['slc_row', 'slc_col']
                                      ].to_numpy().astype(int) - aoi_offset


#raw_data = get_SM_raw_aoi_data(buffered_slc_aoi,files)

# plot aoi and buffer region
short_slcid=shorten_slc_id(slc_id)
fig1,ax1 = plot_2d_array_with_labelled_points(intensity(slc_data), ship_locations_in_aoi.tolist(),f'{short_slcid} {slc_aoi}')



# Draw rectangle around non-buffered aoi
# Offset into buffered_slc_aoi is buffer//2 in both axes.
# Note x,y <-> col,row correspondence
rect = patches.Rectangle((buffer//2, buffer//2), slc_aoi[3]-slc_aoi[2],slc_aoi[1]-slc_aoi[0], linewidth=1, edgecolor='y', facecolor='none')
ax1.add_patch(rect)


# Get ships only in slc_aoi

roi = slc_aoi
row_vals = subset['slc_row'].to_numpy()
col_vals = subset['slc_col'].to_numpy()
row_min=roi[0]
row_max=roi[1]
col_min=roi[2]
col_max=roi[3]

mask = (
    (row_vals >= row_min) & (row_vals <= row_max) &
    (col_vals >= col_min) & (col_vals <= col_max)
)

# Mask now only highlights ships in non-buffered AOI
points_in_roi = subset[mask]

aoi_offset = [buffered_slc_aoi[0],buffered_slc_aoi[2]]
ship_locations_in_slc_aoi = points_in_roi[
                       ['slc_row', 'slc_col']
                                      ].to_numpy().astype(int) - aoi_offset

add_circles_to_figure(ax1, ship_locations_in_slc_aoi,radius=25, edge_colour='green', linewidth=2)