#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:33:23 2026

@author: joe
"""


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


###############################################################################
# Plot an AOI and then buffer surrounding it. Label all ships in AOI
# The find ships in AOI + buffer and label those

# Accepts csv file for labels: must have column headings of:
#          scene_name    -- matches with slc_id
#          slc_row       -- row in slc file for ship locations 
#          slc_col       -- column in slc file for ship locations

def plot_aoi_buffer_with_points(slc_id, 
                                aoi_name, 
                                buffer_size,
                                files, 
                                csv_file,
                                plot_raw = False):
    # Choose AOI
    swath      = slc_id[4:6]
    buffer     = buffer_size
    slc_file   = files[1]
    annot_file = files[2]    
    slc_parameters = get_slc_parameters(annot_file)
    n_slc_rows = slc_parameters[0]
    n_slc_cols = slc_parameters[1] 
    #aoi_name = 'C'
    aoi_gps_lon_lat= scenes[swath][aoi_name]
    aoi_SW_row_col = np.round(get_SM_cartesian_point_using_annotation(
                                        aoi_gps_lon_lat[0], 
                                        annot_file)
                                 ).astype(int)
    aoi_NE_row_col = np.round(get_SM_cartesian_point_using_annotation(
                                        aoi_gps_lon_lat[1], 
                                        annot_file)
                                 ).astype(int)
    aoi_centre     = (aoi_SW_row_col + aoi_NE_row_col)//2
    
    # Re-use code above for centre of aoi
    slc_point_row_col = aoi_centre
    
    # Define slc aoi with buffer 
    buffered_slc_aoi_start_row = slc_point_row_col[0] - (aoi_size[0]+buffer)//2
    buffered_slc_aoi_start_col = slc_point_row_col[1] - (aoi_size[1]+buffer)//2
    
    if buffered_slc_aoi_start_row < 0:
        buffered_slc_aoi_start_row = 0
    if buffered_slc_aoi_start_col < 0:
        buffered_slc_aoi_start_col =0
    
    if buffered_slc_aoi_start_row + aoi_size[0] + buffer > n_slc_rows:
        buffered_slc_aoi_start_row = n_slc_rows - aoi_size[0]- buffer//2
    if buffered_slc_aoi_start_col + aoi_size[1] + buffer > n_slc_cols:
        buffered_slc_aoi_start_col = n_slc_cols - aoi_size[1] - buffer//2
    
    buffered_slc_aoi_end_row = buffered_slc_aoi_start_row + aoi_size[0] + buffer
    buffered_slc_aoi_end_col = buffered_slc_aoi_start_col + aoi_size[1] + buffer
    buffered_slc_aoi         = [buffered_slc_aoi_start_row, 
                                buffered_slc_aoi_end_row, 
                                buffered_slc_aoi_start_col,
                                buffered_slc_aoi_end_col
                                ]    
    slc_aoi                  = [buffered_slc_aoi_start_row + buffer//2, 
                                buffered_slc_aoi_end_row   - buffer//2, 
                                buffered_slc_aoi_start_col + buffer//2,
                                buffered_slc_aoi_end_col   - buffer//2
                                ]
    # Load slc file
    slc_data = get_SM_slc_aoi(buffered_slc_aoi, slc_file)
    # Get ship locations 
    df = pd.read_csv(csv_file)
    # Filter for ships in scene of interest
    subset = df.loc[df['scene_name'] == slc_id]
    # Get ships in aoi and buffer
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
    # Apply offset into slc file  top get correct locations of ships in
    # buffered aoi
    aoi_offset = [buffered_slc_aoi[0],buffered_slc_aoi[2]]
    ship_locations_in_aoi = points_in_roi[
                           ['slc_row', 'slc_col']
                                          ].to_numpy().astype(int) - aoi_offset

    
    
    
    # plot aoi and buffer region
    short_slcid=shorten_slc_id(slc_id)
    fig1,ax1 = plot_2d_array_with_labelled_points(intensity(slc_data), 
                                                  ship_locations_in_aoi.tolist(),
                                                  f'{short_slcid} {slc_aoi}'
                                                  )
    
    # Draw rectangle around non-buffered aoi
    # Offset into buffered_slc_aoi is buffer//2 in both axes.
    # Note x,y <-> col,row correspondence
    width = slc_aoi[3]-slc_aoi[2]
    height = slc_aoi[1]-slc_aoi[0]
    rect = patches.Rectangle((buffer//2, buffer//2), 
                             width,
                             height,
                             linewidth=1,
                             linestyle ='--',
                             edgecolor='y', facecolor='none')
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
    # Get offset of slc_aoi into buffered aoi
    aoi_offset = [buffered_slc_aoi[0],buffered_slc_aoi[2]]
    ship_locations_in_slc_aoi = points_in_roi[
                       ['slc_row', 'slc_col']].to_numpy().astype(int) - aoi_offset
    
    add_circles_to_figure(ax1, ship_locations_in_slc_aoi,radius=50, edge_colour='green', linewidth=1)
    
    handles = ax1
    if plot_raw:
        raw_data = get_SM_raw_aoi_data(buffered_slc_aoi,files)
        fig2,ax2 = plot_2d_array_with_labelled_points(np.real(raw_data), 
                                                      ship_locations_in_aoi.tolist(),
                                                      f'{short_slcid} {np.squeeze(slc_aoi)}'
                                                      )
        rect = patches.Rectangle((buffer//2, buffer//2), width, height, 
                                 linewidth=1, linestyle ='--',
                                 edgecolor='cy', facecolor='none')
        ax2.add_patch(rect)
        add_circles_to_figure(ax2, ship_locations_in_slc_aoi,
                              radius=25, edge_colour='green', linewidth=2)
        handles = [ax1, ax2]
    return handles




buffer_size=1000
aoi_name = 'C'
handles = plot_aoi_buffer_with_points(slc_id, aoi_name,buffer_size,files, 
                                      csv_file, plot_raw = True 
                                      )
