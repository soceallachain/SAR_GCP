#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# November 11th 2026
# Name:       extract_ship_position_crops.py
# Script to:
#            1. Search for instances of named ship in Houston SM data
#            2. Extract all rows and determine SLC_ID and gps and row/col loc
#            3. For each instance extract crop around ship in raw and slc
#            4. Plot slc and raw & highlight ship position with a label
#            5. Save plots to pdf file
#


from modules.Houston_SM_Config import *
import copy

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


cvs_df            = pd.read_csv(csv_file)
df=cvs_df
ship_mssi_id = 338215000
def get_ship_information(df, ship_mssi_id):
    filtered_df = df.loc[(df['MMSI'] == ship_mssi_id) & (df['AOI_name'] != 'Z')]
    # OR
    # filtered_df = df.query("MMSI == @ship_mssi_id and AOI_name != 'Z'")
    return filtered_df
    
def get_ship_crop(df_row,crop_size, buffer=0):
    slc_id = df_row['scene_name']
    gps_point_lon_lat= [df_row['longitude'],df_row['latitude']]
    ship_name = df_row['VesselName']
    ship_mmsi = int(df_row['MMSI'])
    ship_imo  = df_row['IMO']
    if len(crop_size) ==1:
        aoi_size = [crop_size,crop_size]
    elif len(crop_size)==2:
        aoi_size = crop_size
    else:
        print(f'Error: Invalid crop_size entered in get_ship_crop()'
              f' - Must be list of 1 or 2 elements!')
        return -1
        
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
    raw_data = get_SM_raw_aoi_data(buffered_slc_aoi,files)
    
    # plot aoi and buffer region
    
    
    zoom_size = 512
    zoom_range_start = slc_data.shape[1]//2 - zoom_size//2
    zoom_range_end   = zoom_range_start + zoom_size
    zoom_azimuth_start = slc_data.shape[0]//2 - zoom_size//2
    zoom_azimuth_end   = zoom_azimuth_start + zoom_size
    zoom_slc_data      = slc_data[zoom_azimuth_start:zoom_azimuth_end,
                                  zoom_range_start: zoom_range_end
                                  ]
    zoom_raw_data      = raw_data[zoom_azimuth_start:zoom_azimuth_end,
                                  zoom_range_start: zoom_range_end
                                  ]
    titles=['SLC','Zoomed SLC', 'Raw', 'Zoomed Raw']
    data=[intensity(slc_data), intensity(zoom_slc_data),
          np.real(raw_data), np.real(zoom_raw_data)
          ]
    
    #fig,ax = plot_2d_array(intensity(slc_data))
    fig, axes = plt.subplots(2,2,figsize=(10,8))
    lon, lat = gps_point_lon_lat
    title_str = f"Location: ({lon:.4f}, {lat:.4f})"
    plt.suptitle(f'{shorten_slc_id(slc_id)} Ship: {ship_name}:MSSI:- {ship_mmsi} '
                 f'{title_str}', 
                 fontsize = 10
                 )
    for i, ax in enumerate(axes.ravel()):
        # Plot SLC data & Raw using log scale
        ax.imshow(data[i],
                      norm=colors.LogNorm(vmin=10,vmax = 1000000), 
                          aspect = 'auto'
                          )
                    
        ax.set_title(titles[i], fontsize=10, fontweight="bold")
        ax.set_xlabel('Range', fontsize = 10)
        ax.set_ylabel('Azimuth', fontsize = 10)
        ax.grid(True, linestyle="--", alpha=0.6)
        """
        x_start, x_end = buffered_slc_aoi[0],  buffered_slc_aoi[1]
        ax.set_xticks(np.linspace(x_start, x_end, num=5, dtype=int))
        """ 
    plt.tight_layout()
        
    
    
    # Add cicles for ship co-ordinates
    # Correct co-ordinates for AOI as 
    aoi_offset = [buffered_slc_aoi[0], buffered_slc_aoi[2]]
    corrected_point = slc_point_row_col - aoi_offset
    #add_circles_to_figure(ax, slc_point_row_col)
    add_circles_to_figure(axes[0,0], [corrected_point], radius = 50)
    add_circles_to_figure(axes[1,0], [corrected_point], radius = 50)
    

        
    # Draw rectangle around non-buffered aoi
    # Offset into buffered_slc_aoi is buffer//2 in both axes.
    # Note x,y <-> col,row correspondence
    rect = patches.Rectangle((buffer//2, buffer//2), 
                             slc_aoi[3]-slc_aoi[2],
                             slc_aoi[1]-slc_aoi[0], 
                             linewidth=1, edgecolor='g', facecolor='none')
    axes[0,0].add_patch(rect)
    rect2 = patches.Rectangle((buffer//2, buffer//2), 
                              slc_aoi[3]-slc_aoi[2],
                              slc_aoi[1]-slc_aoi[0], 
                              linewidth=1, edgecolor='g', facecolor='none')
    axes[1,0].add_patch(rect2)
    image_name = f'{shorten_slc_id(slc_id)}_Ship_{ship_name}_{ship_imo}.jpg'
    image_fname =f'{save_path}{image_name}'
    plt.savefig(image_fname, dpi=300, bbox_inches="tight")
    return axes
    

#ship_MSSI = 338215000 # Harvey Gladiator
ship_MSSI = 367651380 # ELK
df_rows = get_ship_information(df, ship_MSSI)
ship_position_of_interest = 0     
ship_crop_size =[512,512]  
buffer = 2048                        
ax = get_ship_crop(df_rows.iloc[ship_position_of_interest], 
                   ship_crop_size,
                   buffer)

for i, (_, row) in enumerate(df_rows.iterrows()):
    ax = get_ship_crop(row, ship_crop_size, buffer)
    # plt.savefig(f"ship_{ship_MSSI}_crop_{i}.jpg", dpi=300, bbox_inches='tight')
    plt.close('all')  # Prevent Jupyter memory bloat







