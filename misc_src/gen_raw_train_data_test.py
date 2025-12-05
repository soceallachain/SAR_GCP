###############################################################################
"""
# Script to generate raw training data
# Accepts paths to directories containing RAW and SLC data
# Loops through RAW directories
#      - For each directory locates equivalent SLC based on mapping csv file
#      - Locates SLC annotation file (depending on swath and assumes vh pol)
#      - Reads SLC annotation file for burst information
#      - Reads RAW file for burst metadata
#      - Determines row and columns in raw bursts that match with SLC
#      - Extracts raw data and extracts Real and Imag & saves each to a file
#      - Generates matching SLC binary mask for each burst
#      - Re-samples binary mask to same range sampling frequency as RAW 
#      - Saves re-sampled binary mask to file
# Note: np.save saves to current directory unless supplied with path
#
"""
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
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.precision', 12)
sys.path.insert(0, '/home/joe/Misc')

from  SAR_code_module import *


swath_of_interest = 2
polarization      ='vh'
swath_of_interest_index = swath_of_interest - 1
IW_mode = swath_of_interest_index

# Get SLC file to match raw based on mapping file
scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"

#scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/xView3/Info/ESA_xView3_SceneName_Mapping.csv'
mapping_df         = pd.read_csv(scene_mapping_file)

#raw_path = "/mnt/smbshare/tempTest/Data/RAW_Data/"
#slc_path = "/mnt/smbshare/tempTest/Data/SLC_Data/"
raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"



#raw_directrories = os.listdir(raw_path)
raw_dirs =os.listdir(raw_path)
raw_directories = raw_dirs[0]

if (os.path.exists(raw_path)):
    print(f' Path to {raw_path} inaccessible: Exiting!')
    exit(0)
else:
    print(f' Path to {raw_path} found') 


for raw_directory in raw_directories:
    
    raw_root_file_path   = f'{raw_path}/{raw_directory}/RAW/'
    raw_safe_dir         = os.listdir(raw_root_file_path)
    raw_src_path         = f'{raw_root_file_path}{raw_safe_dir[0]}/'
    raw_filenames        = glob.glob(f'{raw_src_path}*{polarization}*.dat')
    raw_filename         = raw_filenames[0]
    if not (os.path.exists(raw_filename[0])):
        print(f' File {raw_filename} not found. Exiting')
        exit(0)
    else:
        raw_iw_data       = get_raw_burst_info(raw_filename)
        print(f' File {raw_filename} found. ')
        
    ########################################################################### 
    ### Get matching SLC file
    target_scene_id = raw_directory
    result = mapping_df.loc[mapping_df["scene_id"] == target_scene_id, "SLC_product_identifier"]
    if not result.empty:
         print(f"SLC_product_identifier for scene_id {target_scene_id}: {result.iloc[0]}")
    else:
         print(f"No match found for scene_id {target_scene_id}.")

    SLC_root_file_path = f'{slc_path}/{result.iloc[0]}'     
    SLC_root_file_path = SLC_root_file_path.replace("__","_")
    p1,p2,p3,p4,p5    = getSLCFilesFromRoot(SLC_root_file_path, swath_of_interest)
    slc_src_file      = p1
    annotation_file   = p2
    annotation_path   = p3
    SLC_fnames        = p4
    SAFE_dir          = p5
    del p1,p2,p3,p4,p5  
    print(f' Annotation file is: {annotation_file}')
    print(f' Annotation_dir  is: {annotation_path}')
    print(f' SLC_fname(s) are  : {SLC_fnames}')
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    Annotation_dir = annotation_path
    SLC_safe_id, _ = os.path.splitext(SAFE_dir)
    label_file     = scene_mapping_file
    #label_file     = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
    #############################################################################
    if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir) 
                                       and os.path.exists(slc_src_file)):
        print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
    else:
        print(f'ERROR: {annotation_file} or {Annotation_dir} '
              f' or {slc_src_file} Not found. Exiting')
        exit(0)
    
    # Have SLC src and annotation file and raw filename

    ###########################################################################
    # Get raw burst data and save Real and Imag bursts to match SLC shape
    slc_iw_burst_data = []
    slc_iw_data       = []
    pre_post_burst_information = get_pre_post_mid_times(annotation_file)
    #Get boundaries for each swath, burst &  grid in [row,col] & GPS coordinates 
    burst_data        = getSwathBurstGridGPSBoundaries(annotation_file)
    slc_iw_data.append(pre_post_burst_information)
    slc_iw_burst_data.append(burst_data)
    
    iw_mode = IW_mode
    #annotation_file = path + GRD_dir +'/SLC/' + Annotation_dir + change_extension(SLC_fnames[iw_mode],"xml")
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    slc_range_sampling_freq = float(root.find('.//rangeSamplingRate').text)

    bursts_not_to_save = []
    # for slc_burst in range(1, len(burst_data)) ### test and change
    for slc_burst in range(1,10):
        burst_of_interest = str(slc_burst)
        index_of_burst    = find_raw_start_burst(raw_iw_data[iw_mode], 
                            slc_iw_data[iw_mode][slc_iw_data[iw_mode]['burst_number']==burst_of_interest])
        
        print(f"\n\033[31m RAW index of matching IW{iw_mode+1}-slc burst#{slc_burst}    = {index_of_burst}\033[0m")

        # Using iloc to find times in slc_iw_data so need to subtract 1 from burst_number to align correctly
        burst_pre_time       = slc_iw_data[iw_mode].iloc[slc_burst-1]['pre_mid_time']
        burst_post_time      = slc_iw_data[iw_mode].iloc[slc_burst-1]['post_mid_time']
        slc_burst_start_time = slc_iw_data[iw_mode].iloc[slc_burst-1]['sensing_time']

        # get raw burst selected
        raw_burst_timings    = raw_iw_data[iw_mode].loc[index_of_burst]
        raw_burst_mid_time   = raw_burst_timings['Mid Burst Time']
        raw_burst_start_time = raw_burst_timings['Burst Start Time']
        raw_burst_end_time   = raw_burst_timings['Burst End Time']
        new_burst_start_time = raw_burst_mid_time - timedelta(seconds=burst_pre_time)
        new_burst_end_time   = raw_burst_mid_time + timedelta(seconds=burst_post_time)

        if (new_burst_start_time > raw_burst_start_time and 
                          new_burst_end_time < raw_burst_end_time):

            selected_burst   = index_of_burst
            l0file           = sentinel1decoder.Level0File(raw_filename)
            raw_burst_info   = l0file.get_burst_metadata(selected_burst)
            df               = raw_burst_info[['Coarse Time', 'Fine Time']]
            df['Start Time'] = pd.to_datetime(df.apply(lambda row:
                                              gps_to_utc(row['Coarse Time'], 
                                              row['Fine Time']), axis=1))
            pri              = raw_burst_info['PRI'].iloc[0]
            start_index      = df.index[0]
            result_row       = df[df['Start Time'] > (new_burst_start_time
                                - pd.Timedelta(seconds=pri/100)) ].index[0]
            start_row        = result_row - start_index
            result_row       = df[df['Start Time'] > (new_burst_end_time 
                                - pd.Timedelta(seconds=pri/100)) ].index[0]
            end_row          = result_row - start_index


            number_slc_range_samples     = slc_iw_burst_data[iw_mode][1][slc_burst-1][2][1]
            number_raw_range_samples     = raw_burst_info['Number of Quads'].iloc[0] *2
            RGDEC                        = raw_burst_info["Range Decimation"].unique()[0]
            raw_range_sampling_freq      = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
            raw_azimuth_line_time_length = number_raw_range_samples * 1/raw_range_sampling_freq
            slc_azimuth_line_time_length = number_slc_range_samples * 1/slc_range_sampling_freq

            raw_pre_mid_time      = slc_azimuth_line_time_length/2
            raw_post_mid_time     = slc_azimuth_line_time_length/2
            raw_start_range_index = int(number_raw_range_samples/2 
                                    - raw_pre_mid_time  * raw_range_sampling_freq)
            raw_end_range_index   = int(number_raw_range_samples/2 
                                    + raw_post_mid_time * raw_range_sampling_freq)

            number_azimuth_slc_lines = slc_iw_burst_data[iw_mode][1][slc_burst-1][1][0] \
                                        - slc_iw_burst_data[iw_mode][1][slc_burst-1][0][0]
            number_azimuth_raw_lines = end_row - start_row
            raw_slc_az_line_delta    = (number_azimuth_slc_lines 
                                       - number_azimuth_raw_lines)

            print(f'Raw burst initial start time = {raw_burst_start_time}'
                  f' ; Raw burst new start time = {new_burst_start_time}')
            print(f' Pre-time  = {burst_pre_time}, Post-time = {burst_post_time}')
            print(f' Number Azimuth SLC lines = {number_azimuth_slc_lines}')
            print(f' Number Azimuth Raw lines = {number_azimuth_raw_lines}')
            print(f' Number SLC Range Samples = {number_slc_range_samples}')
            print(f' Number RAW Range Samples = {number_raw_range_samples}')
            print(f' SLC Azimuth Line Time Length = {slc_azimuth_line_time_length}')
            print(f' RAW Azimuth Line Time Length = {raw_azimuth_line_time_length}')
            print(f' New RAW Burst Start Row is : {start_row}')
            print(f' New RAW Burst End Row is   : {end_row}')
            print(f' RAW Range Start = {raw_start_range_index}')
            print(f' RAW Range End = {raw_end_range_index}')   

            if raw_slc_az_line_delta != 0:
                print(f' WARNING: Mismatch in number of azimuth lines in RAW & SLC')
                print(f' \n\n number_azimuth_slc_lines = {number_azimuth_slc_lines}')
                print(f' number_azimuth_raw_lines = {number_azimuth_raw_lines}')
            else:
                radar_data = l0file.get_burst_data(index_of_burst)
                rd         = radar_data[start_row:end_row,raw_start_range_index, raw_end_range_index]
                np.save(f"{SLC_safe_id}_RAW_IW{IW_mode}_real_burst#{slc_burst}",np.real(rd).astype(np.float))
                np.save(f"{SLC_safe_id}_RAW_IW{IW_mode}_imag_burst#{slc_burst}",np.imag(rd).astype(np.float))

        else:
            print(f'ERROR : Burst {slc_burst} of {raw_filename} is not partitioned correctly. Either start or end line needs to be amended')
            bursts_not_to_save.append(slc_burst)

    ###########################################################################
    # get binary masks for SLC bursts
    ship_coords = getShipCoordinates(label_file, swath_of_interest, SLC_safe_id)
    # IW_gps            = getAllSwathBoundaries(SLC_fnames, Annotation_dir)
    burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
    IW_gps_boundary           = burst_information[0]
    burst_boundaries          = burst_information[1]
    burst_gps_boundaries      = burst_information[2]
    burst_grid_boundaries     = burst_information[3]
    burst_grid_gps_boundaries = burst_information[4]
    number_of_bursts          = len(burst_boundaries)
    bounding_box_size         = 100
    border                    = 25     
    burst_list                = list(range(1, number_of_bursts+1))
    
    # Need to modify bursts_to_save based on which bursts successfully matched 
    # between SLC & RAW
    # bursts_to_save === bursts_to_save/bursts_not_to_plot
    # intersection   = set(burst_list) & set(bursts_not_to_save)
    # bursts_to_save = [x for x in burst_list if not in intersection]
    # or .... bursts_to_save =set(burst_list) - set(bursts_not_to_save)
    swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
                                             burst_boundaries,
                                             burst_gps_boundaries,
                                             burst_grid_boundaries,
                                             burst_grid_gps_boundaries,
                                             bounding_box_size)
    print(f'SLC Training File Directory contains : {SLCTrainingFileDirectories}')
    
    if os.path.exists(slc_src_file):
        print(f"{SLC_fnames[swath_of_interest_index]} found")
    else:
        print(f"ERROR: {slc_src_file} Not found. Exiting")
        #exit(0)
    
    src            = rasterio.open(slc_src_file)
    img            = src.read(1)
    mask_storage_location = f'{SLC_safe_id}' 

    ### Note: Amend bursts_to_plot based on which bursts have equal
    ###       numbers of azimuth lines in SLC and RAW
    # Set multiplier for re-sampling
    freq_multiplier = raw_range_sampling_freq/slc_range_sampling_freq
    mask = save_SLC_binary_burst_training_data(img, SLC_safe_id,
                             bursts_to_plot,burst_boundaries,
                             np.array(swath_bounding_boxes),
                             burst_indices, bounding_box_size, border,
                             freq_multiplier, IW_mode
                            )
    # Check that resampled mask has same dimensions as raw burst data
    if mask.shape[0] == number_azimuth_raw_lines and mask.shape[1] == number_raw_range_samples:
       print(f'Sizes of RAW Bursts and Mask Match:')
    else:
       print(f'ERROR: Size of RAW Bursts and Mask do not match')
       exit(0)


##############################################################################
