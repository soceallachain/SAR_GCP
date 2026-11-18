## SAR_code_module
##
## Functions to support SAR SLC file burst extraction and ship 
## 


from lxml import etree
# system modules
import sentinel1decoder
import os
import io
import re
import sys
import glob
import random
import shutil
# data manipulation
import shapely
from shapely.ops import nearest_points
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
from scipy.interpolate import interp1d
from scipy.stats import norm, gamma
#from shapely.geometry import Polygon, Point
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import geodatasets
import reverse_geocoder as rg
import pycountry
import warnings
from modules.blob_module import *

from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from matplotlib.lines import Line2D

from modules.SAR_focusing_module import *



pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.precision', 12)


default_fontsize = 6
   
freq_multipliers=[1, 54.595959/64.345238, 46.915402/64.345238]

def get_date():
    
    now = datetime.now()
    return(now.strftime('%d%b%Y'))  # e.g. 01Aug2019
    


def get_ship_locations(chip_information, csv_file):
    df           = pd.read_csv(csv_file)    
    slc_id       = chip_information[0]
    burst_number = chip_information[1]
    chip_number  = chip_information[2]
    # Filter rows based on extracted parameters
    filtered_df  = df[(df['Product_ID'] == slc_id) &
                  (df['Burst_Number'] == burst_number) &
                  (df['Chip_Number'] == chip_number)]
    # Extract relevant columns
    relevant_columns = ['gps_lon', 'gps_lat', 'ship_burst_row', 'ship_burst_col', 'ship_chip_row', 'ship_chip_col', 'ship_length']
    extracted_data = filtered_df[relevant_columns]
    return extracted_data

def get_ship_locations_in_bursts(chip_information, csv_file):
    df           = pd.read_csv(csv_file)    
    slc_id       = chip_information[0]
    burst_number = chip_information[1]
    # Filter rows based on extracted parameters
    filtered_df  = df[(df['Product_ID'] == slc_id) &
                  (df['Burst_Number'] == burst_number)]
    # Extract relevant columns
    relevant_columns = ['gps_lon', 'gps_lat', 'ship_burst_row', 'ship_burst_col', 'ship_length']
    extracted_data = filtered_df[relevant_columns]
    return extracted_data



def get_slc_id_burst_and_chip_numbers(filename):

    # Extract SLC_ID, burst_number, chip_number, and number_of_ships
    slc_id_match = re.match(r'(S1A_IW_SLC__1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_\w{6}_\w{4})', filename)
    slc_id = slc_id_match.group(1) if slc_id_match else None
    if slc_id ==None:
        slc_id_match = re.match(r'(S1B_IW_SLC__1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_\w{6}_\w{4})', filename)
        slc_id = slc_id_match.group(1) if slc_id_match else None

    burst_number_match = re.search(r'burst#(\d+)', filename)
    burst_number = int(burst_number_match.group(1)) if burst_number_match else None

    chip_number_match = re.search(r'chip#(\d+)', filename)
    chip_number = int(chip_number_match.group(1)) if chip_number_match else None
    return slc_id, burst_number, chip_number





def get_scene_id_and_dataset(SLC_Product_ID):

    """
    Function to find scene_id associated with Product_ID
    and also so determine what dataset it belongs to
    Parameters
    ----------
    SLC_Product_ID : 
    Returns
    -------
    scene_id 

    """
    datasets =['Training', 'Validation', 'Public']
    i=0
    scene_mapping_file_list=['SLC_train.csv', 'SLC_validation.csv', 'SLC_Public_Labels.csv']
    for file in scene_mapping_file_list:
        scene_mapping_file = f'/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/{file}'
        df                 = pd.read_csv(scene_mapping_file)
        # Remove any double underscore
        #SLC_PRODUCT_ID = SLC_Product_ID.replace("__","_")
        filtered_df = df.loc[df["SLC_product_identifier"] == SLC_Product_ID]
        if not filtered_df.empty:
            scene_id = filtered_df["scene_id"].iloc[0]
            dataset  = datasets[i]
            break
            #print(f"scene_id  for SLC_product_identifier {SLC_Product_ID}: {scene_id}")
        else:
            scene_id=''
            dataset=''
            #print(f"ERROR: No SLC match found for scene_id {target_scene_id}. Continuing to next raw file .....")
        i=i+1
    
    return scene_id, dataset



def get_single_raw_burst_from_files(raw_file, slc_annotation_file, burst_to_get,
                                    swath_of_interest=2, polarization='vh'):

##
# Function to return a raw burst based on the SLC burst number
# Requires access to raw data file and SLC annotation file
# Assumes these are in standard RAW and SLC directories
# Takes raw burst as is - does not match to SLC timings!!!
##    

    IW_mode         = swath_of_interest - 1
    raw_filename    = raw_file
    annotation_file = slc_annotation_file
    if not (os.path.exists(raw_filename)):
        print(f'Error: Filename {raw_filename} not found.Exiting!')
        sys.exit(0)
    else:
        raw_iw_data       = get_raw_burst_info(raw_filename)
        print(f' File {raw_filename} found. ')
    
    if os.path.exists(annotation_file):
        print(f'{annotation_file} found ')
    else:
        print(f'ERROR: {annotation_file}  Not found. Exiting')
        sys.exit(0)
    #Check RAW and SLC annotation 
    if not (polarization in annotation_file and polarization in raw_filename and
            (('iw1' in annotation_file and IW_mode==0) or
             ('iw2' in annotation_file and IW_mode==1) or
             ('iw3' in annotation_file and IW_mode==2))):
        print(f'ERROR: IW mode of SLC annotation file does not match expected swath'
              f' or there is a polarization mis-match !')
        print('Exiting function....')
        raise ValueError("ERROR: IW mode or polarization mismatch with SLC file")
    
    # Have SLC annotation file and raw filename
    slc_iw_burst_data = []
    slc_iw_data       = []
    #Get timing positions of slc bursts to enable matching with raw times
    pre_post_burst_information = get_pre_post_mid_times(annotation_file)
    slc_iw_data.append(pre_post_burst_information)
    #Get boundaries for each swath, burst &  grid in [row,col] & GPS coordinates 
    burst_data        = getSwathBurstGridGPSBoundaries(annotation_file)
    slc_iw_burst_data.append(burst_data)
    
    iw_mode = IW_mode
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    slc_range_sampling_freq = float(root.find('.//rangeSamplingRate').text)
    
    for slc_burst in [burst_to_get]:
        burst_of_interest = str(slc_burst)
        matching_burst    = slc_iw_data[0][slc_iw_data[0]['burst_number']==burst_of_interest]
        if not matching_burst.empty:
            index_of_burst  = find_raw_start_burst(raw_iw_data[iw_mode], matching_burst)
            selected_burst  = index_of_burst
            l0file          = sentinel1decoder.Level0File(raw_filename)
            raw_burst_info  = l0file.get_burst_metadata(selected_burst)
            radar_data      = l0file.get_burst_data(index_of_burst)
        else:
            print(f'WARNING: No matching bursts for burst #{burst_of_interest} in {slc_src_file}')
    
    return radar_data, raw_burst_info, l0file



# Assumes a raw burst matching its SLC equivalent is sought
# For this need SLC annotation information to ensure correct temporal overlap
def get_slc_matched_raw_burst_from_file(raw_filename, slc_annotation_file, 
                                        burst_to_get, swath_of_interest, 
                                        polarization
                                        ):

##
# Function to return a raw burst based on the SLC burst number
# Requires access to raw data file and SLC annotation file
# Assumes paths to these included in name
# Partitions raw data to match with SLC timings!!!!
##    
    
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index
    annotation_file = slc_annotation_file
    
    if not (os.path.exists(raw_filename)):
        print(f' Path to {raw_filename} inaccessible: Exiting!')
        raise ValueError(f"Error: Path to {raw_filename} inaccessible: Exiting!")
    else:
        print(f' Path to {raw_filename} found')
        raw_iw_data       = get_raw_burst_info(raw_filename)
    #############################################################################
    if (os.path.exists(annotation_file)):
        print(f'{annotation_file} found')
    else:
        print(f'ERROR: {annotation_file} - Exiting!  ')
        raise ValueError(f"Error: Path to {annotation_file} inaccessible: Exiting!")

    
    #Check RAW and SLC annotation 
    if not (polarization in annotation_file and polarization in raw_filename and
            (('iw1' in annotation_file and IW_mode==0) or
             ('iw2' in annotation_file and IW_mode==1) or
             ('iw3' in annotation_file and IW_mode==2))):
        print(f'ERROR: IW mode of SLC annotation file does not match expected swath'
              f' or there is a polarization mis-match !')
        print('Exiting ....')
        raise ValueError(f"Error: IW mode or polarization mismatch with SLC")    
        return -1
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
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    slc_range_sampling_freq = float(root.find('.//rangeSamplingRate').text)
    
    bursts_not_to_save = []
    #for slc_burst in range(1, len(burst_data[1])+1):
    for slc_burst in [burst_to_get]:
        burst_of_interest = str(slc_burst)
        matching_burst    = slc_iw_data[0][slc_iw_data[0]['burst_number']==burst_of_interest]
        #matching_burst    =  slc_iw_data[['burst_number']==burst_of_interest]
        if not matching_burst.empty:
            index_of_burst    = find_raw_start_burst(raw_iw_data[iw_mode], matching_burst)
        else:
            print(f'WARNING: No matching bursts for burst #{burst_of_interest} in {slc_src_file}')
    
        # Using iloc to find times in slc_iw_data so need to subtract 1 from burst_number to align correctly
        burst_pre_time       = slc_iw_data[0].iloc[slc_burst-1]['pre_mid_time']
        burst_post_time      = slc_iw_data[0].iloc[slc_burst-1]['post_mid_time']
        slc_burst_start_time = slc_iw_data[0].iloc[slc_burst-1]['sensing_time']
    
        # get raw burst selected
        raw_burst_timings    = raw_iw_data[iw_mode].loc[index_of_burst]
        raw_burst_mid_time   = raw_burst_timings['Mid Burst Time']
        raw_burst_start_time = raw_burst_timings['Burst Start Time']
        raw_burst_end_time   = raw_burst_timings['Burst End Time']
        
        # Check that burst_pre_time not greater than half of raw_burst time
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
    
            # number of range samples is end_range - start_range
            number_slc_range_samples     = slc_iw_burst_data[0][1][slc_burst-1][2][1] \
                                         - slc_iw_burst_data[0][1][slc_burst-1][0][1]

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
        
            number_azimuth_slc_lines = slc_iw_burst_data[0][1][slc_burst-1][1][0] \
                                        - slc_iw_burst_data[0][1][slc_burst-1][0][0]
            number_azimuth_raw_lines = end_row - start_row
            number_raw_range_samples = raw_end_range_index - raw_start_range_index +1
            raw_slc_az_line_delta    = (number_azimuth_slc_lines
                                       - number_azimuth_raw_lines)
    
            print(f'Raw burst initial start time = {raw_burst_start_time}')
            print(f'Raw burst new start time = {new_burst_start_time}')
            print(f' Pre-time  = {burst_pre_time}, Post-time = {burst_post_time}')
            print(f' Number Azimuth SLC lines = {number_azimuth_slc_lines}')
            print(f' Number Azimuth Raw lines = {number_azimuth_raw_lines}')
            print(f' Number SLC Range Samples = {number_slc_range_samples}')
            print(f' Number RAW Range Samples = {number_raw_range_samples}')
            print(f' SLC Azimuth Line Time Length = {slc_azimuth_line_time_length}')
            print(f' RAW Azimuth Line Time Length = {raw_azimuth_line_time_length}')
            print(f' RAW Azimuth SubL Time Length = {number_raw_range_samples * 1/raw_range_sampling_freq}')

            print(f' Offset from start  is : {start_index}')
            print(f' New RAW Burst Start Row is : {start_row}')
            print(f' New RAW Burst End Row is   : {end_row}')
            print(f' RAW Range Start = {raw_start_range_index}')
            print(f' RAW Range End = {raw_end_range_index}')
            print(f' RAW Range Sampling Freq  = {raw_range_sampling_freq} Hz')
            print(f' SLC Range Sampling Freq  = {slc_range_sampling_freq} Hz')
        
            if raw_slc_az_line_delta != 0:
                print(f' WARNING: Mismatch in number of azimuth lines in RAW & SLC')
                print(f' \n\n number_azimuth_slc_lines = {number_azimuth_slc_lines}')
                print(f' number_azimuth_raw_lines = {number_azimuth_raw_lines}')
            else:
                print(f'Azimuth lines match')
                radar_data = l0file.get_burst_data(index_of_burst)
                rd         = radar_data[start_row:end_row,raw_start_range_index:raw_end_range_index]
    
        else:
            print(f'ERROR : Burst {slc_burst} of {raw_filename} is not partitioned ')
            print(f' correctly. Either start or end line needs to be amended')
            bursts_not_to_save.append(slc_burst)
        return rd, radar_data, start_row, end_row, raw_start_range_index, raw_end_range_index, raw_burst_info, l0file


def get_raw_burst_with_offset_info(SLC_ID, burst_to_get, swath_of_interest=2, 
                                   polarization='vh', source = 'int'
                                   ):

##
# Function to return a raw burst based on the SLC burst number
# Requires access to raw data file and SLC annotation file
# Assumes these are in standard RAW and SLC directories
# Partitions raw data to match with SLC timings!!!!
##    
    
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index
    
    
    sceneID, scene_dataset = get_scene_id_and_dataset(SLC_ID)
    
    Dataset = 'Train' if scene_dataset == 'Training' else scene_dataset
        
        
    if source == 'ext':
        raw_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_data'
        slc_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_data'
    elif source == 'int':    
        raw_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/raw_data'
        slc_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/slc_data'     
    elif source == 'Spain':
        raw_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/raw_data'
        slc_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/slc_data'
    else:
        print(f'Incorrect Source given. Must be "ext", "int" or "Spain"')
        sys.exit()
        
    
    
    print(f'Scene _ID : {sceneID}')
    print(f'Dataset whence scene originates: {scene_dataset}')
    #raw_path = "/home/joe/Work/Projects/RSAR/Test/SampleData/Public/raw_data"
    #slc_path = "/home/joe/Work/Projects/RSAR/Test/SampleData/Public/slc_data"
    
    
    if scene_dataset == 'Public':
        # Get SLC file to match raw based on mapping file
        scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_Public_Labels.csv'
    elif scene_dataset == 'Validation':
        scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_validation.csv'
    elif scene_dataset == 'Training':
       scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv'
    else:
        print(f' No matching sceneID found for {SLC_ID}')
        print(f' Check sceneID is correct. Exiting.................')
        exit(0)
    
    save_path ='/home/joe/tmp/'
    mapping_df         = pd.read_csv(scene_mapping_file)
    
    if not (os.path.exists(raw_path)):
        print(f' Path to {raw_path} inaccessible: Exiting!')
        sys.exit()
    else:
        print(f' Path to {raw_path} found')
        raw_directories = f'{raw_path}/{sceneID}'
    
    
    
    for raw_directory in [raw_directories]:
        raw_root_file_path   = f'{raw_directory}/RAW/'
        if os.listdir(raw_root_file_path) and contains_SAFE_file(raw_root_file_path):
            raw_safe_dir  = os.listdir(raw_root_file_path)
            raw_src_path  = f'{raw_root_file_path}{raw_safe_dir[0]}/'
            raw_filenames = glob.glob(f'{raw_src_path}*{polarization}*.dat')
            # Need to sort filenames to access correct .dat file
            raw_filenames.sort(reverse=True)
            raw_filename = raw_filenames[0]
            if not (os.path.exists(raw_filename[0])):
                print(f'Error: Filename {raw_filename} not found. Continuing to next raw file ....')
                continue
            else:
                raw_iw_data       = get_raw_burst_info(raw_filename)
                print(f' File {raw_filename} found. ')
        else:
            print(f'Error: Raw file path {raw_root_file_path} not found or does not contain .SAFE directory. Continuing to next raw file ......')
            continue
    
        ########################################################################### 
        ### Get matching SLC file
        target_scene_id     = sceneID
        #target_scene_id     = raw_directory
        #Check if target scene exists in dataframe
        filtered_mapping_df = mapping_df.loc[mapping_df["scene_id"] == target_scene_id]
        if not filtered_mapping_df.empty:
            slc_id = filtered_mapping_df["SLC_product_identifier"].iloc[0]
            print(f"SLC_product_identifier for scene_id {target_scene_id}: {slc_id}")
        else:
             print(f"ERROR: No SLC match found for scene_id {target_scene_id}. Continuing to next raw file .....")
             continue
    
        SLC_root_file_path = f'{slc_path}/{slc_id}'
        SLC_root_file_path = SLC_root_file_path.replace("__","_")
        #p1,p2,p3,p4,p5    = getSLCFilesFromRoot(, swath_of_interest)
        slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
        slc_src_file      = slc_file_info[0]
        annotation_file   = slc_file_info[1]
        annotation_path   = slc_file_info[2]
        SLC_fnames        = slc_file_info[3]
        SAFE_dir          = slc_file_info[4]
        get_slc_info_status = slc_file_info[5]
    
        if get_slc_info_status != 1:
            print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
            continue
    
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
            continue
            #exit(0)
    
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
        #for slc_burst in range(1, len(burst_data[1])+1):
        for slc_burst in [burst_to_get]:
            burst_of_interest = str(slc_burst)
            matching_burst    = slc_iw_data[0][slc_iw_data[0]['burst_number']==burst_of_interest]
            #matching_burst    =  slc_iw_data[['burst_number']==burst_of_interest]
            if not matching_burst.empty:
                index_of_burst    = find_raw_start_burst(raw_iw_data[iw_mode], matching_burst)
            else:
                print(f'WARNING: No matching bursts for burst #{burst_of_interest} in {slc_src_file}')
        
            # Using iloc to find times in slc_iw_data so need to subtract 1 from burst_number to align correctly
            burst_pre_time       = slc_iw_data[0].iloc[slc_burst-1]['pre_mid_time']
            burst_post_time      = slc_iw_data[0].iloc[slc_burst-1]['post_mid_time']
            slc_burst_start_time = slc_iw_data[0].iloc[slc_burst-1]['sensing_time']
        
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
        
                # number of range samples is end_range - start_range
                number_slc_range_samples     = slc_iw_burst_data[0][1][slc_burst-1][2][1] \
                                             - slc_iw_burst_data[0][1][slc_burst-1][0][1]

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
        
                number_azimuth_slc_lines = slc_iw_burst_data[0][1][slc_burst-1][1][0] \
                                            - slc_iw_burst_data[0][1][slc_burst-1][0][0]
                number_azimuth_raw_lines = end_row - start_row
                number_raw_range_samples = raw_end_range_index - raw_start_range_index +1
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
                print(f' RAW Azimuth SubL Time Length = {number_raw_range_samples * 1/raw_range_sampling_freq}')

                print(f' Offset from start  is : {start_index}')
                print(f' New RAW Burst Start Row is : {start_row}')
                print(f' New RAW Burst End Row is   : {end_row}')
                print(f' RAW Range Start = {raw_start_range_index}')
                print(f' RAW Range End = {raw_end_range_index}')
                print(f' RAW Range Sampling Freq  = {raw_range_sampling_freq} Hz')
                print(f' SLC Range Sampling Freq  = {slc_range_sampling_freq} Hz')
        
                if raw_slc_az_line_delta != 0:
                    print(f' WARNING: Mismatch in number of azimuth lines in RAW & SLC')
                    print(f' \n\n number_azimuth_slc_lines = {number_azimuth_slc_lines}')
                    print(f' number_azimuth_raw_lines = {number_azimuth_raw_lines}')
                else:
                    print(f'Azimuth lines match')
                    radar_data = l0file.get_burst_data(index_of_burst)
                    rd         = radar_data[start_row:end_row,raw_start_range_index:raw_end_range_index]
                    #np.save(f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_real_burst#{slc_burst}",np.real(rd).astype(np.float64))
                    #np.save(f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_imag_burst#{slc_burst}",np.imag(rd).astype(np.float64))
        
        
            else:
                print(f'ERROR : Burst {slc_burst} of {raw_filename} is not partitioned correctly. Either start or end line needs to be amended')
                bursts_not_to_save.append(slc_burst)
    return rd, radar_data, start_row, end_row, raw_start_range_index, raw_end_range_index, raw_burst_info, l0file



def get_raw_burst(SLC_ID, burst_to_get, swath_of_interest=2, polarization='vh'):

##
# Function to return a raw burst based on the SLC burst number
# Requires access to raw data file and SLC annotation file
# Assumes these are in standard RAW and SLC directories
# Partitions raw data to match with SLC timings!!!!
##    
    
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index
    
    
    sceneID, scene_dataset = get_scene_id_and_dataset(SLC_ID)
    print(f'Scene _ID : {sceneID}')
    print(f'Dataset whence scene originates: {scene_dataset}')
    
    
    if scene_dataset == 'Public':
        # Get SLC file to match raw based on mapping file
        scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_Public_Labels.csv'
        raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"  # Note: This is trainng data
        slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"  # Note: This is trainng data
    elif scene_dataset == 'Validation':
        scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_validation.csv'
        raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"  # Note: This is trainng data
        slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"  # Note: This is trainng data
    elif scene_dataset == 'Training':
       scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv'
       raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
       slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"
    else:
        print(f' No matching sceneID found for {SLC_ID}')
        print(f' Check sceneID is correct. Exiting.................')
        exit(0)
    
    save_path ='/home/joe/Misc/tempTest/Data/'
    mapping_df         = pd.read_csv(scene_mapping_file)
    
    if not (os.path.exists(raw_path)):
        print(f' Path to {raw_path} inaccessible: Exiting!')
        sys.exit()
    else:
        print(f' Path to {raw_path} found')
        raw_directories = f'{raw_path}/{sceneID}'
    
    
    
    for raw_directory in [raw_directories]:
        raw_root_file_path   = f'{raw_directory}/RAW/'
        if os.listdir(raw_root_file_path) and contains_SAFE_file(raw_root_file_path):
            raw_safe_dir  = os.listdir(raw_root_file_path)
            raw_src_path  = f'{raw_root_file_path}{raw_safe_dir[0]}/'
            raw_filenames = glob.glob(f'{raw_src_path}*{polarization}*.dat')
            # Need to sort filenames to access correct .dat file
            raw_filenames.sort(reverse=True)
            raw_filename = raw_filenames[0]
            if not (os.path.exists(raw_filename[0])):
                print(f'Error: Filename {raw_filename} not found. Continuing to next raw file ....')
                continue
            else:
                raw_iw_data       = get_raw_burst_info(raw_filename)
                print(f' File {raw_filename} found. ')
        else:
            print(f'Error: Raw file path {raw_root_file_path} not found or does not contain .SAFE directory. Continuing to next raw file ......')
            continue
    
        ########################################################################### 
        ### Get matching SLC file
        target_scene_id     = sceneID
        #target_scene_id     = raw_directory
        #Check if target scene exists in dataframe
        filtered_mapping_df = mapping_df.loc[mapping_df["scene_id"] == target_scene_id]
        if not filtered_mapping_df.empty:
            slc_id = filtered_mapping_df["SLC_product_identifier"].iloc[0]
            print(f"SLC_product_identifier for scene_id {target_scene_id}: {slc_id}")
        else:
             print(f"ERROR: No SLC match found for scene_id {target_scene_id}. Continuing to next raw file .....")
             continue
    
        SLC_root_file_path = f'{slc_path}/{slc_id}'
        SLC_root_file_path = SLC_root_file_path.replace("__","_")
        #p1,p2,p3,p4,p5    = getSLCFilesFromRoot(, swath_of_interest)
        slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
        slc_src_file      = slc_file_info[0]
        annotation_file   = slc_file_info[1]
        annotation_path   = slc_file_info[2]
        SLC_fnames        = slc_file_info[3]
        SAFE_dir          = slc_file_info[4]
        get_slc_info_status = slc_file_info[5]
    
        if get_slc_info_status != 1:
            print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
            continue
    
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
            continue
            #exit(0)
    
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
        #for slc_burst in range(1, len(burst_data[1])+1):
        for slc_burst in [burst_to_get]:
            burst_of_interest = str(slc_burst)
            matching_burst    = slc_iw_data[0][slc_iw_data[0]['burst_number']==burst_of_interest]
            #matching_burst    =  slc_iw_data[['burst_number']==burst_of_interest]
            if not matching_burst.empty:
                index_of_burst    = find_raw_start_burst(raw_iw_data[iw_mode], matching_burst)
            else:
                print(f'WARNING: No matching bursts for burst #{burst_of_interest} in {slc_src_file}')
        
            # Using iloc to find times in slc_iw_data so need to subtract 1 from burst_number to align correctly
            burst_pre_time       = slc_iw_data[0].iloc[slc_burst-1]['pre_mid_time']
            burst_post_time      = slc_iw_data[0].iloc[slc_burst-1]['post_mid_time']
            slc_burst_start_time = slc_iw_data[0].iloc[slc_burst-1]['sensing_time']
        
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
        
                # number of range samples is end_range - start_range
                number_slc_range_samples     = slc_iw_burst_data[0][1][slc_burst-1][2][1] \
                                             - slc_iw_burst_data[0][1][slc_burst-1][0][1]

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
        
                number_azimuth_slc_lines = slc_iw_burst_data[0][1][slc_burst-1][1][0] \
                                            - slc_iw_burst_data[0][1][slc_burst-1][0][0]
                number_azimuth_raw_lines = end_row - start_row
                number_raw_range_samples = raw_end_range_index - raw_start_range_index +1
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
                print(f' RAW Azimuth SubL Time Length = {number_raw_range_samples * 1/raw_range_sampling_freq}')

                print(f' Offset from start  is : {start_index}')
                print(f' New RAW Burst Start Row is : {start_row}')
                print(f' New RAW Burst End Row is   : {end_row}')
                print(f' RAW Range Start = {raw_start_range_index}')
                print(f' RAW Range End = {raw_end_range_index}')
                print(f' RAW Range Sampling Freq  = {raw_range_sampling_freq} Hz')
                print(f' SLC Range Sampling Freq  = {slc_range_sampling_freq} Hz')
        
                if raw_slc_az_line_delta != 0:
                    print(f' WARNING: Mismatch in number of azimuth lines in RAW & SLC')
                    print(f' \n\n number_azimuth_slc_lines = {number_azimuth_slc_lines}')
                    print(f' number_azimuth_raw_lines = {number_azimuth_raw_lines}')
                else:
                    print(f'Azimuth lines match')
                    radar_data = l0file.get_burst_data(index_of_burst)
                    rd         = radar_data[start_row:end_row,raw_start_range_index:raw_end_range_index]
                    np.save(f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_real_burst#{slc_burst}",np.real(rd).astype(np.float64))
                    np.save(f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_imag_burst#{slc_burst}",np.imag(rd).astype(np.float64))
        
        
            else:
                print(f'ERROR : Burst {slc_burst} of {raw_filename} is not partitioned correctly. Either start or end line needs to be amended')
                bursts_not_to_save.append(slc_burst)
    return rd


def plot_raw_burst(burst_data, SLC_ID, burst_number, mode = 'real'):
    if mode == 'real':
        data =np.real(burst_data)
    elif mode == 'imag':
        data = np.imag(burst_data)
    elif mode == 'amplitude':
        data = np.abs(burst_data)
    
    fig1, ax1 = plt.subplots(1,1)
    ax1.imshow(data, vmin=data.min(),vmax = data.max(), aspect = 'auto')
    ax1.set_title(f'RAW {mode} Data ')
    fig1.suptitle(f'{SLC_ID}_Burst#{burst_number}', y=0.95)
    plt.show()
    return fig1, ax1
 



def extract_product_burst_and_chip_information(file):
    """

    Parameters
    ----------
    file : Function to parse filename of a chip to obtain its root
           SLC_ID and the burst it belongs to
    Returns
    -------
    product_id : SLC_PRODUCT_ID
    burst_number : 
    chip_number : 

    """
    if not os.path.exists(file):
        print(f'{file} not found ....')
    
    else:
        filename= os.path.basename(file)
        parts = filename.split("_")
        product_id = "_".join(parts[0:10])
        burst_index = next(i for i, part in enumerate(parts) if "burst#" in part)
        chip_index = next(i for i, part in enumerate(parts) if "chip#" in part)
        burst_number = parts[burst_index].split("#")[1]
        chip_number =parts[chip_index].split("#")[1].split(".")[0]
        return product_id, burst_number, chip_number
    
    
    
def normalise_data(data, range_max, range_min):
    max  = data.max()
    min  = data.min()
    normed_data = ((data - min) / (max - min)) * (range_max - range_min) + range_min
    #os.remove(file)
    return normed_data
    

def normalise(file, range_max, range_min):
    data = np.load(file)
    normed_data=normalise_data(data, range_max, range_min)
    root        = file.split('.')[0]
    #dir_path    = os.path.dirname(file)
    new_filename = f'{root}_N.npy'
    np.save(new_filename, normed_data)

    return normed_data

def apply_clahe_to_data(data, clip_limit=2.0, tile_grid_size=(8,8)):
    data_8bit = ((data +1)/2 *255).astype(np.uint8)
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    clahe_data = clahe.apply(data_8bit)
    #np.save(clahe_fname, clahe_data)
    return clahe_data

###################################################################################
def get_slc_amplitude_burst(SLC_ID, burst_number, swath_of_interest=2, polarization ='vh'):
    """
    Function to return single burst of amplitude data
    NOTE: Very similar to get_slc_burst_data_with_ships_tagged function but has 
          some subtle differences (so subtle I've forgotten what they are!')

    Parameters
    ----------
    SLC_ID : TYPE
        DESCRIPTION.
    burst_to_plot : TYPE
        DESCRIPTION.
    swath_of_interest : TYPE, optional
        DESCRIPTION. The default is 2.
    polarization : TYPE, optional
        DESCRIPTION. The default is 'vh'.

    Returns
    -------
    None.

    """

    #swath_of_interest = 2
    #polarization      ='vh'
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index


    sceneID, scene_dataset = get_scene_id_and_dataset(SLC_ID)
    print(f'Scene _ID : {sceneID}')
    print(f'Dataset whence scene originates: {scene_dataset}')


    if scene_dataset == 'Public':
        # Get SLC file to match raw based on mapping file
        scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_Public_Labels.csv'
        slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_Public"
    elif scene_dataset == 'Validation':
        scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_validation.csv'
        slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_Validation"
    elif scene_dataset == 'Training':
        scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_train.csv'
        slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_TrainingData"
    else:
        print(f' No matching sceneID found for {SLC_ID}')
        print(f' Check sceneID is correct. Exiting.................')
        exit(0)

    save_path ='/mnt/smbshare/RAWSARAI_Data/Copernicus/Sandbox/Dump'
    mapping_df         = pd.read_csv(scene_mapping_file)

    if not (os.path.exists(slc_path)):
        print(f' Path to {slc_path} inaccessible: Exiting!')
        sys.exit()
    else:
        print(f' Path to {slc_path} found')

    ### Have matching SLC file
    slc_id = SLC_ID
    SLC_root_file_path = f'{slc_path}/{slc_id}'
    SLC_root_file_path = SLC_root_file_path.replace("__","_")
    slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
    slc_src_file      = slc_file_info[0]
    annotation_file   = slc_file_info[1]
    annotation_path   = slc_file_info[2]
    SLC_fnames        = slc_file_info[3]
    SAFE_dir          = slc_file_info[4]
    get_slc_info_status = slc_file_info[5]

    if get_slc_info_status != 1:
        print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
        #continue

    print(f' Annotation file is: {annotation_file}')
    print(f' Annotation_dir  is: {annotation_path}')
    print(f' SLC_fname(s) are  : {SLC_fnames}')
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    Annotation_dir = annotation_path
    SLC_safe_id, _ = os.path.splitext(SAFE_dir)
    label_file     = scene_mapping_file
    #############################################################################
    if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir)
                       and os.path.exists(slc_src_file)):
        print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
    else:
        print(f'ERROR: {annotation_file} or {Annotation_dir} '
              f' or {slc_src_file} Not found. Exiting')
    # Have SLC src and annotation file and raw filename
    ###########################################################################
    burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
    burst_boundaries  = burst_information[1]
    number_of_bursts  = len(burst_boundaries)
    # Choose burst of interest
    if burst_number > number_of_bursts:
        print(f'WARNING: There are only {number_of_bursts} bursts. Burst#{burst_number} does not exist! Choose another!')

    src            = rasterio.open(slc_src_file)
    img            = src.read(1)
    #Make list of slc_burst as list expected by function
    # Need index for burst    
    burst_boundary = burst_boundaries[burst_number - 1]
    burst_img = create_roi_from_polygon(img, burst_boundary)
    return burst_img, number_of_bursts, burst_information



def get_slc_burst_from_files(slc_file, slc_annotation_file, burst_number):
    """
    Function to accept slc .tiff file and annotation .xml file
    Reads annotation file and determines the number of bursts in the file
    and the de-bursted boundaries for each 
    The polarization and swath number are implicit in the filenames
    """
    burst_information = getSwathBurstGridGPSBoundaries(slc_annotation_file)
    burst_boundaries  = burst_information[1]
    number_of_bursts  = len(burst_boundaries)
    # Choose burst of interest
    if burst_number > number_of_bursts:
        print(f'WARNING: There are only {number_of_bursts} bursts. Burst#{burst_number} does not exist! Choose another!')

    src            = rasterio.open(slc_file)
    img            = src.read(1)
    burst_boundary = burst_boundaries[burst_number - 1]
    burst_img      = create_roi_from_polygon(img, burst_boundary)

    return burst_img, number_of_bursts, burst_information



####################################################################################
def get_slc_burst_with_ships(SLC_ID, burst_to_plot, swath_of_interest=2, polarization ='vh'):
    """
    Function to return burst data, burst data with ships
    tagged and the binary mask all as np-arrays
    NOTE: Very similar to get_slc_burst_data_with_ships_tagged function but has 
          some subtle differences (so subtle I've forgotten what they are!')

    Parameters
    ----------
    SLC_ID : TYPE
        DESCRIPTION.
    burst_to_plot : TYPE
        DESCRIPTION.
    swath_of_interest : TYPE, optional
        DESCRIPTION. The default is 2.
    polarization : TYPE, optional
        DESCRIPTION. The default is 'vh'.

    Returns
    -------
    None.

    """
    
    #swath_of_interest = 2
    #polarization      ='vh'
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index


    sceneID, scene_dataset = get_scene_id_and_dataset(SLC_ID)
    print(f'Scene _ID : {sceneID}')
    print(f'Dataset whence scene originates: {scene_dataset}')
    
    
    if scene_dataset == 'Public':
        scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_Public_Labels.csv'
        slc_path = "/home/joe/Work/Projects/RSAR/Test/SampleData/Public/slc_data"
        #slc_path = "/media/joe/Elements/jc_work/Copernicus/SLC_Public"
        #slc_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/SandBox/Test_norming/SLC_files"
    elif scene_dataset == 'Validation':
        scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_validation.csv'
        slc_path = "/media/joe/Elements/jc_work/Copernicus/SLC_Validation"
    elif scene_dataset == 'Training':
        scene_mapping_file = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv'
        slc_path = "/media/joe/Elements/jc_work/Copernicus/SLC_Train"
    else:
        print(f' No matching sceneID found for {SLC_ID}')
        print(f' Check  that sceneID and mapping filenames are correct. Exiting.................')
        sys.exit(0)
    
    save_path ='/home/joe/Misc/tempTest/Data/'
    mapping_df         = pd.read_csv(scene_mapping_file)
    
    if not (os.path.exists(slc_path)):
        print(f' Path to {slc_path} inaccessible: Exiting!')
        sys.exit()
    else:
        print(f' Path to {slc_path} found')


    # Get SLC file to match raw based on mapping file
    #scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_Public_Labels.csv"

    #scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_train.csv'
    #mapping_df         = pd.read_csv(scene_mapping_file)

    
   	########################################################################### 
   	### Have matching SLC file
    slc_id = SLC_ID
    SLC_root_file_path = f'{slc_path}/{slc_id}'
    SLC_root_file_path = SLC_root_file_path.replace("__","_")
    #p1,p2,p3,p4,p5    = getSLCFilesFromRoot(, swath_of_interest)
    slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
    slc_src_file      = slc_file_info[0]
    annotation_file   = slc_file_info[1]
    annotation_path   = slc_file_info[2]
    SLC_fnames        = slc_file_info[3]
    SAFE_dir          = slc_file_info[4]
    get_slc_info_status = slc_file_info[5]
   
    if get_slc_info_status != 1:
        print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
   	    #continue
   
    print(f' Annotation file is: {annotation_file}')
    print(f' Annotation_dir  is: {annotation_path}')
    print(f' SLC_fname(s) are  : {SLC_fnames}')
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    Annotation_dir = annotation_path
    SLC_safe_id, _ = os.path.splitext(SAFE_dir)
    label_file     = scene_mapping_file
    #############################################################################
    if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir)
  					   and os.path.exists(slc_src_file)):
        print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
    else:
  	    print(f'ERROR: {annotation_file} or {Annotation_dir} '
  		  f' or {slc_src_file} Not found. Exiting')
  	   
    # Have SLC src and annotation file and raw filename
    ###########################################################################
  
    ###########################################################################
    # get binary masks for SLC bursts
    ship_coords = getShipCoordinates(label_file, swath_of_interest, SLC_safe_id)
    burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
    IW_gps_boundary           = burst_information[0]
    burst_boundaries          = burst_information[1]
    burst_gps_boundaries      = burst_information[2]
    burst_grid_boundaries     = burst_information[3]
    burst_grid_gps_boundaries = burst_information[4]
    number_of_bursts          = len(burst_boundaries)
    bounding_box_size         = 100
    border                    = 25
    swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
  						 burst_boundaries,
  						 burst_gps_boundaries,
  						 burst_grid_boundaries,
  						 burst_grid_gps_boundaries,
  						 bounding_box_size)
    # Choose burst of interest
    n = len(burst_boundaries)
    burst_counts = [burst_indices.count(i) for i in range(0, n)]
    # Print the table
    print("Burst Number:\t" ,  "\t".join(map(str, range(1, n + 1))))
    print("#Ships:      \t",  "\t".join(map(str, burst_counts)))
    burst_of_interest = burst_counts.index(max(burst_counts)) + 1   
    #slc_burst         = burst_of_interest
    slc_burst         = burst_to_plot
   
  	
    if os.path.exists(slc_src_file):
        print(f"{SLC_fnames[swath_of_interest_index]} found")
    else:
        print(f"ERROR: {slc_src_file} Not found. Exiting")
        exit(0)
    src            = rasterio.open(slc_src_file)
    img            = src.read(1)
    #Make list of slc_burst as list expected by function
    bursts_to_plot = [slc_burst]
    output=get_chips_and_masks(img, SLC_safe_id,
  				 bursts_to_plot, burst_boundaries,
  				 np.array(swath_bounding_boxes),
  				 burst_indices, bounding_box_size, border
  				)
    # Need index for burst    
    burst_boundary = burst_boundaries[slc_burst - 1]
    burst_img = create_roi_from_polygon(img, burst_boundary) 
    intensity_burst_image = np.square(np.abs(burst_img))
    burst_img_with_tagged_ships = output[2]
    burst_binary_ground_truth   = output[3]
    return intensity_burst_image, burst_img_with_tagged_ships, \
          burst_binary_ground_truth       


def resample_slc_burst(SLC_ID, path_to_slc, burst_number, scale_factor):
    a,b,c = get_slc_burst_with_ships(SLC_ID, burst_number)
    burst_data = a
    burst_data_with_ships = b
    burst_binary = c
    # Re-sample burst data to match RAW size 
    if scale_factor != 1:
       scale_factor          = int(scale_factor * burst_data.shape[1])
       burst_data            = resample(burst_data,scale_factor)
       burst_data_with_ships = resample(burst_data_with_ships,scale_factor)
       burst_binary          = resample(burst_binary,scale_factor)
       
    return burst_data, burst_data_with_ships, burst_binary




def get_slc_chip_and_bursts(SLC_ID, burst_number, chip_number, scale_factor = 1, chip_size =[512, 3072]):
    """
    Parameters
    ----------
    SLC_ID : TYPE
        DESCRIPTION.
    burst_number : TYPE
        DESCRIPTION.
    chip_number : TYPE
        DESCRIPTION.
    scale_factor : resamplng factor for SLC to RAW bursts
    chip_size : TYPE, optional
        DESCRIPTION. The default is [512, 3072].

    Returns
    -------
    chip_data  : data for chip of interest
    burst_data : burst image from whence chip derived
    burst_data_with_ships : burst image with ships tagged (if any)
    burst_binary : binary image of burst

    """
    a,b,c = get_slc_burst_with_ships(SLC_ID, burst_number)
    burst_data = a             #Intensity data
    burst_data_with_ships = b  #Amplitude data
    burst_binary = c
    # Re-sample burst data to match RAW size 
    print(f'#### In get_slc_chips_and_bursts   : scale_factor  = {scale_factor}')
    if scale_factor != 1:
       scale_factor          = int(scale_factor * burst_data.shape[1])
       print(f' #### scale_factor now = : {scale_factor}')
       print(f' #### burst_data.shape[1] = {burst_data.shape[1]}')
       burst_data            = resample(burst_data,scale_factor)
       burst_data_with_ships = resample(burst_data_with_ships,scale_factor)
       burst_binary          = resample(burst_binary,scale_factor)

    # If returning chip data with ship bounding boxes convert to Intensity
    intensity_burst_data_with_ships = np.square(burst_data_with_ships)
    chip_data = get_chip_data(intensity_burst_data_with_ships, chip_number, chip_size)
    return chip_data, burst_data, burst_data_with_ships, burst_binary


 
def plot_chip(chip, SLC_safe_id, slc_burst, chip_number):   

    fig1, ax1 = plt.subplots(1,1)
    ax1.imshow(chip, norm=colors.LogNorm(vmin=10,vmax = 3000), aspect = 'auto')
    ax1.set_title(f'SLC Amplitude: Burst#{slc_burst} Chip#{chip_number}')
    fig1.suptitle(f'{SLC_safe_id}', y=0.95)
    plt.show()
    return fig1, ax1

def plot_blobs_in_figure(ax_handle, blobs):
    #plt.imshow(image, cmap='gray', aspect = 'auto')
    for blob in blobs:
        contours = np.argwhere(blob)
        y, x = zip(*contours)
        ax_handle.scatter(x, y, color='red', s=1)
    ax_handle.set_title(f'Number of Blobs Detected = {len(blobs)}')
    #plt.show()


def plot_blobs(image, blobs):
    plt.imshow(image, cmap='gray', aspect = 'auto')
    for blob in blobs:
        contours = np.argwhere(blob)
        y, x = zip(*contours)
        plt.scatter(x, y, color='red', s=1)
    plt.title(f'Number of Blobs: {len(blobs)}')
    plt.show()

def plot_contiguous_blobs(image, blobs):
    mask = np.zeros_like(image, dtype=np.uint8)
    for blob in blobs:
        contours = np.argwhere(blob)
        y, x = zip(*contours)
        mask[y, x] = 255

    # Display the mask
    plt.imshow(mask, cmap='gray', aspect='auto')
    plt.title(f'Number of Blobs: {len(blobs)}')
    plt.show()










def cfar_detector(slc_data, num_train, num_guard, pfa):
    """
    CFAR detector for SLC SAR data with Gaussian-distributed clutter.
    
    Parameters:
    slc_data (np.array): SLC SAR data in intensity domain
    num_train (int): Number of training cells
    num_guard (int): Number of guard cells
    pfa (float): Desired probability of false alarm
    
    Returns:
    np.array: Binary detection map
    """
    # Convert SLC data to intensity
    #intensity = np.abs(slc_data)**2
    # SLC chip data already in intensity domain
    intensity = slc_data
    
    # Calculate the number of cells on each side
    num_side = num_train + num_guard
    
    # Pad the intensity array
    padded_intensity = np.pad(intensity, ((num_side, num_side), (num_side, num_side)), mode='reflect')
    
    # Initialize the detection map
    detection_map = np.zeros_like(intensity, dtype=bool)
    
    # Calculate the threshold factor - if assuming normally distributed
    #threshold_factor = norm.ppf(1 - pfa)
    
    # Perform CFAR detection
    for i in range(num_side, padded_intensity.shape[0] - num_side):
        for j in range(num_side, padded_intensity.shape[1] - num_side):
            # Extract the cell under test (CUT)
            cut = padded_intensity[i, j]
            
            # Extract the training cells
            training_cells = np.concatenate([
                padded_intensity[i-num_side:i+num_side+1, j-num_side:j-num_guard-1].flatten(),
                padded_intensity[i-num_side:i+num_side+1, j+num_guard+1:j+num_side+1].flatten(),
                padded_intensity[i-num_side:i-num_guard-1, j-num_guard:j+num_guard+1].flatten(),
                padded_intensity[i+num_guard+1:i+num_side+1, j-num_guard:j+num_guard+1].flatten()
            ])
            
            # Calculate the mean and standard deviation of the training cells
            mean = np.mean(training_cells)
            std  = np.std(training_cells)
            var  = std*std

            #print(f'mean = {mean} : std_dev = {std} : variance = {var}')
            
            # Calculate the adaptive threshold for normally distributed clutter
            # threshold = mean + threshold_factor * std
            # Calculate the adaptive threshold for gamma distributed clutter
            """
            equiv_zero = 1e-12
            if abs(mean) < equiv_zero  or abs(std) < equiv_zero:
                threshold = 0
            else: 
                alpha = mean*mean/(var)
                beta  = var/mean
                threshold = gamma.ppf(1-pfa, alpha, beta)
                #threshold = gamma.ppf(pfa, alpha, beta) #Remember to fix back to 1-pfa after testing
            """
            alpha = mean*mean/(var)
            beta  = var/mean
            # For gamma distribution threshold = mean*threshold_multiplier
            threshold = mean*gamma.ppf(1-pfa, alpha, beta)
            # Perform detection
            detection_map[i-num_side, j-num_side] = cut > threshold
    
    return detection_map



import numpy as np
from scipy.signal import convolve2d
from scipy.stats import gamma

def create_cfar_mask(num_side, num_guard):
    """Create 2D mask for training cells (1s) and guard/CUT (0s)"""
    mask = np.ones((2*num_side+1, 2*num_side+1), dtype=float)
    guard_region = slice(num_side-num_guard, num_side+num_guard+1)
    mask[guard_region, guard_region] = 0
    return mask

def optimized_cfar_detector(slc_data, num_train, num_guard, pfa):
    """
    Vectorized CFAR detector using convolution-based statistics
    """
    num_side = num_train + num_guard
    mask = create_cfar_mask(num_side, num_guard)
    
    # Pad array with reflected values
    padded = np.pad(slc_data, num_side, mode='reflect')
    
    # Calculate training cell statistics using 2D convolution
    kernel = mask / mask.sum()  # Normalize for mean calculation
    sum_train = convolve2d(padded, mask, mode='same')
    sum_sq_train = convolve2d(padded**2, mask, mode='same')
    
    # Crop to original size
    valid_region = (slice(num_side, -num_side), slice(num_side, -num_side))
    sum_train = sum_train[valid_region]
    sum_sq_train = sum_sq_train[valid_region]
    
    # Calculate statistics
    n = mask.sum()
    mean = sum_train / n
    var = (sum_sq_train / n) - mean**2
    
    # Handle edge cases
    var = np.maximum(var, 1e-12)
    mean = np.maximum(mean, 1e-12)
    
    # Gamma parameters and threshold calculation
    alpha = mean**2 / var
    beta = var / mean
    threshold = gamma.ppf(1 - pfa, alpha, scale=beta)
    #threshold = mean * gamma.ppf(1 - pfa, alpha, scale=beta)
    
    return slc_data > threshold


def plot_hist_with_fit_on_ax(ax, data, bins=100, outlier_threshold=10, 
                             dist_type='gaussian', title=' '
                             ):
    if outlier_threshold == 0:
        values = data.flatten()
    else:
        values = exclude_high_outliers(data, outlier_threshold)
    bin_width = (values.max() - values.min()) / bins
    x = np.linspace(values.min(), values.max(), 1000)
    label  = (f'Data Histogram\n Mean = {np.mean(values):.2f}, \n sigma = {np.std(values):.2f}')
    count, bins_edges, _ = ax.hist(values, bins=bins, alpha=0.6, rwidth=0.6, 
                                   color='g', edgecolor='black', label=label)
    

    if dist_type == 'gaussian':
        mu = np.mean(values)
        sigma = np.std(values)
        y = norm.pdf(x, mu, sigma) * len(values) * bin_width
        label = f'Gaussian Fit\n($\mu$={mu:.2f},\n $\sigma$={sigma:.2f})'
        ax.plot(x, y, 'r-', lw=2, label=label)
    
    elif dist_type == 'gamma':
        # Fit gamma parameters to the data - sample data if >1e4
        if len(values)>10000:
            sample_values=np.random.choice(values,10000,replace=False)
        shape, loc, scale = gamma.fit(sample_values)
        y = gamma.pdf(x, shape, loc, scale) * len(values) * bin_width
        label = f'Gamma Fit\n(shape={shape:.2f}\n loc={loc:.2f}\n scale={scale:.2f})'
        ax.plot(x, y, 'b-', lw=2, label=label)
        
    elif dist_type == 'weibull':
        # Fit Weibull parameters to the data
        if len(values)>10000:
            sample_values=np.random.choice(values,10000,replace=False)
        c, loc, scale = weibull_min.fit(sample_values)
        y = weibull_min.pdf(x, c, loc, scale) * len(values) * bin_width
        label = f'Weibull Fit\n(c={c:.2f} \n loc={loc:.2f} \n scale={scale:.2f})'
        ax.plot(x, y, 'm-', lw=2, label=label)

    
    
    # Enable scientific notation (usually default for large numbers)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Get the scalar formatter used by y axis
    formatter = ax.yaxis.get_major_formatter()
    
    # Get the order of magnitude (exponent), e.g. 6 for 1e6
    exponent = int(np.log10(count.max()))
    #exponent = formatter.orderOfMagnitude
    #print(f'Exponent = {exponent}')
    
    # Construct new y-label with the scale factor
    if exponent != 0:
        new_ylabel = f'Count x $10^{exponent}$'
    else:
        new_ylabel = 'Count'
    
    # Disable offset text (the 10^6 label in corner)
    ax.get_yaxis().get_offset_text().set_visible(False)
        
    # similar for gamma, weibull...
    ax.set_title(f'{title}',fontsize=6) 
    ax.set_xlabel('Value',fontsize=6)    
    ax.set_ylabel(new_ylabel, fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.legend(fontsize=5)
    


from scipy.stats import norm, gamma, weibull_min

def plot_hist_with_fit(data, bins=100, outlier_threshold = 10,
                       dist_type='gaussian',title =' '
                       ):
    """
    Plots a histogram (counts) of the flattened 2D array 'data' with 'bins' bins,
    and overlays a fit: Gaussian, Gamma, or Weibull, based on dist_type.
    
    Parameters:
        data (np.ndarray): 2D numpy array of data.
        bins (int): Number of bins for the histogram (default: 100).
        dist_type (str): 'gaussian', 'gamma', or 'weibull'.
    """
    default_fontsize=8
    fig,ax = plt.subplots()
    values = exclude_high_outliers(data,outlier_threshold)
    label  = (f'Data Histogram\n Mean = {np.mean(values):.2f}, sigma = {np.std(values):.2f}')

    count, bins_edges, _ = plt.hist(values, bins=bins, alpha=0.6, color='g', 
                                    edgecolor='black', label=label
                                    )
    bin_width = bins_edges[1] - bins_edges[0]
    x = np.linspace(bins_edges[0], bins_edges[-1], 1000)
    
    # Overlay the chosen distribution
    if dist_type == 'gaussian':
        mu = np.mean(values)
        sigma = np.std(values)
        y = norm.pdf(x, mu, sigma) * len(values) * bin_width
        label = f'Gaussian Fit\n($\mu$={mu:.2f}, $\sigma$={sigma:.2f})'
        
        plt.plot(x, y, 'r-', lw=2, label=label)
    elif dist_type == 'gamma':
        # Fit gamma parameters to the data - sample data if >1e4
        if len(values)>10000:
            sample_values=np.random.choice(values,10000,replace=False)
        shape, loc, scale = gamma.fit(sample_values)
        y = gamma.pdf(x, shape, loc, scale) * len(values) * bin_width
        label = f'Gamma Fit\n(shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f})'
        
        plt.plot(x, y, 'b-', lw=2, label=label)
    elif dist_type == 'weibull':
        # Fit Weibull parameters to the data
        if len(values)>10000:
            sample_values=np.random.choice(values,10000,replace=False)
        c, loc, scale = weibull_min.fit(sample_values)
        y = weibull_min.pdf(x, c, loc, scale) * len(values) * bin_width
        label = f'Weibull Fit\n(c={c:.2f}, loc={loc:.2f}, scale={scale:.2f})'
        

       
        plt.plot(x, y, 'm-', lw=2, label=label)
    else:
        raise ValueError("dist_type must be 'gaussian', 'gamma', or 'weibull'")
    
    plt.title(f'Histogram of {title} with fit',fontsize=default_fontsize+1)
    plt.xlabel('Value',fontsize=default_fontsize)
    plt.ylabel('Count',fontsize=default_fontsize)
    plt.legend(fontsize=default_fontsize)
    plt.show()
    return fig, ax

# Example usage:
# plot_hist_with_fit(your_2d_array, bins=100, dist_type='gamma')

def plot_hist_with_gaussian(data, bins=100):
    """
    Plots a histogram of the flattened 2D array 'data' with 'bins' bins,
    and overlays a Gaussian curve using the data's mean and standard deviation.
    
    Parameters:
        data (np.ndarray): 2D numpy array of data.
        bins (int): Number of bins for the histogram (default: 100).
    """
    # Flatten the data to 1D
    values = exclude_high_outliers(data,2)
    
    # Compute mean and standard deviation
    mu = np.mean(values)
    sigma = np.std(values)
    
    # Plot histogram
    count, bins_edges, _ = plt.hist(values, bins=bins, density=True, alpha=0.6, color='g', label='Data Histogram')
    
    # Plot Gaussian curve
    x = np.linspace(bins_edges[0], bins_edges[-1], 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Gaussian Fit')
    
    plt.title('Histogram with Gaussian Overlay')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def exclude_high_outliers(data, threshold=2):
    """
    Exclude values that are more than 'threshold' standard deviations above the mean.
    Returns a 1D array of filtered values.
    """
    values = data.ravel()
    mu = np.mean(values)
    sigma = np.std(values)
    # Keep only values less than or equal to mean + threshold*std
    filtered = values[np.abs(values) <= mu + threshold * sigma]
    return filtered



import numpy as np
from numba import njit, prange
#from scipy.stats import gamma
#from numba-stats import gamma as numba_gamma
@njit(parallel=True, fastmath=True)
def numba_cfar(slc_data, num_train, num_guard, pfa):
    """
    Numba-optimized 2D CFAR detector with gamma-distributed clutter
    """
    rows, cols = slc_data.shape
    num_side = num_train + num_guard
    pad_size = num_side
    padded = np.pad(slc_data, pad_size, mode='reflect')
    
    # Precompute squared values for variance calculation
    padded_sq = padded ** 2
    
    # Initialize outputs
    detection_map = np.zeros((rows, cols), dtype=np.bool_)
    thresholds = np.zeros((rows, cols), dtype=np.float32)
    
    # Cache distance calculations
    max_dist = int(np.ceil(np.sqrt(2*(num_side)**2)))
    dist_mask = np.zeros((2*num_side+1, 2*num_side+1), dtype=np.bool_)
    for i in prange(-num_side, num_side+1):
        for j in prange(-num_side, num_side+1):
            dist = np.sqrt(i**2 + j**2)
            dist_mask[i+num_side, j+num_side] = (abs(i) > num_guard) | (abs(j) > num_guard)
    
    # Main processing loop
    for i in prange(rows):
        for j in prange(cols):
            center_i = i + pad_size
            center_j = j + pad_size
            
            # Extract training cells using precomputed mask
            sum_val = 0.0
            sum_sq = 0.0
            count = 0
            
            for di in range(-num_side, num_side+1):
                for dj in range(-num_side, num_side+1):
                    if dist_mask[di+num_side, dj+num_side]:
                        val = padded[center_i + di, center_j + dj]
                        if not np.isnan(val):
                            sum_val += val
                            sum_sq += val ** 2
                            count += 1
            
            if count < 1:
                thresholds[i, j] = 0.0
                continue
                
            # Compute statistics
            mean = sum_val / count
            var = max((sum_sq / count) - (mean ** 2), 1e-12)
            
            # Gamma threshold calculation
            alpha = (mean ** 2) / var
            beta = var / mean
            #threshold = mean * numba_gamma.ppf(1 - pfa, alpha, scale=beta)
            threshold = numba_gamma.ppf(1 - pfa, alpha, scale=beta)
            
            thresholds[i, j] = threshold
            detection_map[i, j] = padded[center_i, center_j] > threshold
    
    return detection_map, thresholds






def get_slc_burst_data_with_ships(SLC_ID, burst_to_plot, swath_of_interest=2, polarization ='vh'):
    """
    Function to return burst data, burst data with ships
    tagged and the binary mask all as np-arrays

    Parameters
    ----------
    SLC_ID : TYPE
        DESCRIPTION.
    burst_to_plot : TYPE
        DESCRIPTION.
    swath_of_interest : TYPE, optional
        DESCRIPTION. The default is 2.
    polarization : TYPE, optional
        DESCRIPTION. The default is 'vh'.

    Returns
    -------
    None.

    """
    scene_id = get_scene_id(SLC_ID)
    if scene_id:
        print(f'Scene_id is: {scene_id}')
    else:
        print(f'scene_id failed')
        sys.exit()
    #swath_of_interest = 2
    #polarization      ='vh'
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index

    # Get SLC file to match raw based on mapping file
    scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"

    #scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_train.csv'
    mapping_df         = pd.read_csv(scene_mapping_file)

    #raw_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Train"
    #slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_TrainingData"
    #raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
    #slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"
    raw_path = "/media/joe/Elements/jc_work/Copernicus/Public"
    slc_path = "/media/joe/Elements/jc_work/Copernicus/SLC_Public"


    #save_path = "/mnt/smbshare/tempTest/Data/Burst_Training_Data/" plt.subplots
    #save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/raw_burst_training/"
    save_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/"

    if not (os.path.exists(raw_path)):
	    print(f' Path to {raw_path} inaccessible: Exiting!')
	    sys.exit()
    else:
	    print(f' Path to {raw_path} found')

    for raw_directory in [scene_id]:
        raw_root_file_path   = f'{raw_path}/{raw_directory}/RAW/'
        if os.listdir(raw_root_file_path) and contains_SAFE_file(raw_root_file_path):
            raw_safe_dir  = os.listdir(raw_root_file_path)
            raw_src_path  = f'{raw_root_file_path}{raw_safe_dir[0]}/'
            raw_filenames = glob.glob(f'{raw_src_path}*{polarization}*.dat')
            # Need to sort filenames to access correct .dat file
            raw_filenames.sort(reverse=True)
            raw_filename = raw_filenames[0]
            if not (os.path.exists(raw_filename[0])):
                print(f'Error: Filename {raw_filename} not found. Continuing to next raw file ....')
                continue
            else:
                raw_iw_data       = get_raw_burst_info(raw_filename)
                print(f' File {raw_filename} found. ')
        else:
            print(f'Error: Raw file path {raw_root_file_path} not found or does not contain .SAFE directory. Continuing to next raw file ......')
            continue
    	########################################################################### 
    	### Have matching SLC file
        slc_id = SLC_ID
        SLC_root_file_path = f'{slc_path}/{slc_id}'
        SLC_root_file_path = SLC_root_file_path.replace("__","_")
        #p1,p2,p3,p4,p5    = getSLCFilesFromRoot(, swath_of_interest)
        slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
        slc_src_file      = slc_file_info[0]
        annotation_file   = slc_file_info[1]
        annotation_path   = slc_file_info[2]
        SLC_fnames        = slc_file_info[3]
        SAFE_dir          = slc_file_info[4]
        get_slc_info_status = slc_file_info[5]
    
        if get_slc_info_status != 1:
    	    print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
    	    continue
    
        print(f' Annotation file is: {annotation_file}')
        print(f' Annotation_dir  is: {annotation_path}')
        print(f' SLC_fname(s) are  : {SLC_fnames}')
        print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
       	Annotation_dir = annotation_path
       	SLC_safe_id, _ = os.path.splitext(SAFE_dir)
       	label_file     = scene_mapping_file
       	#############################################################################
       	if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir)
       					   and os.path.exists(slc_src_file)):
       	    print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
       	else:
       	    print(f'ERROR: {annotation_file} or {Annotation_dir} '
       		  f' or {slc_src_file} Not found. Exiting')
       	    continue
       	    #exit(0)
       
       	# Have SLC src and annotation file and raw filename
       	###########################################################################
       
       	###########################################################################
       	# get binary masks for SLC bursts
       	ship_coords = getShipCoordinates(label_file, swath_of_interest, SLC_safe_id)
       	burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
       	IW_gps_boundary           = burst_information[0]
       	burst_boundaries          = burst_information[1]
       	burst_gps_boundaries      = burst_information[2]
       	burst_grid_boundaries     = burst_information[3]
       	burst_grid_gps_boundaries = burst_information[4]
       	number_of_bursts          = len(burst_boundaries)
       	bounding_box_size         = 100
       	border                    = 25
       	swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
       						 burst_boundaries,
       						 burst_gps_boundaries,
       						 burst_grid_boundaries,
       						 burst_grid_gps_boundaries,
       						 bounding_box_size)
       	# Choose burst of interest
       	n = len(burst_boundaries)
       	burst_counts = [burst_indices.count(i) for i in range(0, n)]
       	# Print the table
       	print("Burst Number:\t" ,  "\t".join(map(str, range(1, n + 1))))
       	print("#Ships:      \t",  "\t".join(map(str, burst_counts)))
       	burst_of_interest = burst_counts.index(max(burst_counts)) + 1   
       	#slc_burst         = burst_of_interest
       	slc_burst         = burst_to_plot
       	############################################################################
       	# Load real and imaginary components of burst
       	burst_real_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_real_burst#{slc_burst}.npy"
       	burst_imag_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_imag_burst#{slc_burst}.npy"
       	burst_bin_file  = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_binary_mask_burst#{slc_burst}.npy"
       	if os.path.exists(burst_real_file) and os.path.exists(burst_imag_file) and os.path.exists(burst_bin_file):
       	    burst_real = np.load(burst_real_file)
       	    burst_imag = np.load(burst_imag_file)
       	    burst_bin  = np.load(burst_bin_file)
       	    print(f'All burst files found')
       	else:
       	    print(f'ERROR : At least one raw burst data file not found in {save_path}. Exiting')
       	    exit(0)
       	###########################################################################
       
       
       	
       	if os.path.exists(slc_src_file):
       	    print(f"{SLC_fnames[swath_of_interest_index]} found")
       	else:
       	    print(f"ERROR: {slc_src_file} Not found. Exiting")
       	    #exit(0)
       
       	src            = rasterio.open(slc_src_file)
       	img            = src.read(1)
        #Make list of slc_burst as list expected by function
       	bursts_to_plot = [slc_burst]
       	output=get_chips_and_masks(img, SLC_safe_id,
       				 bursts_to_plot, burst_boundaries,
       				 np.array(swath_bounding_boxes),
       				 burst_indices, bounding_box_size, border
       				)
        # Need index for burst    
        burst_boundary = burst_boundaries[slc_burst - 1]
        burst_img = create_roi_from_polygon(img, burst_boundary) 
        intensity_burst_image = np.square(np.abs(burst_img))
        burst_img_with_tagged_ships = output[2]
        burst_binary_ground_truth   = output[3]
        return intensity_burst_image, burst_img_with_tagged_ships, \
               burst_binary_ground_truth       
       

def get_chip_aoi_in_burst(chip_number, chip_size, burst_size):
    """
    Function to get co-ordinates of vertices of  chip within a given burst

    Parameters
    ----------
    chip_number : TYPE
        DESCRIPTION.
    chip_size : TYPE
        DESCRIPTION.
    burst_size : TYPE
        DESCRIPTION.

    Returns
    -------
    chip_aoi : TYPE
        DESCRIPTION.

    """
    chip_width  = chip_size[1]
    chip_height = chip_size[0]
    burst_width = burst_size[1]
    burst_height= burst_size[0]
    
    
    num_chips_per_burst_row = (burst_width -1) // chip_width +1
    row = ((chip_number-1) // num_chips_per_burst_row) + 1
    col_start = ((chip_number- 1) % num_chips_per_burst_row )*chip_width
    if burst_width - col_start < chip_width:
        col_start = burst_width - chip_width 
    col_end = col_start + chip_width -1 
    row_start = (row - 1)*chip_height
    if burst_height - row_start < chip_height:
        row_start = burst_height - chip_height
    row_end = row_start + chip_height -1
    chip_aoi = [[row_start, col_start], [row_end, col_start], 
                [row_end, col_end], [row_start, col_end]]
    
    return chip_aoi

def float_to_int_list(my_tuple_list):
   int_list = [(round(x), round(y)) for x, y in my_tuple_list]
   return int_list


def plot_2d_Capella_array(array, title='Title', _type ='log_normal'):
    fig, ax = plt.subplots()
    if _type == 'log_normal':
        # Assume data detected aleady : detect_data=[np.abs(data)]^2
        dB_data = 10 * np.log10(array) 
        im = ax.imshow(dB_data,
                       vmin = np.percentile(dB_data, 10),
                       vmax = np.percentile(dB_data, 97), 
                       aspect = 'equal',
                       cmap = 'gray',
                      )
        #plt.colorbar(im)
    elif _type == 'linear':
       im = ax.imshow(array,
                      vmin = array.min() + 0.3*(array.max() - array.min()),
                      vmax = 0.9*array.max(),
                      aspect = 'equal'
                     )
    elif _type == 'binary':
       im = ax.imshow(array, cmap='Greys_r' ,aspect = 'equal')

    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()
    return fig, ax



def plot_2d_array(array, title='Title', _type ='log_normal', xlabel='X-axis', ylabel='Y-axis'):
    fig, ax = plt.subplots()
    if _type == 'log_normal':
        im = ax.imshow(array,norm=colors.LogNorm(vmin=10,vmax = 1000000), aspect = 'auto')
        #plt.colorbar(im)
    elif _type == 'linear':
       im = ax.imshow(array,vmin=array.min(), vmax=array.max(), aspect = 'auto')
    elif _type == 'binary':
       im = ax.imshow(array, cmap='Greys_r' ,aspect = 'auto')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    return fig, ax


def plot_2d_array_with_labelled_points(array, points, title='Title', _type ='log_normal'):
    fig, ax = plt.subplots()
    if _type == 'log_normal':
        im = ax.imshow(array,norm=colors.LogNorm(vmin=10,vmax = 1000000), aspect = 'auto')
        #plt.colorbar(im)
    elif _type == 'linear':
       im = ax.imshow(array,vmin=array.min(), vmax=array.max(), aspect = 'auto')
    elif _type == 'binary':
       im = ax.imshow(array, cmap='Greys_r' ,aspect = 'auto')
# --- Added Code to Draw Circles ---
    if points:
        # Unpack the list of (row, col) tuples. 
        # Note: 'col' maps to X-axis, 'row' maps to Y-axis
        rows, cols = zip(*points)
        
        # s=200 sets the size of the circle marker
        # facecolors='none' keeps the inside hollow so you can see the array underneath
        # edgecolors sets the color of the circle boundary
        ax.scatter(cols, rows, s=200, facecolors='none', edgecolors='red', linewidths=2, label='Points')
        ax.legend()
    # ----------------------------------
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()
    return fig, ax


#Add labelled points to a figure
def add_circles_to_figure(ax, locations, radius=25, edge_colour='red', linewidth=2):
# --- Added Code using Circle Patches ---
    # Define a radius size (adjust this number depending on your array dimensions)
    circle_radius = radius 
    
    for row, col in locations:
        # Note: Circle takes center as (X, Y), which corresponds to (col, row)
        circle = patches.Circle((col, row), radius=circle_radius, 
                               edgecolor=edge_colour, facecolor='none', linewidth=2)
        ax.add_patch(circle)



#
# Function to return [row,col] co-ordinates of ships in a burst based
# Returns coords in orginal slc burst and for burst matched with raw sampling
# rate 
def get_ship_row_col(ship_coords, slc_filename, burst_info, burst_number):
    
    
   IW_MODE = next((int(part.replace('iw', '')) 
                    for part in slc_filename.split('-') 
                    if 'iw' in part), None
                   )
   sampling_freq_mul = freq_multipliers[IW_MODE-1]
     
   bounding_box_size         = 100
   border                    = 25
   swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
                                                burst_info[1],
                                                burst_info[2],
                                                burst_info[3],
                                                burst_info[4],
                                                bounding_box_size)
   
   num_bursts   = len(burst_info[1])
   burst_counts = [burst_indices.count(i) for i in range(0, num_bursts)]
   burst_index = burst_number -1
   num_ships_in_burst = burst_counts[burst_index]
   
   if num_ships_in_burst ==0:
       print(f' No ships in burst#{burst_number}!')
       
       
   burst_dims = np.array(burst_info[1][burst_index])
   burst_start_offset = burst_dims[0,0]
   burst_col_offset   = burst_dims[0,1]   ### Check this same as burst_boundary[0][1]
   burst_size=[(burst_dims[1,0]-burst_start_offset),round(burst_dims[2,1]*sampling_freq_mul)]
   #Get ships in each burst
   ship_indices = np.where(np.array(burst_indices) == burst_index)[0]
   
   ship_raw_matched_locations=[]
   ship_non_raw_matched_locations=[]
   # Get ship locations in raw matched slc_file -> return burst number
   for ship_number in ship_indices:
       print(f'Swath Bounding box for ship at index {ship_number} is {swath_bounding_boxes[ship_number]}')
       ship_bounding_box = Polygon(swath_bounding_boxes[ship_number])
       ship_centroid = ship_bounding_box.centroid
       # Shapely uses loc.x amd loc.y not indexing. Note [row,col] of 
       # image format means row == .y and col == .x
       # Have to re-sample in range (column , or .x) direction ????????
       ship_location = [round(ship_centroid.x) - burst_start_offset,
                        round((ship_centroid.y - burst_col_offset) * sampling_freq_mul)
                       ]
       #Get location of ship in matched and non-matched raw-burst [row, col]
       ship_non_raw_matched_burst_row = ship_location[0]
       ship_non_raw_matched_burst_col = round(ship_centroid.y - burst_col_offset)
       #ship_locations.append([ship_burst_row,ship_burst_col])
       #ship_locations.append(ship_location)
       # return ship locations matched to raw and original slc (non-raw-matched)
       ship_raw_matched_locations.append(ship_location)
       ship_non_raw_matched_locations.append([ship_non_raw_matched_burst_row,
                                              ship_non_raw_matched_burst_col
                                              ]
                                             )
   return ship_raw_matched_locations, ship_non_raw_matched_locations, ship_indices
   
 
#
# Function determines ships coordinates within a burst from label file
# Extracts burst and its boundary information from the SLC annotation file 
def plot_burst_and_ships(slc_file, slc_annotation_file, 
                         burst_number, label_file, raw_matched=True
                         ):
    
    SLC_ID = next((part[:-5] for part in slc_file.split('/') 
                             if part.endswith('.SAFE')), None
                  )
    
    IW_MODE = next((int(part.replace('iw', '')) 
                    for part in slc_file.split('-') 
                    if 'iw' in part), None
                   )
    #burst_number = 4
    sampling_freq_mul = freq_multipliers[IW_MODE-1]
    
    
    # Get co-ordinates of ships in SLC file based on unfiltered list of ships in label file
    # If want to discriminate on ship size use a different label file
    ship_data   = getShipCoordinatesLengthConfidence(label_file, 
                                                     IW_MODE, 
                                                     SLC_ID
                                                     )
    ship_coords = ship_data[0]
    
    burst_data,num_bursts, burst_info = get_slc_burst_from_files(slc_file, 
                                                               slc_annotation_file,
                                                               burst_number
                                                               )
    
    ship_info = get_ship_row_col(ship_coords, slc_file, burst_info, burst_number)
   
    ship_row_col_matched     = ship_info[0]
    ship_row_col_non_matched = ship_info[1]
    ship_indices             = ship_info[2]
    
    # get ship locations in burst -> use gps_lon_lat of ships and interpolate
    
    # Need to re-sample SLC burst to same range sampling as raw burst
    scaling_factor    = int(freq_multipliers[IW_MODE -1]*burst_data.shape[1])
    if raw_matched:
       burst_data      = resample(burst_data, scaling_factor)
       ship_locations  = ship_row_col_matched
       print(f'Viewing raw matched SLC burst. Burst size is {burst_data.shape}')
    else:
        ship_locations = ship_row_col_non_matched
        print(f'Viewing non-raw matched SLC burst. Burst size is {burst_data.shape}')
        
    f,ax=plot_2d_array(intensity(burst_data),f'Burst#{burst_number}', 
                        xlabel ='Range', ylabel='Azimuth'
                        )
    f.suptitle(SLC_ID, fontsize=10)
    add_circles_to_figure(ax, ship_locations)
    return f,ax, ship_info, ship_data[0]





# Plot 2D array in location defined by handle ax
def plot_2d_array_on_ax(ax, array, title='Title', _type ='log_normal', 
                        fontsize=default_fontsize, grid_on=0, grid_colour='gray'
                        ):
    #fig, ax = plt.subplots()
    if _type == 'log_normal':
        ax.imshow(array,norm=colors.LogNorm(vmin=10,vmax = 1000000), aspect = 'auto')
     
    elif _type == 'linear':
       im = ax.imshow(array,vmin=array.min(), vmax=array.max(), aspect = 'auto')
    elif _type == 'binary':
       im = ax.imshow(array, cmap='Greys_r' ,aspect = 'auto')
       
    if grid_on:
        # Set major ticks to be at the pixel boundaries
        #ax.set_xticks(np.arange(array.shape[1]/100) - 0.5, minor=True)
        #ax.set_yticks(np.arange(array.shape[0]/1000+1) - 0.5, minor=True)
        # Enable grid on minor ticks to draw gridlines between pixels
        ax.grid(which='major', color=grid_colour, linestyle='-.', linewidth=0.5)
        # Optionally, disable major ticks if you don't want them shown
        ax.tick_params(which='major', bottom=False, left=False)


    ax.set_title(title, fontsize=fontsize+1)
    ax.set_xlabel('X-axis', fontsize=fontsize)
    ax.set_ylabel('Y-axis', fontsize=fontsize)
    

def plot_1d_array(arr, title):
    """
    Plots a 1D array and returns the figure handle.
    
    Parameters:
        arr (array-like): 1D array of values to plot
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Handle to the figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(arr)
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid(True)
    
    return fig



def add_text_to_plot(ax, text, location, offset, colour = 'blue',
                     arrow_colour ='red'
                     ):
   
    ax.annotate(
    text,                                        # Text annotation
    xy=(location[1], location[0]),               # Point to annotate (arrow tip)
    xytext=(location[1]+offset[1], location[0]+offset[0]), # Location of the text
    arrowprops=dict(
        arrowstyle="->",        # Arrow style
        color=arrow_colour,
        connectionstyle="arc3,rad=0.3"  # curved line style, optional
    ),
    fontsize=8,
    color=colour
    
)


 
def plot_output_of_get_slc_burst_data(SLC_safe_id, slc_burst,
                                      intensity_burst_image, final_burst_img, 
                                      final_binary_img):   

    fig1, ax1 = plt.subplots(3,1)
    ax1[0].imshow(intensity_burst_image, norm=colors.LogNorm(vmin=10,vmax = 30000), aspect = 'auto')
    ax1[1].imshow(final_burst_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = 'auto')
    ax1[2].imshow(final_binary_img, cmap='Greys_r', aspect = 'auto')
    ax1[0].set_title(f'SLC Intensity: Burst#{slc_burst} ')
    ax1[1].set_title(f'SLC Amplitude, Ships Boxed  Burst#{slc_burst} Max = {final_burst_img.max():.2f}')
    ax1[2].set_title(f'Binary Mask for  Burst#{slc_burst}')
    fig1.suptitle(f'{SLC_safe_id}', y=0.95)

    #plt.show()
    return fig1, ax1[0]


def get_chip_aoi_in_burst(chip_number, chip_size, burst_size):
    chip_width  = chip_size[1]
    chip_height = chip_size[0]
    burst_width = burst_size[1]
    burst_height= burst_size[0]
        
    num_chips_per_burst_row = (burst_width -1) // chip_width +1
    row = ((chip_number-1) // num_chips_per_burst_row) + 1
    col_start = ((chip_number- 1) % num_chips_per_burst_row )*chip_width
    if burst_width - col_start < chip_width:
        col_start = burst_width - chip_width 
    col_end = col_start + chip_width -1 
    row_start = (row - 1)*chip_height
    if burst_height - row_start < chip_height:
        row_start = burst_height - chip_height
    row_end = row_start + chip_height -1
    chip_aoi = [[row_start, col_start], [row_end, col_start], 
                [row_end, col_end], [row_start, col_end]]
    
    return chip_aoi


def get_chip_data(burst, chip_number, chip_size):
    burst_size = burst.shape
    aoi = get_chip_aoi_in_burst(chip_number, chip_size, burst_size)
    chip = burst[aoi[0][0]:aoi[1][0]+1,aoi[0][1]:aoi[2][1]+1]
    return chip




def draw_chip(burst, chip_number, chip_size,ax_handle):
    burst_size = burst.shape
    chip_boundary = get_chip_aoi_in_burst(chip_number, chip_size, burst_size)
    #chip = get_chip_data(burst, chip_boundary)
    x_min = min(point[1] for point in chip_boundary)
    y_min = min(point[0] for point in chip_boundary)
    # Compute width and height
    width = max(point[1] for point in chip_boundary) - x_min
    height = max(point[0] for point in chip_boundary) - y_min
    # Draw the rectangle (x, y start at top-left)
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='y', facecolor='none')
    ax_handle.add_patch(rect)
    




def get_chip_number(ship_coords, chip_size, burst_size):
    ship_col    = ship_coords[1]
    ship_row    = ship_coords[0]
    chip_width  = chip_size[1]
    chip_height = chip_size[0]
    burst_width = burst_size[1]
    burst_height= burst_size[0]
    
    
    num_chips_per_burst_row = (burst_width -1) // chip_width +1
    row_number = ship_row//chip_height 
    chip_number = ship_col//chip_width +1 + num_chips_per_burst_row*row_number
    return chip_number

def get_chips_to_check(burst_size, chip_size):
    """
    Want to determine how many ships in each chip. 
    Use get_chip_number above to determine if ship is in chip
    But then because the chips at the periphery overlap with their
    immediate neighbour below them for chips at top, or to the left
    of them for chips at end of a row, is possible for ships to be in 
    more than one chip. So need to check if ship in any of these 
    neighbouring chips is within 1 chip_height of top or chip_width 
    of end of row and if so then it must also be included in the
    peripheral chip 

    """
    n_chips_per_row = (burst_size[1] -1)//chip_size[1]  +1
    n_chips_per_col = (burst_size[0] -1)//chip_size[0]  +1
    row_chips_to_check = (n_chips_per_col - 2)*n_chips_per_row + list(range(1,n_chips_per_row+1))
    col_chips_to_check = [x *n_chips_per_row for x in (range(1,n_chips_per_col))]

    return row_chips_to_check, col_chips_to_check

def get_scene_id_and_dataset(SLC_Product_ID):

    """
    Function to find scene_id associated with Product_ID
    and also so determine what dataset it belongs to
    Parameters
    ----------
    SLC_Product_ID : 
    Returns
    -------
    scene_id 

    """
    datasets =['Training', 'Validation', 'Public']
    i=0
    scene_mapping_file_list=['SLC_train.csv', 'SLC_validation.csv', 'SLC_Public_Labels.csv']
    for file in scene_mapping_file_list:
        scene_mapping_file = f'/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/{file}'
        df                 = pd.read_csv(scene_mapping_file)
        # Remove any double underscore
        #SLC_PRODUCT_ID = SLC_Product_ID.replace("__","_")
        filtered_df = df.loc[df["SLC_product_identifier"] == SLC_Product_ID]
        if not filtered_df.empty:
            scene_id = filtered_df["scene_id"].iloc[0]
            dataset  = datasets[i]
            break
            #print(f"scene_id  for SLC_product_identifier {SLC_Product_ID}: {scene_id}")
        else:
            scene_id=''
            dataset=''
            #print(f"ERROR: No SLC match found for scene_id {target_scene_id}. Continuing to next raw file .....")
        i=i+1
    
    return scene_id, dataset



def get_scene_id(SLC_Product_ID):
    """
    Function to find scene_id associated with Product_ID
    Parameters
    ----------
    SLC_Product_ID : 
    Returns
    -------
    scene_id 

    """
    scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
    df          = pd.read_csv(scene_mapping_file)
    # Remove any double underscore
    SLC_PRODUCT_ID = SLC_Product_ID.replace("__","_")
    filtered_df = df.loc[df["SLC_product_identifier"] == SLC_Product_ID]
    if not filtered_df.empty:
        scene_id = filtered_df["scene_id"].iloc[0]
        #print(f"scene_id  for SLC_product_identifier {SLC_Product_ID}: {scene_id}")
    else:
        scene_id=''
        #print(f"ERROR: No SLC match found for scene_id {target_scene_id}. Continuing to next raw file .....")
    
    return scene_id


def get_sea_status(gps_coords, plot_map=False):
    """
    Determines if an ROI (list of 4 [lon, lat] points) contains land.
    If sea-based, finds the nearest land and identifies the country.
    """
    # 1. Setup ROI
    roi_polygon = Polygon(gps_coords)
    centroid = roi_polygon.centroid
    center_lat, center_lon = centroid.y, centroid.x
    
    # 2. Load Political Land Geometries (GeoJSON is more reliable for URLs)
    # This dataset includes the 'name' column for countries
    # url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"

    # 1:10m is the highest resolution available from Natural Earth.
    # It includes virtually all inhabited islands and many uninhabited ones.
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    try:
        land_gdf = gpd.read_file(url)
    except Exception as e:
        print(f"Error loading map data: {e}")
        return None, "Data Error", 0

    # Create a single geometry for intersection checks
    land_geometry = land_gdf.union_all() 

    # 3. Check Land vs Sea
    is_land = land_geometry.intersects(roi_polygon)
    
    if is_land:
        # Identify the country by intersection
        intersected_df = land_gdf[land_gdf.intersects(roi_polygon)]
        # Robustly find name column (usually 'name' in GeoJSON)
        name_col = next((c for c in intersected_df.columns if c.lower() == 'name'), 'name')
        country_names = intersected_df[name_col].tolist()
        country = country_names[0] if country_names else "Unknown Land"
        min_dist = 0
        nearest_lat, nearest_lon = center_lat, center_lon
        print(f"The ROI is land-based in: {country}")
    else:
        print("The ROI is sea-based. Calculating nearest land...")
        # Efficiently find the single closest point on the world's landmass
        # nearest_points returns (point_on_roi, point_on_land)
        _, nearest_land_geom = nearest_points(roi_polygon, land_geometry)
        nearest_lat, nearest_lon = nearest_land_geom.y, nearest_land_geom.x
        
        # Calculate precise geodesic distance
        min_dist = geodesic((center_lat, center_lon), (nearest_lat, nearest_lon)).kilometers
        
        # Use offline reverse_geocoder to identify the country of that nearest land point
        # Input must be [(lat, lon)]
        res = rg.search([(nearest_lat, nearest_lon)])[0]
        try:
            country = pycountry.countries.get(alpha_2=res['cc']).name
        except:
            country = res['cc'] # Fallback to code if name not found
            
        print(f"Nearest land is {min_dist:.2f} km away in {country} ({res['name']})")

    # 4. Plotting
    if plot_map:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Determine extent based on ROI and nearest land
        min_lon, min_lat, max_lon, max_lat = roi_polygon.bounds
        margin = 2.0  # Degrees
        ax.set_extent([min(min_lon, nearest_lon) - margin, 
                       max(max_lon, nearest_lon) + margin, 
                       min(min_lat, nearest_lat) - margin, 
                       max(max_lat, nearest_lat) + margin])

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, facecolor='#e0e0e0')
        ax.add_feature(cfeature.OCEAN, facecolor='#afeeee')
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot ROI
        roi_x, roi_y = roi_polygon.exterior.xy
        ax.plot(roi_x, roi_y, color='red', linewidth=2, label='ROI')
        ax.scatter(center_lon, center_lat, color='blue', label='ROI Center')
        
        if not is_land:
            ax.scatter(nearest_lon, nearest_lat, color='green', label='Nearest Land')
            ax.plot([center_lon, nearest_lon], [center_lat, nearest_lat], 
                    color='black', linestyle='--', transform=ccrs.Geodetic())

        ax.legend()
        plt.title(f"ROI Status: {'Land' if is_land else 'Sea'}\nNearest Country: {country}")
        plt.show()

    return is_land, country, min_dist




def get_sea_statusi_old(gps_coords, plot_map):
    """
    Function to return whether ROI specified by gps_coords is wholly sea
    based or whether there is land within the boundary.
    User can determine if a map of the location of the ROI is plotted
    Returns the status of the ROI: True for land, False if sea
    
    Parameters
    ----------
    gps_coords : Array of 4 gps locations defining the vertices of a polygon
                 Each gps location is in [longitude, latitude] format
                 The vertices are ordered as [bottom-left, top-left, top-right
                 and bottom right]
    plot_map : Boolean - True => plot map; False => Do not plot map

    Returns
    -------
    is_land : Boolean - True if ROI contains land; False if wholly sea based
    country : String - country closest to or encompassing ROI
    min_dist : Real - distance from centre of ROI to land
        
    """
    # Reformat GPS coordinates for Polygon ([lon, lat] instead of [lat, lon])
    #gps_coords_lon_lat = [(lon, lat) for lat, lon in gps_coords]
    # Had error previously as had gps_coords as [lat,long]- now correcting (if inelegantly!)
    gps_coords_lon_lat = gps_coords

    # Convert GPS coordinates to a polygon
    roi_polygon = Polygon(gps_coords_lon_lat)
    
    # Calculate the center of the ROI
    centroid = roi_polygon.centroid
    center_lat, center_lon = centroid.y, centroid.x
    
    # Calculate the bounding box of the ROI
    min_lon, min_lat, max_lon, max_lat = roi_polygon.bounds
    
    # Expand the bounding box to 50 times the area
    lon_margin = (max_lon - min_lon) * 3.5
    lat_margin = (max_lat - min_lat) * 3.5
    expanded_bbox = [min_lon - lon_margin, max_lon + lon_margin, min_lat - lat_margin, max_lat + lat_margin]
    
    # Set up the map
    if plot_map:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([expanded_bbox[0], expanded_bbox[1], expanded_bbox[2], expanded_bbox[3]])
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN, edgecolor='black')
    
        # Plot the ROI
        roi_x, roi_y = roi_polygon.exterior.xy
        ax.plot(roi_x, roi_y, color='red', linewidth=2, label='ROI')
        ax.scatter(center_lon, center_lat, color='blue', label='ROI Center')
    
    # Load global land geometries
    path = geodatasets.get_path("naturalearth.land")
    #land_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    land_gdf = gpd.read_file(path)
    land_geometry = land_gdf.geometry.unary_union  # Combine all land polygons
    
    # Check if the ROI is on land or sea
    geolocator = Nominatim(user_agent="geoapi")
    is_land = land_geometry.intersects(roi_polygon)
    
    if is_land:
        # Identify the country
        #location = geolocator.reverse((center_lat, center_lon), language='en')
        #country = location.raw.get('address', {}).get('country', 'Unknown')
        """
        try:
            location = geolocator.reverse((nearest_lat, nearest_lon), language='en')
            if location:
                country = location.raw.get('address', {}).get('country', 'Unknown')
            else:
                raise ValueError("Reverse geocoding failed.")
        except Exception as e:
            country = "Unknown"
            print(f"Error during reverse geocoding: {e}")
        """
        # Find the intersecting country
        intersected_country = land_gdf[land_gdf.intersects(roi_polygon)]
        if not intersected_country.empty:
            # Extract country names
            country_names = intersected_country['name'].tolist()
            print(f"The ROI intersects with the following country/countries: {', '.join(country_names)}")
        else:
            print("The ROI is on land, but no country was identified.")

        min_dist = 0
        country = country_names[0]
        #print(f"The ROI is land-based and is located in {country_names}.")
    else:
        # If sea-based, find the nearest land-based location
        print("The ROI is sea-based.")
        min_dist = float('inf')
        nearest_land = None
        
        # Iterate through all exterior points of the polygons in the land geometry
        for polygon in land_geometry.geoms:  # Iterate through each polygon in the MultiPolygon
            for point in polygon.exterior.coords:  # Iterate through the exterior points of the polygon
                land_point = Point(point)
                dist = geodesic((center_lat, center_lon), (land_point.y, land_point.x)).kilometers
                if dist < min_dist:
                    min_dist = dist
                    nearest_land = land_point
        
        # Identify the nearest land-based location and its country
        nearest_lat, nearest_lon = nearest_land.y, nearest_land.x
        #location = geolocator.reverse((nearest_lat, nearest_lon), language='en')
        #country = location.raw.get('address', {}).get('country', 'Unknown')
        try:
            location = geolocator.reverse((nearest_lat, nearest_lon), language='en')
            if location:
                country = location.raw.get('address', {}).get('country', 'Unknown')
            else:
                raise ValueError("Reverse geocoding failed.")
        except Exception as e:
            country = "Unknown"
            print(f"Error during reverse geocoding: {e}")

        print(f'Nearst Land at Latitude: {nearest_lat:.3f},  Longitude: {nearest_lon:.3f} in {country}')

        if plot_map:
            # Expand the bounding box further to show the nearest land
            expanded_bbox = [
                min(expanded_bbox[0], nearest_land.x - lon_margin),
                max(expanded_bbox[1], nearest_land.x + lon_margin),
                min(expanded_bbox[2], nearest_land.y - lat_margin),
                max(expanded_bbox[3], nearest_land.y + lat_margin),
                ]
            ax.set_extent([expanded_bbox[0], expanded_bbox[1], expanded_bbox[2], expanded_bbox[3]])
            ax.scatter(nearest_land.x, nearest_land.y, color='green', label='Nearest Land')
            print(f"The nearest land is approximately {min_dist:.2f} km away at {nearest_lat:.2f}, {nearest_lon:.2f} in {country}.")
    
    if plot_map:
        # Add a legend and display the map
        ax.legend()
        plt.title(f'ROI and Expanded Area: ROI located {min_dist:.0f}km from gps coords[{nearest_lat:.2f}, {nearest_lon:.2f}] in {country}')
        plt.show()
        
    return is_land, country, min_dist


###############################################################################
# Series of functions to interrogate SM SLC files and return reference gps 
# and grid co-ordinates
#
#
# Function to return grid points and gps grid points for Stripmap file
# Usually only 1 Azimuth Block of echo data in Stripmap - but might be more
# Here it is assumed that there is only 1
# References to bursts are a relic of IW mode processiing
#                          
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









###############################################################################
def get_SM_GridPoints(SM_SLC_annotation_file):
    """
    Function to extract dataframe of location and timing
    for reference gps and grid(Azimuth, Range) pixels for 
    Strip Map mode SLC annotation files.
    
    Typically SM SLC files have very few data bursts, 1-2. 
    Other bursts in RAW are for calibration and testing.
    For convenience have ordered grids as bursts whenever a 
    range value of 0 is found - which is similar to what is done
    for IW mode but should note does not denote a burst in SM
    
    Input: Strip Mode SLC annotation xml file
    Output: Dataframe with locations of each reference grid point in 
            GPS and grid co-ordinates along with azimuth time for 
            for start of grid line.
    """
    bursts =[]
    #SLC_annotation_file=SM_annotation_file
    tree = etree.parse(SM_SLC_annotation_file)
    root = tree.getroot()

    nBurstLines     = int(root.find('.//numberOfLines').text)
    nRangeSamples   = int(root.find('.//numberOfSamples').text)
    azTimeInterval  = float(root.find('.//azimuthTimeInterval').text)
    az_pri          = float(root.find('.//pri').text)
    burst_number    = 0
    burstGridPoints = []
    for burstGridPoint in root.findall('.//geolocationGridPoint'):
        azimuth_time = datetime.strptime(burstGridPoint.find('.//azimuthTime').text,
                                           '%Y-%m-%dT%H:%M:%S.%f')
        longitude    = burstGridPoint.find('.//longitude').text
        latitude     = burstGridPoint.find('.//latitude').text
        line         = burstGridPoint.find('.//line').text
        pixel        = burstGridPoint.find('.//pixel').text
        slant_range_time =  burstGridPoint.find('.//slantRangeTime').text

        if pixel =='0':
            burst_number +=1
            azimuth_burst_end_time = azimuth_time + timedelta(seconds
                                      = (nBurstLines * azTimeInterval)
                                      )
        burstGridPoints.append({
            'azimuth_time': azimuth_time,
             'longitude': longitude,
             'latitude': latitude,
             'line': line,
             'pixel': pixel,
             'burst_number' : str(burst_number),
             'azimuth_burst_end_time': azimuth_burst_end_time,
             'slant_range_time' : slant_range_time

        })
    #Get Ground Control Points from GeoLocationGridPoints
    df                = pd.DataFrame(burstGridPoints)
    #gridPoints        = df[['burst_number','line','pixel','longitude','latitude']]
    #gridControlPoints = gridPoints.to_numpy().astype(float)
    return(df)




###############################################################################
## Function to return both meta data and radar data for a Stripmap file
## Assumes only one signal echo (burst) or azimuth block with radar data
## If > 1 use something else


def get_SM_data(raw_file):
        
#    l0file = sentinel1decoder.Level0File(raw_file)
    
    raw_burst_info, l0file = get_SM_raw_burst_info(raw_file)
    if len(raw_burst_info)>1:
        print(f'Warning: More than 1 SM Azimuth Block - need to process'
              f'differently!')
    
    else:
        # Finds first burst/block that is a signal echo
        selected_burst = raw_burst_info[0].index[0]
        meta_data      = l0file.get_burst_metadata(selected_burst)
        radar_data     = l0file.get_burst_data(selected_burst)
        
    return meta_data, radar_data, l0file
    





###############################################################################
##  July 9th 2026
## Function to determine offset between start/end of raw file and slc in
## both azimuth and range.
## Based on timings in raw meta data and slc annotation file
## Matches azimuth line start time (? mid-slant range?) in both raw and slc
## Range matching currently assumed symmetrical - but may not be
##
def get_SM_raw_slc_offsets(raw_file, slc_annotation_file):
    
    
    raw_burst_info, l0file = get_SM_raw_burst_info(raw_file)
    if len(raw_burst_info)>1:
        print(f'Warning: More than 1 SM Azimuth Block - need to process'
              f'differently!')
    
    else:
        selected_burst = raw_burst_info[0].index[0]
        #l0file         = sentinel1decoder.Level0File(raw_file)
        selection      = l0file.get_burst_metadata(selected_burst)
        radar_data     = l0file.get_burst_data(selected_burst)
        
    slc_burst_info = get_SM_GridPoints(slc_annotation_file)
    tree = etree.parse(slc_annotation_file)
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
    slc_az_end_offset_from_raw   = int((n_raw_az_lines 
                                 -  (slc_az_start_offset_from_raw + n_slc_az_lines))
                                   )
    """
    #Check if any difference from mide
    mid_range_pixel = n_slc_range_samples /2 
    closest_row_label = (pd.to_numeric(slc_burst_info['pixel']) 
                        - mid_range_pixel).abs().idxmin()
    #Look up the azimuth_time for that row
    mid_range_azimuth_time = slc_burst_info.loc[closest_row_label, 'azimuth_time']
    delta_from_mid = (mid_range_azimuth_time - raw_start_time.iloc[0]).total_seconds()
    slc_az_start_offset_from_raw_mid = np.ceil(delta / az_pri)
    """
    
    # Check from range offset 
    range_sample_diff = n_raw_range_samples - n_slc_range_samples
    slc_range_start_offset = int(np.round(range_sample_diff / 2 ))
    slc_range_end_offset   = int(np.round(n_raw_range_samples - (slc_range_start_offset + n_slc_range_samples)))
    
    return slc_az_start_offset_from_raw, slc_az_end_offset_from_raw, slc_range_start_offset, slc_range_end_offset






# Function to find the 4 nearest neighbors
def find_nearest_neighbours(df, target_line, target_pixel):
    
    """
    Function to take dataframe containing gps and grid co-ordinates
    of SM SLC grid data and determine which gps and co-ordinate grid
    the Azimuth and Range target pixel belongs to by determining
    the 4 nearest neighbouring points of the SLC reference grid and their
    corresponding gps locations
    Returns a dataframe with 4 lines: one for each of nearest neighbours
    """
    # Convert line and pixel to numeric
    df['line_num'] = df['line'].astype(int)
    df['pixel_num'] = df['pixel'].astype(int)
    target_line_num = int(target_line)
    target_pixel_num = int(target_pixel)
    
    # Calculate distances
    df['distance'] = np.sqrt((df['line_num'] - target_line_num)**2 + (df['pixel_num'] - target_pixel_num)**2)
    
    # Exclude the target point itself
    df_filtered = df[(df['line'] != target_line) | (df['pixel'] != target_pixel)].copy()
    
    # Group by line and select the two closest points per line
    nearest_by_line = (
        df_filtered.sort_values(by='distance')
        .groupby('line')
        .head(2)  # Limit to 2 per line
    )
    
    # Get the top 4 closest neighbours overall
    top_neighbours = (
        nearest_by_line.nsmallest(4, 'distance')
        .sort_values(by=['line_num', 'pixel_num', 'longitude', 'latitude','azimuth_time'])  # Sort by line and pixel
    )
    
    # Return result as a list of [line, pixel]
    #return top_neighbours[['line', 'pixel']].values.tolist()
    return top_neighbours.sort_values(by='line')


def get_SM_ref_grids(df, target_line, target_pixel):
    """
    Functon to return gps and co-ordinate grid that a 
    target pixel belongs to
    
    
    """
    neighbours = find_nearest_neighbours(df, target_line, 
                                         target_pixel).sort_values(by='pixel')
    tmp        = neighbours.iloc[3]
    neighbours.iloc[3]=neighbours.iloc[2]
    neighbours.iloc[2]=tmp
    ref_grid   = neighbours[['line','pixel']].astype(int).to_numpy()
    gps_grid   = neighbours[['longitude','latitude']].astype(float).to_numpy()
    grid       =[]
    grid.append(ref_grid)
    grid.append(gps_grid)
    return grid


###############################################################################
def contains_SAFE_file(directory):
    for file in os.listdir(directory):  # List all files and directories
        if file.endswith('.SAFE'):  # Check if the file ends with .sst
            return True
    return False



def change_extension(file_name, new_extension):
    base_name, _ = os.path.splitext(file_name)
    new_file_name = base_name + "." + new_extension
    return new_file_name




from datetime import datetime, timedelta
def gps_to_utc_datetime(coarse_time, fine_time):
    # Reference epoch for Sentinel-1
    reference_epoch = datetime(1980, 1, 6, 0, 0, 0)
    # Convert coarse time (seconds since epoch)
    coarse_delta = timedelta(seconds=int(coarse_time))
    # Convert fine time (picoseconds to seconds)
    fine_delta = timedelta(seconds=float(fine_time))
    #Adjust for addition of leap seconds
    leap_seconds = timedelta(seconds = 18)
    
    # Combine and return UTC time
    return reference_epoch + coarse_delta + fine_delta - leap_seconds



def gps_to_utc(coarse_time, fine_time):
    """
    Converts GPS time (coarse + fine) to UTC time using numpy.datetime64.
    
    Parameters:
        coarse_time (float): Coarse time in seconds since the GPS epoch.
        fine_time (float): Fine time in fractional seconds (sub-second precision).
    
    Returns:
        numpy.datetime64: UTC time corresponding to the input GPS time.
    """
    # Reference epoch for Sentinel-1 (GPS epoch)
    reference_epoch = np.datetime64('1980-01-06T00:00:00', 'ns')
    
    # Convert coarse time (seconds) to timedelta in nanoseconds
    coarse_delta = np.timedelta64(int(coarse_time * 1e9), 'ns')
    
    # Convert fine time (fractional seconds) to timedelta in nanoseconds
    fine_delta = np.timedelta64(int(fine_time * 1e9), 'ns')
    
    # Leap seconds adjustment (18 seconds as of now, convert to nanoseconds)
    leap_seconds = np.timedelta64(18, 's')
    
    # Combine reference epoch, coarse time, fine time, and subtract leap seconds
    utc_time = reference_epoch + coarse_delta + fine_delta - leap_seconds
    
    return utc_time



# Function to return gps co-ordinates of region of interest
# indices contains the [row,col] locations for the ROI boundary
def find_coordinates_array(df, indices):
    coordinates = []

    for row_index, column_index in indices:
        # Filter the DataFrame for the specific row and column indices
        # Note line and pixel are strings when extracted from xml annotation file
        result = df[(df['line'] == str(row_index)) & (df['pixel'] == str(column_index))]

        if not result.empty:
            longitude = result['longitude'].values[0]
            latitude  = result['latitude'].values[0]
            coordinates.append((longitude, latitude))
        else:
            coordinates.append((None, None))  # Append (None, None) if no match is found

    return coordinates



# Function to return information on each raw burst in each swath
#
#
def get_raw_burst_info(raw_filename):
    """
    Function to report information regarding bursts, their size and start
    end times and their number.
    Solely for IW mode: SM mode requires reconfiguring for S1-S6
    """
    raw_data = sentinel1decoder.Level0File(raw_filename)
               
    IW=[]
    for swath_number in range(10,13):
        filtered_df = raw_data.packet_metadata[raw_data.packet_metadata['Swath Number'] == swath_number]
        #filtered_df['Row Number'] = filtered_df.index
        result = filtered_df.groupby('Azimuth Block Number').agg(['first', 'last'])
        #result = filtered_df.groupby('Azimuth Block Number').first()

        result.reset_index()[['Azimuth Block Number', 'Coarse Time', 
                              'Fine Time', 'ECC Number','Signal Type',
                              'Swath Number', 'Number of Quads']]
        result = result.loc[:, [('Coarse Time', 'first'),('Fine Time', 'first'),
                                ('Coarse Time', 'last'), ('Fine Time', 'last'), 
                                ('Swath Number', 'first'), ('Number of Quads','first')]]
        result.columns = ['Coarse Time First',  'Fine Time First','Coarse Time Last', 
                          'Fine Time Last', 'Swath Number', 'Number of Quads']
        result['Burst Start Time'] = pd.to_datetime(result.apply(lambda row: 
                                      gps_to_utc(row['Coarse Time First'], 
                                              row['Fine Time First']), axis=1))
        result['Burst End Time'] = pd.to_datetime(result.apply(lambda row:
                                     gps_to_utc(row['Coarse Time Last'], 
                                               row['Fine Time Last']), axis=1))
        result['Burst Duration'] = (result['Burst End Time'] 
                                 - result['Burst Start Time']).dt.total_seconds()
        result['Mid Burst Time'] = result['Burst Start Time'] + pd.to_timedelta(result['Burst Duration']/2, unit='s')
        # Get Number of Azimuth Lines in each burst
        aa=filtered_df.groupby('Azimuth Block Number',as_index=False).head(1)
        packet_numbers1 = aa.index.get_level_values('Packet Number')
        aa=filtered_df.groupby('Azimuth Block Number',as_index=False).tail(1)
        packet_numbers2 = aa.index.get_level_values('Packet Number')
        number_of_azimuth_lines = packet_numbers2 - packet_numbers1 + 1 
        result['Num Azimuth Lines']=number_of_azimuth_lines
        IW.append(result)
    return(IW)

##
## Function to determine start and end time of SM raw 'burst' and report
## number of lines and range-samples. Also returns all the decoded data
## so can be used in other functions and reduces memory requirements by 
## preventing duplication
##
def get_SM_raw_burst_info(raw_filename):
    raw_data = sentinel1decoder.Level0File(raw_filename)
               
    SM=[]
    # Note: Labelling of SM swaths starts at 0  => S6 -> Swath Number of 5
    swath_number = int(raw_filename.split('-')[1][1]) - 1
    #swath_number = int(re.search(r's1a-s(\d+)-', raw_filename).group(1)) -1
    #filtered_df = raw_data.packet_metadata[raw_data.packet_metadata['Swath Number'] == swath_number]
    filtered_df = raw_data.packet_metadata[
        (raw_data.packet_metadata['Swath Number'] == swath_number) & 
        (raw_data.packet_metadata['Signal Type'] == 0)
    ]
    result = filtered_df.groupby('Azimuth Block Number').agg(['first', 'last'])
    result.reset_index()[['Azimuth Block Number', 'Coarse Time', 
                          'Fine Time', 'ECC Number','Signal Type',
                          'Swath Number', 'Number of Quads']]
    result = result.loc[:, [('Coarse Time', 'first'),('Fine Time', 'first'),
                            ('Coarse Time', 'last'), ('Fine Time', 'last'),
                            ('Signal Type', 'first'), 
                            ('Swath Number', 'first'), ('Number of Quads','first')]]
    result.columns = ['Coarse Time First',  'Fine Time First','Coarse Time Last', 
                      'Fine Time Last','Signal Type', 'Swath Number', 'Number of Quads']
    result['Burst Start Time'] = pd.to_datetime(result.apply(lambda row: 
                                  gps_to_utc(row['Coarse Time First'], 
                                          row['Fine Time First']), axis=1))
    result['Burst End Time'] = pd.to_datetime(result.apply(lambda row:
                                 gps_to_utc(row['Coarse Time Last'], 
                                           row['Fine Time Last']), axis=1))
    result['Burst Duration'] = (result['Burst End Time'] 
                             - result['Burst Start Time']).dt.total_seconds()
    result['Mid Burst Time'] = result['Burst Start Time'] + pd.to_timedelta(result['Burst Duration']/2, unit='s')
    # Get Number of Azimuth Lines in each burst
    aa=filtered_df.groupby('Azimuth Block Number',as_index=False).head(1)
    packet_numbers1 = aa.index.get_level_values('Packet Number')
    aa=filtered_df.groupby('Azimuth Block Number',as_index=False).tail(1)
    packet_numbers2 = aa.index.get_level_values('Packet Number')
    number_of_azimuth_lines = packet_numbers2 - packet_numbers1 + 1 
    result['Num Azimuth Lines']=number_of_azimuth_lines
    SM.append(result)
    return SM, raw_data

# Function to return time before and after mid-point of burst for which 
# SLC data was used for de-merged bursts
# Time used to determine how long before and how long after mid-point of RAW
# burst that RAW lines (corresponding to SLC azimuth lines) should be accepted
# for processing
#
def get_pre_post_mid_times(SLC_annotation_file):
    
    bursts =[]
    tree = etree.parse(SLC_annotation_file)
    root = tree.getroot()
    
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        azimuth_anx_time = burst.find('.//azimuthAnxTime').text
        sensing_time = burst.find('.//sensingTime').text
        
        bursts.append({
            'azimuth_time': azimuth_time,
            'azimuth_anx_time': azimuth_anx_time,
            'sensing_time' : datetime.strptime(sensing_time,'%Y-%m-%dT%H:%M:%S.%f')
        })
        df_burst=pd.DataFrame(bursts)
        
    nBurstLines     = int(root.find('.//numberOfLines').text)
    nRangeSamples   = int(root.find('.//numberOfSamples').text)
    azTimeInterval  = float(root.find('.//azimuthTimeInterval').text)
    az_pri          = float(root.find('.//pri').text)
    burst_number    = 0
    burstGridPoints = []
    for burstGridPoint in root.findall('.//geolocationGridPoint'):
        azimuth_time = datetime.strptime(burstGridPoint.find('.//azimuthTime').text,
                                           '%Y-%m-%dT%H:%M:%S.%f')
        longitude    = burstGridPoint.find('.//longitude').text
        latitude     = burstGridPoint.find('.//latitude').text
        line         = burstGridPoint.find('.//line').text
        pixel        = burstGridPoint.find('.//pixel').text

        if pixel =='0':
            burst_number +=1

        azimuth_burst_end_time = azimuth_time + timedelta(seconds
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
    gridPoints        = df[['burst_number','line','pixel','longitude','latitude']]
    gridControlPoints = gridPoints.to_numpy().astype(float)
    #print(gridPoints[gridPoints['burst_number'] == '5'])
    #print(df['azimuth_burst_end_time'])
    pixel_value = '0'
    burst_start_times   = df[df['pixel'] == pixel_value]['azimuth_time']
    original_burst_info = df[df['pixel'] == pixel_value][['burst_number','azimuth_time', 'azimuth_burst_end_time','line', 'pixel']]
        
    
    merged_burst_information = getSwathBurstGridGPSBoundaries(SLC_annotation_file)
    merged_burst_corners     = merged_burst_information[1]
    
    merged_burst_start_line =[]
    merged_burst_end_line =[]
    burst_start_line=[]
    burst_end_line=[]

    for i in range(0, len(bursts)):
        merged_burst_start_line.append(merged_burst_information[1][i][0][0])
        merged_burst_end_line.append(merged_burst_information[1][i][1][0])
        burst_start_line.append(original_burst_info[original_burst_info['burst_number']
                                 == str(i+1)]['line'].to_numpy().astype(int)[0])
        burst_end_line.append(original_burst_info[original_burst_info['burst_number']
                                 == str(i+2)]['line'].to_numpy().astype(int)[0])
    
    # Need next 2 lines as a kluge to make array length 10 to match original_burst_info
    burst_start_line.append(original_burst_info[original_burst_info['burst_number']
                                 == str(i+1)]['line'].to_numpy().astype(int)[0])
    burst_end_line.append(original_burst_info[original_burst_info['burst_number']
                                 == str(i+2)]['line'].to_numpy().astype(int)[0])
    # Ditto
    merged_burst_start_line.append(merged_burst_information[1][i][0][0])
    merged_burst_end_line.append(merged_burst_information[1][i][1][0])
    
    original_burst_info['merged_start_line'] = merged_burst_start_line
    original_burst_info['merged_end_line']   = merged_burst_end_line
    original_burst_info['start_line']        = burst_start_line
    original_burst_info['end_line']          = burst_end_line
    original_burst_info['end_line']          = original_burst_info['end_line'] - 1
    original_burst_info['mid_line']          = round((original_burst_info['end_line'] + original_burst_info['start_line'])/2)
    original_burst_info['pre_mid']           = original_burst_info['mid_line']-original_burst_info['merged_start_line']
    original_burst_info['post_mid']          = original_burst_info['merged_end_line']-original_burst_info['mid_line']
    original_burst_info['pre_mid_time']      = original_burst_info['pre_mid']*az_pri
    original_burst_info['post_mid_time']     = original_burst_info['post_mid']*az_pri
    original_burst_info                      = original_burst_info.iloc[:-1]
    original_burst_info                      = original_burst_info.reset_index()
    original_burst_info.insert(4,'sensing_time', df_burst['sensing_time'])
    return(original_burst_info)





def find_raw_start_burst(raw_data, slc_data):
    """
    Finds the burst number for the first raw burst that matches
    with the first SLC burst.

    Args:
        raw_data (pd.DataFrame): SLC data with a 'sensing_time' column.
        slc_data (pd.DataFrame): RAW data with a 'Burst Start Time' column.

    Returns:
        int: Index of the row in the RAW data with the smallest positive time
        difference, corresponding to the matching burst. The index is with 
        reference to all the raw bursts of which there are typically over 70
        as they include test and calibration bursts as well.
    """
    max_permitted_diff = 0.1
    df1 = slc_data
    df2 = raw_data
    # Convert start_time to UTC
    first_start_time_utc = df1['sensing_time'].iloc[0].tz_localize('UTC')
    # print(f'SLC Start Time UTC = {first_start_time_utc}')
    # Calculate time differences in seconds
    #time_diffs = -1*(df2['Burst Start Time'].dt.tz_localize('UTC') - first_start_time_utc).dt.total_seconds()
    time_diffs = (df2['Burst Start Time'].dt.tz_localize('UTC') - first_start_time_utc).dt.total_seconds()
    time_diffs = abs(time_diffs)
    # print(f' Time Differences = {time_diffs}')
    # Filter for positive time differences and find the index of the minimum
    min_positive_diff_index = time_diffs[time_diffs > 0].idxmin()
    
    print(f' Time Difference at index {min_positive_diff_index} = {time_diffs.loc[min_positive_diff_index]}')
    
    if time_diffs.loc[min_positive_diff_index] > max_permitted_diff:
        print(f'WARNING: Burst {min_positive_diff_index} has time offset from'
              f' SLC of {time_diffs[min_positive_diff_index]} : Not a match')    
        
    return min_positive_diff_index
  

#
def get_SM_BurstBoundaries(annotation_file):
    
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    
    nLines         = int(root.find('.//numberOfLines').text)
    nRangeSamples   = int(root.find('.//numberOfSamples').text)
    #nBursts        = len(bursts)
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
    
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
        # burst_end_time not used. Would remove but want to keep fn close to 
        # getBurstBoundaries
        azimuth_burst_end_time = datetime.strptime(azimuth_time, 
                                                   '%Y-%m-%dT%H:%M:%S.%f') \
                                  + timedelta(seconds 
                                  = (0 * azTimeInterval))

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
    nBursts = burst_number - 1   
    for i in range(0,nBursts):
        burst_start_line  = gridPoints[gridPoints['burst_number'] 
                                 == str(i+1)]['line'].to_numpy().astype(int)[0]
        burst_end_line    = gridPoints[gridPoints['burst_number'] 
                                 == str(i+2)]['line'].to_numpy().astype(int)[0]
        merged_start_line = burst_start_line
        merged_end_line   = burst_end_line -1

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

    burst_info=[]
    burst_info.append(burst_corners)
    burst_info.append(burst_gps_corners)
    
    return burst_info

#
# Function to return boundaries of bursts prior to de-bursting
#
def getBurstBoundaries(annotation_file):
    
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        bursts.append({
            'azimuth_time': azimuth_time,
        })
    nLines         = int(root.find('.//numberOfLines').text)
    nSamples       = int(root.find('.//numberOfSamples').text)
    nBursts        = len(bursts)
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    nRangeSamples  = int(root.find('.//samplesPerBurst').text)
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
    
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
   
    for i in range(0,nBursts):
        burst_start_line  = gridPoints[gridPoints['burst_number'] 
                                 == str(i+1)]['line'].to_numpy().astype(int)[0]
        burst_end_line    = gridPoints[gridPoints['burst_number'] 
                                 == str(i+2)]['line'].to_numpy().astype(int)[0]
        merged_start_line = burst_start_line
        merged_end_line   = burst_end_line -1

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

    burst_info=[]
    burst_info.append(burst_corners)
    burst_info.append(burst_gps_corners)
    
    return burst_info

def get_single_non_debursted_slc_burst(slc_file, slc_annotation_file, burst):
    all_burst_data = getBurstBoundaries(slc_annotation_file)
    all_burst_corners = all_burst_data[0]
    burst_boundary = all_burst_corners[burst -1]
    src            = rasterio.open(slc_file)
    img            = src.read(1)
    burst_img = create_roi_from_polygon(img, burst_boundary)
    return burst_img







#
# Function to get GPS coordinates of a swath pointed to by 'annotation file'
#
def getSwathGPSBoundaries(annotation_file):
    #print(f'%%%%%%%%% INSIDE getSwathGPSBoundaries --- annotation_file :'
    #      f'{annotation_file} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        bursts.append({
            'azimuth_time': azimuth_time,

        })
    nLines         = int(root.find('.//numberOfLines').text)
    nSamples       = int(root.find('.//numberOfSamples').text)
    nBursts        = len(bursts)
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    nRangeSamples  = int(root.find('.//samplesPerBurst').text)
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
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
    gridPoints        = df[['burst_number','line','pixel','longitude','latitude']]
    gridControlPoints = gridPoints.to_numpy().astype(float)
    #print(gridPoints[gridPoints['burst_number'] == '5'])
    #print(df['azimuth_burst_end_time'])
    pixel_value = '0'
    burst_start_times = df[df['pixel'] == pixel_value]['azimuth_time']
    #Define boundaries of SLC file 
    file_corners = [(0,0),
                    (nBurstLines*nBursts-1,0),
                    (nBurstLines*nBursts-1,nRangeSamples-1),
                    (0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    #print(f'GPS corners of SLC file are {gps_corners}')
    return gps_corners

"""

    
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        bursts.append({
            'azimuth_time': azimuth_time,
        })

""" 

def get_valid_corners_of_bursts(annotation_file):
    """
    Parameters
    ----------
    annotation_file : TYPE
        DESCRIPTION.
        Reads annotation file to determine the valid lines and range samples
        in SLC data.
        Interrogates firstValidSample and lastValidSample parameters:
         -number of '-1' at start of first/lastValidSample gives number of
          lines at start of burst to ignore
         -value >-1 gives number of range samples to ignore at start of
          corresponding line for firstValidSample
         -value >-1 gives number of range samples to ignore at end of
          corresponding line for lastValidSample
         -number of '-1' at end of first/lastValidSample gives number of
          lines at end of burst to ignore
    Returns
    -------
    valid_corners_of_bursts : TYPE
        DESCRIPTION.
        list of valid corners for each burst
    """
    # Find index of first value not equal to -1
    def first_not_minus_one(lst):
        for i, val in enumerate(lst):
            if val != -1:
                return i
        return None  # or -1 if you prefer
    
    # Find index of last value not equal to -1
    def last_not_minus_one(lst):
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] != -1:
                return i
        return None  # or -1 if you prefer

    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        firstValidSample_text=burst.find('.//firstValidSample').text
        lastValidSample_text=burst.find('.//lastValidSample').text
        # Convert to list of ints (handle empty or missing gracefully)
        firstValidSample = [int(x) for x in firstValidSample_text.split()] if firstValidSample_text else []
        lastValidSample = [int(x) for x in lastValidSample_text.split()] if lastValidSample_text else []
        bursts.append({
            'firstValidSample': firstValidSample,
            'lastValidSample': lastValidSample,
            'azimuth_time': azimuth_time,
        })
        
    df  = pd.DataFrame(bursts)
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    # Count number of values in each list
    df['count_values']     = df['firstValidSample'].apply(len)    
    df['first_valid_line'] = df['firstValidSample'].apply(first_not_minus_one)
    df['last_valid_line']  = df['firstValidSample'].apply(last_not_minus_one)
    
    # Extract values using np.select
    first_valid_range_pixels = np.select(
        [df['first_valid_line'] < df['firstValidSample'].str.len()],  # Condition: index in bounds
        [df.apply(lambda row: row['firstValidSample'][row['first_valid_line']], axis=1)],  # Value if True
        default=np.nan  # Value if False (out of bounds)
    ).astype(int)
    
    last_valid_range_pixels = np.select(
        [df['last_valid_line'] < df['lastValidSample'].str.len()],  # Condition: index in bounds
        [df.apply(lambda row: row['lastValidSample'][row['last_valid_line']], axis=1)],  # Value if True
        default=np.nan  # Value if False (out of bounds)
    ).astype(int)
    
    df['first_valid_range_pixels'] = first_valid_range_pixels  # Add results to the DataFrame
    df['last_valid_range_pixels']  = last_valid_range_pixels  # Add results to the DataFrame    
    


    # Test effect of ignoring invalid lines in azimuth 
    # - as long as overlap greater shouldn't matter
    # Remove when finished testing
    #
    #df['first_valid_line'] = 0 
    #df['last_valid_line']  = nBurstLines

    # Create 'valid_burst_corners' as list of tuples per row
    df['valid_burst_corners'] = df.apply(lambda row: [(row['first_valid_line'], row['first_valid_range_pixels']),
                                                      (row['last_valid_line'], row['first_valid_range_pixels']),
                                                      (row['last_valid_line'], row['last_valid_range_pixels']),
                                                      (row['first_valid_line'], row['last_valid_range_pixels'])], axis=1)
    
    #valid_corners_of_bursts = np.array(df['valid_burst_corners'].tolist())
    valid_corners_of_bursts = df['valid_burst_corners'].tolist()
    return valid_corners_of_bursts




def get_number_valid_lines_in_slc_burst(valid_burst_corners):
    number_valid_burst_lines=[]
    for i in range(0,len(valid_burst_corners)):
        number_valid_burst_lines.append(valid_burst_corners[i][1][0] - valid_burst_corners[i][0][0])
    return number_valid_burst_lines

def get_number_valid_range_pixels_in_slc_burst(valid_burst_corners):
    number_valid_burst_range_pixels=[]
    for i in range(0,len(valid_burst_corners)):
        number_valid_burst_range_pixels.append(valid_burst_corners[i][2][1] - valid_burst_corners[i][1][1])
    return number_valid_burst_range_pixels


def get_number_invalid_start_lines_in_slc_burst(valid_burst_corners):
    number_invalid_burst_start_lines=[]
    for i in range(0,len(valid_burst_corners)):
        number_invalid_burst_start_lines.append(valid_burst_corners[i][0][0])
    return number_invalid_burst_start_lines


def get_number_invalid_start_range_pixels_in_slc_burst(valid_burst_corners):
    number_invalid_start_range_pixels=[]
    for i in range(0,len(valid_burst_corners)):
        number_invalid_start_range_pixels.append(valid_burst_corners[i][0][1])
    return number_invalid_start_range_pixels




#############################################################################
# Amended getSwathBurstGridGPSBoundaries to correct for valid azimuth and 
# range samples


#def getSwathValidMergedBurstsGPSGrid(annotation_file):
#def getSwathBurstGridGPSBoundaries_test(annotation_file):
def getSwathBurstGridGPSBoundaries(annotation_file):
    #Add code to get valid lines and pixels
    valid_burst_corners             = get_valid_corners_of_bursts(annotation_file)
    number_valid_burst_lines        = get_number_valid_lines_in_slc_burst(valid_burst_corners)
    number_valid_burst_range_pixels = get_number_valid_range_pixels_in_slc_burst(valid_burst_corners)
    invalid_burst_start_lines       = get_number_invalid_start_lines_in_slc_burst(valid_burst_corners)
    invalid_burst_start_pixels      = get_number_invalid_start_range_pixels_in_slc_burst(valid_burst_corners)
    
    
    
    
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        bursts.append({
            'azimuth_time': azimuth_time,
        })
    nLines         = int(root.find('.//numberOfLines').text)
    nSamples       = int(root.find('.//numberOfSamples').text)
    nBursts        = len(bursts)
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    nRangeSamples  = int(root.find('.//samplesPerBurst').text)
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
    #print(f'getSwathBurstGridGPSBoundaries::Number of bursts in'
    #      f'{annotation_file} is {nBursts}')
    #print(f'                                Number of azimuth lines in'
    #      f' each burst is {nLines}')
    #print(f'getSwathBurstGridGPSBoundaries:: Number of lines in each burst is {nBurstLines}')
    #print(f'getSwathBurstGridGPSBoundaries:: Number of Samples is {nRangeSamples}')
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
                               + timedelta(seconds=(nBurstLines*azTimeInterval)
                                           )
    
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
    
    
    """
    #Add code to get valid lines and pixels
    valid_burst_corners             = get_valid_corners_of_bursts(annotation_file)
    number_valid_burst_lines        = get_number_valid_lines_in_slc_burst(valid_burst_corners)
    number_valid_burst_range_pixels = get_number_valid_range_pixels_in_slc_burst(valid_burst_corners)
    invalid_burst_start_lines       = get_number_invalid_start_lines_in_slc_burst(valid_burst_corners)
    invalid_burst_start_pixels      = get_number_invalid_start_range_pixels_in_slc_burst(valid_burst_corners)
    """
    
    valid_burst_start_times=[]
    for i in range(0, nBursts):
        valid_burst_start_times.append(datetime.strptime(burst_start_times.iloc[i],
                                                   '%Y-%m-%dT%H:%M:%S.%f') \
                                  + timedelta(seconds
                                  = (invalid_burst_start_lines[i]* azTimeInterval)))
    #Add extra extry to match size of burst_start_times     
    valid_burst_start_times.append(datetime.strptime(burst_start_times.iloc[-1],'%Y-%m-%dT%H:%M:%S.%f'))
    
    burst_end_time = []
    number_overlap_lines = []
    offset_lines_to_start = []
    for i in range(0,len(burst_start_times)-1):
        # Add burst interval time to start time
        t = valid_burst_start_times[i] + timedelta(seconds=(number_valid_burst_lines[i]*azTimeInterval))
        # Subtract end time of burst i from start of next burst i+1
        overlap_time  = t - valid_burst_start_times[i+1]
        #Round down (why?)
        overlap_lines = int(overlap_time.total_seconds() / azTimeInterval)
        burst_end_time.append(t)
        number_overlap_lines.append(overlap_lines)
        #If using validLines then need offset to actual start of next burst so gps and cartesian align
        offset_time_to_start = valid_burst_start_times[i+1] \
                             - datetime.strptime(burst_start_times.iloc[i+1], '%Y-%m-%dT%H:%M:%S.%f')
        offset_lines_to_start.append(int(offset_time_to_start.total_seconds() / azTimeInterval))
        #print(f' t = {t}')
        #print(f' *** valid burst start times = {valid_burst_start_times[i+1]}')
        #print(f" *** burst start times = {datetime.strptime(burst_start_times.iloc[i+1], '%Y-%m-%dT%H:%M:%S.%f')}")
        #print(f' *** OffsetTimesToStart_total_seconds = {offset_time_to_start.total_seconds():.6f}')
        #print(f'type of offset_lines_to_start is: {type(offset_lines_to_start[i])}')
        #print(f' Azimuth Time Interval = {azTimeInterval}')
        #print(f' offset_lines_to_start = {offset_lines_to_start[i]}')
    
    overlap_offset_lines = [x//2 for x in number_overlap_lines]
    merged_burst_start_line = []
    merged_burst_end_line   = []
    burst_corners           = []
    burst_gps_corners       = []
    non_overlapped_burst_corners = []
    gridPoints['merged_burst_start_line']= 0
    gridPoints['merged_burst_end_line']  = 0
    
    burst_ends           = gridPoints[(gridPoints['pixel']
                                 == str(nRangeSamples-1))]
    burst_starts         = gridPoints[(gridPoints['pixel'] == str(0))]
    burst_end_long_lat   = burst_ends[['longitude',
                                       'latitude']].to_numpy().astype(float)
    burst_start_long_lat = burst_starts[['longitude',
                                         'latitude']].to_numpy().astype(float)
    burst_grid_corners   = []
    burst_grid_gps       = []
    merged_gps_corners   = []
    burst_merged_grid_gps= []
    
    burst_start_line = []
    burst_end_line   = []
    for i in range(0,nBursts):
        valid_burst_start_line = invalid_burst_start_lines[i] + i*nBurstLines
        valid_burst_end_line   = valid_burst_start_line + number_valid_burst_lines[i]
        burst_start_line       = gridPoints[gridPoints['burst_number']
                                     == str(i+1)]['line'].to_numpy().astype(int)[0]
        burst_end_line         = gridPoints[gridPoints['burst_number']
                                     == str(i+2)]['line'].to_numpy().astype(int)[0] 
        
        # First burst : take all valid samples as no overlap with previous burst
        #Amended 17-Dec-2025: Why? To account for situation where due to
        #                          execessive overlap between SLC bursts the 
        #                          overlap, in either pre or post execceds the 
        #                          number of pre or post azimuth lines in the 
        #                          raw burst. So now removed overlap from 
        #                          first and last burst as well, since this is 
        #                          where issue manifests itself.
        # ..... probably easier to remove the if..elsif...else statement ....
        # ... but let's leave it in place in case something else goes wrong!
        if i==0:
            merged_start_line = (valid_burst_start_line 
                                + round(overlap_offset_lines[i]))
            merged_end_line   = (valid_burst_end_line
                                - round(overlap_offset_lines[i]))

        # Last burst different as no overlap with next burst so use all valid samples
        elif i==nBursts-1:
            merged_start_line = valid_burst_start_line + round(overlap_offset_lines[i-1])
            merged_end_line   = valid_burst_end_line   - round(overlap_offset_lines[i-1])

        # Otherwise for all other bursts set boundary as mid-point of overlap
        else:
            merged_start_line = (valid_burst_start_line
                                + round(overlap_offset_lines[i-1]))
            merged_end_line   = (valid_burst_end_line
                                - round(overlap_offset_lines[i]))
    
        valid_range_start = valid_burst_corners[i][0][1]
        valid_range_end   = valid_burst_corners[i][2][1]
        gridPoints.loc[gridPoints['burst_number'] == str(i+1),
                                  'merged_burst_start_line'] = merged_start_line
        gridPoints.loc[gridPoints['burst_number'] == str(i+1),
                                  'merged_burst_end_line']   = merged_end_line
        
        # Burst corners corrected for overlap and invalid lines and range samples
        burst_corners.append([(merged_start_line, valid_range_start),
                              (merged_end_line,   valid_range_start),
                              (merged_end_line,   valid_range_end),
                              (merged_start_line, valid_range_end)]
                              )
        # GPS coords of orginal bursts (overlap removed)
        burst_gps_corners.append([burst_start_long_lat[i,:],
                                  burst_start_long_lat[i+1,:],
                                  burst_end_long_lat[i+1,:],
                                  burst_end_long_lat[i,:]]
                                  )
    
    
    
    
        ## Must calculate gps coordinates for merged bursts.
        ## Grid Control Points(GCP) associated with first line of burst
        ## But have now shifted start of burst to mid-point of overlap between
        ## successive bursts so must concommitantly shift GCP location
        ## Use extrapolation based on known locations of original burst
        ## gps and size of non-overlapped burst
        ## NOTE: GPS locations in annotation file are ONLY given for the first line
        ##       of each burst, and then the final line of the last burst.
        ##       Hence the need to remove the overlap lines to ensure the 
        ##       GPS locations on the final line of a burst match with those 
        ##       on the succeeding line which is the first line of the following
        ##       burst.
        ##       When accounting for valid azimuth lines note that the overlap is 
        ##       now in terms of separation between lastValid line of burst i
        ##       and firstValid line of burst i+1 so in addition to subtracting
        ##       the overlap from the valid_burst_end_line of burst i must also
        ##       subtract the number of invalid lines at the start of burst i+1
        ##       to reach the first line of the i+1'th burst to which the gps 
        ##       locations are referenced.
        ##       
        ##       Location of approriate end line that matches with gps reference
        ##       points is the position of the last valid end line minus the
        ##       overlap with the first valid line of the next burst minus the
        ##       offset of that line from the start of its burst
        ##       This works even for the last burst where number_overlap_lines
        ##       is negative and offset_lines_to_start is 0.

        non_overlapped_end_line = valid_burst_end_line - number_overlap_lines[i] - offset_lines_to_start[i]
        #non_overlapped_end_line = valid_burst_end_line - number_overlap_lines[i]
        #Test code - remove when finished
        #print(f'Non-overlapped End Line = {non_overlapped_end_line}')
        #print(f'Number overlap lines = {number_overlap_lines[i]}')
        #print(f'Number offset lines = {offset_lines_to_start[i]}')


        non_overlapped_burst_corners.append([(burst_start_line, 0),
                                             (non_overlapped_end_line, 0),
                                             (non_overlapped_end_line, nRangeSamples-1),
                                             (burst_start_line, nRangeSamples-1)]
                                             )
        if i==0:   
            target0       = [merged_start_line, valid_range_start]
            merged_gps_sw = extrapolate(target0,
                                    np.array(non_overlapped_burst_corners[i]),
                                    np.array(burst_gps_corners[i])
                                    )
            
            target1       = [merged_end_line, valid_range_start]
            merged_gps_nw = extrapolate(target1,
                                    np.array(non_overlapped_burst_corners[i]),
                                    np.array(burst_gps_corners[i])
                                    )
            target2       = [merged_end_line, valid_range_end]
            merged_gps_ne = extrapolate(target2,
                                    np.array(non_overlapped_burst_corners[i]),
                                    np.array(burst_gps_corners[i])
                                    )
            target3       = [merged_start_line, valid_range_end]
            merged_gps_se = extrapolate(target3,
                                    np.array(non_overlapped_burst_corners[i]),
                                    np.array(burst_gps_corners[i])
                                    )
    
        else:
            target1       = [merged_end_line, valid_range_start]
            merged_gps_nw = extrapolate(target1,
                                    np.array(non_overlapped_burst_corners[i]),
                                    np.array(burst_gps_corners[i])
                                    )
            target2       = [merged_end_line, valid_range_end]
            merged_gps_ne = extrapolate(target2,
                                    np.array(non_overlapped_burst_corners[i]),
                                    np.array(burst_gps_corners[i])
                                    )
    
    
        if i==0:
            # Treat 1st burst differently
            merged_gps_corners.append([merged_gps_sw,
                                       merged_gps_nw,
                                       merged_gps_ne,
                                       merged_gps_se]
                                      )
        else:
            merged_gps_corners.append([merged_gps_corners[i-1][1],
                                      merged_gps_nw,
                                      merged_gps_ne,
                                      merged_gps_corners[i-1][2]]
                                      )
    
        ##### Add code for sub-grids
        ## Making assumption that sub-grid does not need to be amended to account 
        ## for valid range values as since burst corners and gps already corrected
        ## for this then use of sub-grid will be for locating objects that are 
        ## known to be in burst and will then be found using which ever sub-grid
        ## the object is located in
        boi1           = gridPoints[(gridPoints['burst_number'] == str(i+1))]
        boi2           = gridPoints[(gridPoints['burst_number'] == str(i+2))]
        long_lat_start = boi1[['longitude','latitude']].to_numpy().astype(float)
        long_lat_end   = boi2[['longitude','latitude']].to_numpy().astype(float)
        range_col      = boi1['pixel'].to_numpy().astype(int)
        grid_corners   = []
        grid_gps       = []
        non_overlapped_burst_grid_corners = []
        merged_gps_grid_corners = []
    
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
            ##
            ## Need to correct for merged grid gps positions similarly to above
            ## where corrected burst boundary gps locations. 
            ## But for the grid are appending along the Range direction, where as 
            ## for the burst was appending along azimuth direction. So must now
            ## extrapolate along range direction
            non_overlapped_burst_grid_corners.append([(burst_start_line, range_col[j]),
                                                      (non_overlapped_end_line, range_col[j]),
                                                      (non_overlapped_end_line,range_col[j+1]),
                                                      (burst_start_line, range_col[j+1])]
                                                      )
            target1 = [merged_start_line,range_col[j+1]]
    
            #print(f'GetSwatGPSGridBoundaries:: range_col size = {range_col.shape}, {range_col.dtype}, range_col = {range_col}')
            #print(f'GetSwatGPSGridBoundaries:: grid_corners[{j}] = {grid_corners[j]}')
            #print(f'GetSwatGPSGridBoundaries:: range_col[{j}] = {range_col[j]}') 
            #print(f'GetSwatGPSGridBoundaries:: non_ovrlapped_burst_grid_corners[{j}] = {non_overlapped_burst_grid_corners[j]}')
    
    
            merged_gps_grid_border_start = extrapolate(target1,
                                           np.array(non_overlapped_burst_grid_corners[j]),
                                           np.array(grid_gps[j]))
    
            target2 = np.array([merged_end_line,range_col[j+1]])
            merged_gps_grid_border_end  = extrapolate(target2,
                                          np.array(non_overlapped_burst_grid_corners[j]),
                                          np.array(grid_gps[j]))
            if j==0:
                #NB: '_border_end' & '_border_start" refer to end and start in Range direction
                merged_gps_grid_corners.append([long_lat_start[j,:],
                                                long_lat_end[j,:],
                                                merged_gps_grid_border_end,
                                                merged_gps_grid_border_start]
                                                )
            else:
                merged_gps_grid_corners.append([merged_gps_grid_corners[j-1][3],
                                      merged_gps_grid_corners[j-1][2],
                                      merged_gps_grid_border_end,
                                      merged_gps_grid_border_start]
                                      )
        burst_grid_corners.append(grid_corners)
        burst_grid_gps.append(grid_gps)
        burst_merged_grid_gps.append(merged_gps_grid_corners)
    
    #Define boundaries of SLC file 
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),
                   (nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    
    burst_info=[]
    burst_info.append(gps_corners)
    burst_info.append(burst_corners)
    burst_info.append(merged_gps_corners)
    #burst_info.append(burst_gps_corners)
    burst_info.append(burst_grid_corners)
    #burst_info.append(burst_grid_gps)
    burst_info.append(burst_merged_grid_gps)
    return burst_info







# Adapted to remove overlap from end of bursts , except last
# Function parses annotation file and returns Cartesian in [row,col] format
# and GPS locations for the swath boundaries, merged burst boundaries and 
# grid boundaries in the merged bursts
# 
#def getSwathBurstGridGPSBoundaries_test(annotation_file):
#def getSwathBurstGridGPSBoundaries(annotation_file):
def getSwathBurstGridGPSBoundaries_OriginalAndBest(annotation_file):
    
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        bursts.append({
            'azimuth_time': azimuth_time,
        })
    nLines         = int(root.find('.//numberOfLines').text)
    nSamples       = int(root.find('.//numberOfSamples').text)
    nBursts        = len(bursts)
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    nRangeSamples  = int(root.find('.//samplesPerBurst').text)
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
    #print(f'getSwathBurstGridGPSBoundaries::Number of bursts in'
    #      f'{annotation_file} is {nBursts}')
    #print(f'                                Number of azimuth lines in'
    #      f' each burst is {nLines}')
    #print(f'getSwathBurstGridGPSBoundaries:: Number of lines in each burst is {nBurstLines}')
    #print(f'getSwathBurstGridGPSBoundaries:: Number of Samples is {nRangeSamples}')
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
        #overlap_lines = round(overlap_time.total_seconds() / azTimeInterval)
        overlap_lines = (overlap_time.total_seconds() / azTimeInterval)
        # Convert back to string if necessary (optional)
        burst_end_time.append(t)
        number_overlap_lines.append(overlap_lines)
        #print(f'getSwathGPS..Boundaries::Number Overlap lines between burst'
        #      f' {i+1} & {i+2} is {number_overlap_lines[i]}')

    overlap_offset_lines = [x//2 for x in number_overlap_lines]
    merged_burst_start_line = []
    merged_burst_end_line   = []
    burst_corners           = []
    burst_gps_corners       = []
    non_overlapped_burst_corners = []
    gridPoints['merged_burst_start_line']= 0
    gridPoints['merged_burst_end_line']  = 0
    
    burst_ends           = gridPoints[(gridPoints['pixel'] 
                                 == str(nRangeSamples-1))]
    burst_starts         = gridPoints[(gridPoints['pixel'] == str(0))]
    burst_end_long_lat   = burst_ends[['longitude',
                                       'latitude']].to_numpy().astype(float)
    burst_start_long_lat = burst_starts[['longitude',
                                         'latitude']].to_numpy().astype(float)
    burst_grid_corners   = []
    burst_grid_gps       = []
    merged_gps_corners   = []
    burst_merged_grid_gps= []

    for i in range(0,nBursts):
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

        ## Must calculate gps coordinates for merged bursts.
        ## Grid Control Points(GCP) associated with first line of burst
        ## But have now shifted start of burst to mid-point of overlap between
        ## successive bursts so must concommitantly shift GCP location
        ## Use extrapolation based on known locations of original burst
        ## gps and size of non-overlapped burst

        non_overlapped_end_line = burst_end_line - number_overlap_lines[i]
        non_overlapped_burst_corners.append([(burst_start_line, 0),
                                             (non_overlapped_end_line, 0),
                                             (non_overlapped_end_line, nRangeSamples-1),
                                             (burst_start_line, nRangeSamples-1)]
                                             )
        target1                = [merged_end_line,0]
        merged_gps_border_start= extrapolate(target1, 
                                             np.array(non_overlapped_burst_corners[i]),
                                             np.array(burst_gps_corners[i]))
        target2                = [merged_end_line,nRangeSamples-1]
        #target2                = np.array([merged_end_line,nRangeSamples-1])
        merged_gps_border_end  = extrapolate(target2, 
                                             np.array(non_overlapped_burst_corners[i]),
                                             np.array(burst_gps_corners[i]))
        if i==0:
            # Treat 1st burst differently
            merged_gps_corners.append([burst_start_long_lat[i,:],
                                      merged_gps_border_start,
                                      merged_gps_border_end,
                                      burst_end_long_lat[i,:]]
                                      )
        else:
            merged_gps_corners.append([merged_gps_corners[i-1][1],
                                      merged_gps_border_start,
                                      merged_gps_border_end,
                                      merged_gps_corners[i-1][2]]
                                      )
        
        ##### Add code for sub-grids
        boi1           = gridPoints[(gridPoints['burst_number'] == str(i+1))]
        boi2           = gridPoints[(gridPoints['burst_number'] == str(i+2))]
        long_lat_start = boi1[['longitude','latitude']].to_numpy().astype(float)
        long_lat_end   = boi2[['longitude','latitude']].to_numpy().astype(float)
        range_col      = boi1['pixel'].to_numpy().astype(int)    
        grid_corners   = []
        grid_gps       = []
        non_overlapped_burst_grid_corners = []
        merged_gps_grid_corners = []

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
            ##
            ## Need to correct for merged grid gps positions similarly to above
            ## where corrected burst boundary gps locations. 
            ## But for the grid are appending along the Range direction, where as 
            ## for the burst was appending along azimuth direction. So must now
            ## extrapolate along range direction
            non_overlapped_burst_grid_corners.append([(burst_start_line, range_col[j]),
                                                      (non_overlapped_end_line, range_col[j]),
                                                      (non_overlapped_end_line,range_col[j+1]),
                                                      (burst_start_line, range_col[j+1])]
                                                      )
            target1 = [merged_start_line,range_col[j+1]]
           
            #print(f'GetSwatGPSGridBoundaries:: range_col size = {range_col.shape}, {range_col.dtype}, range_col = {range_col}')
            #print(f'GetSwatGPSGridBoundaries:: grid_corners[{j}] = {grid_corners[j]}')
            #print(f'GetSwatGPSGridBoundaries:: range_col[{j}] = {range_col[j]}') 
            #print(f'GetSwatGPSGridBoundaries:: non_ovrlapped_burst_grid_corners[{j}] = {non_overlapped_burst_grid_corners[j]}')


            merged_gps_grid_border_start = extrapolate(target1, 
                                           np.array(non_overlapped_burst_grid_corners[j]),
                                           np.array(grid_gps[j]))

            target2 = np.array([merged_end_line,range_col[j+1]])
            merged_gps_grid_border_end  = extrapolate(target2, 
                                          np.array(non_overlapped_burst_grid_corners[j]),
                                          np.array(grid_gps[j]))
            if j==0:
                #NB: '_border_end' & '_border_start" refer to end and start in Range direction
                merged_gps_grid_corners.append([long_lat_start[j,:],
                                                long_lat_end[j,:],
                                                merged_gps_grid_border_end,
                                                merged_gps_grid_border_start]
                                                )
            else:
                merged_gps_grid_corners.append([merged_gps_grid_corners[j-1][3],
                                      merged_gps_grid_corners[j-1][2],
                                      merged_gps_grid_border_end,
                                      merged_gps_grid_border_start]
                                      )
        burst_grid_corners.append(grid_corners)
        burst_grid_gps.append(grid_gps)
        burst_merged_grid_gps.append(merged_gps_grid_corners)
        
    #Define boundaries of SLC file 
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),
                   (nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    
    burst_info=[]
    burst_info.append(gps_corners)
    burst_info.append(burst_corners)
    burst_info.append(merged_gps_corners)
    #burst_info.append(burst_gps_corners)
    burst_info.append(burst_grid_corners)
    #burst_info.append(burst_grid_gps)
    burst_info.append(burst_merged_grid_gps)

    return burst_info



#
# Modification of getSwathBurstGPSBoundaries to now also return the 
# sub-grids within each burst in both [row,col] and gps(long,lat) formats
#
def getSwathBurstGridGPSBoundaries_gold(annotation_file):
#def getSwathBurstGridGPSBoundaries(annotation_file):
    
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        bursts.append({
            'azimuth_time': azimuth_time,

        })
    nLines         = int(root.find('.//numberOfLines').text)
    nSamples       = int(root.find('.//numberOfSamples').text)
    nBursts        = len(bursts)
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    nRangeSamples  = int(root.find('.//samplesPerBurst').text)
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
    print(f'Number of bursts in {annotation_file} is {nBursts}')
    print(f'Number of azimuth lines in each burst is {nLines}')
    print(f'Number of lines in each burst is {linesPerBurst}')
    print(f'Number of Samples is {nSamples}')
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
        overlap_lines = round(overlap_time.total_seconds() / azTimeInterval)
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

    for i in range(0,nBursts):
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
        range_col      = boi1[['pixel']].to_numpy().astype(int)    
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
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),
                   (nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    
    burst_info=[]
    burst_info.append(gps_corners)
    burst_info.append(burst_corners)
    burst_info.append(burst_gps_corners)
    burst_info.append(burst_grid_corners)
    burst_info.append(burst_grid_gps)
    #return gps_corners, burst_corners, burst_gps_corners, burst_grid_corners, burst_grid_gps, burst_info
    return burst_info

'''
def getAllSwathBoundaries(SLC_file_names, path):
    IW_gps=[]
    for fname in SLC_file_names:
        annotationFile = path + change_extension(fname,"xml")
        swath_gps=getSwathGPSBoundaries(annotationFile)
        IW_gps.append(swath_gps)
    return(IW_gps)

def getAllSwathBoundaries(SLC_file_names, path):
    IW_gps=[]
    for fname in SLC_file_names:
        annotationFile = path + change_extension(fname,"xml")
        print(f'AnnotationFile in getAllSwathBoundaries is: {annotationFile}')
        swath_gps=getSwathGPSBoundaries(annotationFile)
        IW_gps.append(swath_gps)
    return(IW_gps)

'''


# Function to determine if GPS location is within a ROI
# Used to determine if point in particualr swath and/or burst
def isPointInROI(point,ROI):
    roi           = np.array(ROI,dtype=float)
    area          = Polygon(roi)
    isTrueOrFalse = shapely.contains(area,Point(point))
    return isTrueOrFalse
    

def gps_to_cartesian(gps_target, gps_coords, cart_coords):
    # Prepare the input data for the transformation
    #gps_coords = np.array([gps1, gps2, gps3, gps4])
    #cart_coords = np.array([cart1, cart2, cart3, cart4])
    
    # Extract the components of GPS and Cartesian coordinates
    gps_lat = gps_coords[:, 0]
    gps_lon = gps_coords[:, 1]
    
    cart_x = cart_coords[:, 0]
    cart_y = cart_coords[:, 1]
    
    # Build the transformation matrix for latitude and longitude to Cartesian x and y
    A = np.array([
        [gps_lat[0], gps_lon[0], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[0], gps_lon[0], 1],
        [gps_lat[1], gps_lon[1], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[1], gps_lon[1], 1],
        [gps_lat[2], gps_lon[2], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[2], gps_lon[2], 1],
        [gps_lat[3], gps_lon[3], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[3], gps_lon[3], 1]
    ])
    
    B = np.array([
        cart_x[0],
        cart_y[0],
        cart_x[1],
        cart_y[1],
        cart_x[2],
        cart_y[2],
        cart_x[3],
        cart_y[3]
    ])
    
    # Solve for the transformation coefficients
    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    
    # Extract the coefficients for the transformations
    a_x, b_x, c_x, a_y, b_y, c_y = coeffs
    
    # Apply the transformation to the target GPS coordinate
    gps_lat_target = gps_target[0]
    gps_lon_target = gps_target[1]
    
    x_target = a_x * gps_lat_target + b_x * gps_lon_target + c_x
    y_target = a_y * gps_lat_target + b_y * gps_lon_target + c_y
    
    return (x_target, y_target)



def gps_to_cartesian_svd(gps_target, gps_coords, cart_coords):
    # Prepare the input data for the transformation
    #gps_coords = np.array([gps1, gps2, gps3, gps4])
    #cart_coords = np.array([cart1, cart2, cart3, cart4])
    
    # Extract the components of GPS and Cartesian coordinates
    gps_lat = gps_coords[:, 0]
    gps_lon = gps_coords[:, 1]
    
    cart_x = cart_coords[:, 0]
    cart_y = cart_coords[:, 1]
    
    # Build the transformation matrix for latitude and longitude to Cartesian x and y
    A = np.array([
        [gps_lat[0], gps_lon[0], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[0], gps_lon[0], 1],
        [gps_lat[1], gps_lon[1], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[1], gps_lon[1], 1],
        [gps_lat[2], gps_lon[2], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[2], gps_lon[2], 1],
        [gps_lat[3], gps_lon[3], 1, 0, 0, 0],
        [0, 0, 0, gps_lat[3], gps_lon[3], 1]
    ])
    
    B = np.array([
        cart_x[0],
        cart_y[0],
        cart_x[1],
        cart_y[1],
        cart_x[2],
        cart_y[2],
        cart_x[3],
        cart_y[3]
    ])
    
    # Perform Singular Value Decomposition on A
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Compute the pseudo-inverse of Sigma
    Sigma_inv = np.diag(1 / Sigma)
    
    # Compute the coefficients using SVD
    coeffs = Vt.T @ Sigma_inv @ U.T @ B
    
    # Extract the coefficients for the transformations
    a_x, b_x, c_x, a_y, b_y, c_y = coeffs
    
    # Apply the transformation to the target GPS coordinate
    gps_lat_target = gps_target[0]
    gps_lon_target = gps_target[1]
    
    x_target = a_x * gps_lat_target + b_x * gps_lon_target + c_x
    y_target = a_y * gps_lat_target + b_y * gps_lon_target + c_y
    
    return (x_target, y_target)

def cartesian_to_gps(cartesian_target, cartesian_coords, gps_coords):
    return gps_to_cartesian(cartesian_target, cartesian_coords, gps_coords)

# Function to determine location of target in coordinate system
# defined by output_space, given tie points with input space
def extrapolate(target, input_space, output_space):
    return cartesian_to_gps(target, input_space, output_space)

def getBurst(gps_location, burst_gps_grid):
    count = 0
    for i in range(0,len(burst_gps_grid)):
        if isPointInROI(gps_location, burst_gps_grid[i]):
            burst = i
            count +=1    
    if count != 1:
        print(f'ERROR: getBurst has found {count} bursts containing location :{gps_location}')
        return -1
    else:
        return burst
   


#
# Function to determine which of a series of grids a point belongs to
# gps_location ->(latitude, longitude)
# gps_grid     -> array of grid gps corners in format [(sw[lat,long], nw(lat,long),ne(lat,long),se(lat,long)] 
def getGrid(gps_location, gps_grid):
    count = 0
    for i in range(0,len(gps_grid)):   
        if isPointInROI(gps_location, gps_grid[i]):
            grid_number = i
            count +=1    
    if count != 1:
        print(f'ERROR: getGrid has found {count} grids containing location :{gps_location}')
        return -1
    else:
        return grid_number



# Given point in a burst determine its gps location
# Point's offset from starting azimuth and range needs to be determined based
# on offset of burst from start of file 
# Not working with suffcient accuracy - why?
def get_gps_of_point_in_burst(point, burst_number,annotation_file):
    #Get all grid and gps boundaries for all bursts
    burst_info = getSwathBurstGridGPSBoundaries(annotation_file)
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
    
    
    
    

def plotPointInBurst(point, burst_gps):
    fig2, ax2 = plt.subplots()
    fig.set_facecolor('xkcd:salmon')
    plt.style.context("dark_background")
    plt.style.use('dark_background')
    plt.tight_layout()
    x_max = (np.max(burst_gps.reshape([4,2])[:,1]))
    x_min = (np.min(burst_gps.reshape([4,2])[:,1]))
    y_max = (np.max(burst_gps.reshape([4,2])[:,0]))
    y_min = (np.min(burst_gps.reshape([4,2])[:,0]))
    ax2.set_ylim(x_max, x_min)
    ax2.set_xlim(y_max, y_min)
    ax.set_facecolor('xkcd:salmon')
    vertices=burst_gps
    rect = Polygon(vertices, facecolor='none', edgecolor='red')
    ax2.add_patch(rect)
    circle = Circle(point, radius=0.001, facecolor='green', edgecolor='black')
    ax2.add_patch(circle)
    plt.show()

# Function to return GPS boundaries of all swaths and all bursts
# in each swath
def getAllSwathBurstGPSBoundaries(SLC_file_names, path):
    swath_gps=[]
    swath_burst_gps=[]
    for fname in SLC_file_names:
        annotationFile = path + change_extension(fname,"xml")
        swath_info=getSwathBurstGridGPSBoundaries(annotationFile)
        swath_gps.append(swath_info[0])
        swath_burst_gps.append(swath_info[2])
    return  swath_gps, swath_burst_gps

def isPointInROI(point,ROI):
    roi           = np.array(ROI,dtype=float)
# Function to return GPS boundaries of all swaths
def getAllSwathBoundaries(SLC_file_names, path):
    IW_gps=[]
    for fname in SLC_file_names:
        annotationFile = path + change_extension(fname,"xml")
        swath_gps=getSwathGPSBoundaries(annotationFile)
        IW_gps.append(swath_gps)
    return(IW_gps)

def isPointInROI(point,ROI):
    roi           = np.array(ROI,dtype=float)
    area          = Polygon(roi)
    gps_point     = Point(point[0],point[1])
    isTrueOrFalse = shapely.contains(area,gps_point)
    #isTrueOrFalse = shapely.contains(area,Point(point))
    return isTrueOrFalse
    

# Function to return interpolated ship posiiton within burst
# Should be the same as getSHipInterpolatedPositionInGrid but data 
# format issues which have yet to be resolved necessitate a different
# type of reshaping for the gps_grid positions in the latter function
def getShipInterpolatedPositionInBurst(location, burst_gps, burst_boundary):
    reshaped_burst_gps      = np.array(burst_gps).reshape(4,2)
    reshaped_burst_boundary = np.array(burst_boundary).reshape(4,2)
    ship_row, ship_col      = gps_to_cartesian_svd(location, reshaped_burst_gps, reshaped_burst_boundary)
    
    # Check that interpolated value is within boundary and set to boundary if not
    if ship_row  < reshaped_burst_boundary[0][0]:
        ship_row = reshaped_burst_boundary[0][0]
    if ship_row  > reshaped_burst_boundary[1][0]:
        ship_row = reshaped_burst_boundary[1][0]
    if ship_col  < reshaped_burst_boundary[0][1]:
        ship_col = reshaped_burst_boundary[0][1]
    if ship_col  > reshaped_burst_boundary[3][1]:
        ship_col = reshaped_burst_boundary[3][1]
    
    return ship_row, ship_col


def getShipInterpolatedPositionInGrid(location, burst_grid_gps_boundary, burst_grid_boundary):
    reshaped_bggb       = np.array(burst_grid_gps_boundary).reshape(4,2)
    reshaped_bgb_test   = np.array(burst_grid_boundary).reshape(4,2)
    ship_row, ship_col  = gps_to_cartesian_svd(location, reshaped_bggb, reshaped_bgb_test)
    return ship_row, ship_col

"""  
    print(f' getShipInterpolatedPositionInGrid:: burst_grid_gps_boundary ........... ={burst_grid_gps_boundary}')
    print(f' getShipInterpolatedPositionInGrid:: re-shaped burst_grid_gps_boundary.. = {reshaped_bggb}')
    print(f' getShipInterpolatedPositionInGrid:: burst_grid_boundary = {burst_grid_boundary}')
    print(f' getShipInterpolatedPositionInGrid:: re-shaped burst_grid_boundary...... = {reshaped_bgb_test}')
    #reshaped_bgb        = np.array([[x[0], x[1][0]] for x in burst_grid_boundary]) 
    #print(f' getShipInterpolatedPositionInGrid:: re-shaped burst_grid_boundary = {reshaped_bgb}')
    # Check that interpolated value is within boundary and set to boundary if not
    if ship_row  < reshaped_bgb[0][0]:
        ship_row = reshaped_bgb[0][0]
    if ship_row  > reshaped_bgb[1][0]:
        ship_row = reshaped_bgb[1][0]
    if ship_col  < reshaped_bgb[0][1]:
        ship_col = reshaped_bgb[0][1]
    if ship_col  > reshaped_bgb[3][1]:
        ship_col = reshaped_bgb[3][1]
"""


# Function to extract only those ships that can be idenified as definitely
# belonging to a burst and a grid within the burst
# Unsure why a ship would be in a burst but not its associated grid
# Note: Find out!
def getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries):
    good_ship_coords = []
    for i in range(ship_coords.shape[0]):
        burst_index = getGrid(ship_coords[i, :], burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i, :], burst_grid_gps_boundaries[burst_index])
        if (burst_index != -1) and (subgrid_index != -1):
            good_ship_coords.append(ship_coords[i])
    return good_ship_coords  # Return the list of good ship coordinates



def create_roi(array, row_start, row_end, col_start, col_end):
    # Return the slice of the 2D array
    return array[row_start:row_end, col_start:col_end].copy()

def create_roi_from_polygon(array,boundary):
    row_start = boundary[0][0]
    row_end   = boundary[1][0]
    col_start = boundary[0][1]
    col_end   = boundary[3][1]
    return array[row_start:row_end,col_start:col_end].copy()


# function to accept 2 contiguous bursts and plot them
# based on burst start and end line in df
# burst_list -> list of 2 string values for burst numbers
# src -> image file
# azFraction -> controls fraction of azimuth extent to plot; centered on boundary of bursts:
#            1-> displays all of azimuth extent
#          0.5-> displays half of azimuth extent
# rangeFraction -> similar
#
def plotBursts(img_array, df, burstList, azFraction=1, rangeFraction=1):
    
    img=[]
    nRangeSamples = img_array.shape[1]
    burst1 = False
    for b in burstList:
        burst1 = not burst1
        
        #b_end   = gridPoints[gridPoints['burst_number']==str(b)]['merged_burst_end_line'].unique()[0].astype(int)
        #b_start = gridPoints[gridPoints['burst_number']==str(b)]['merged_burst_start_line'].unique()[0].astype(int)
        
        b_start = df[df['burst_number']== str(b)]['merged_burst_start_line'].unique()[0].astype(int)
        b_end   = df[df['burst_number']== str(b)]['merged_burst_end_line'].unique()[0].astype(int)
        
        if burst1:
            b_az_start = round(b_start + (1-azFraction)*(b_end - b_start))
            b_az_end   = b_end
        else:
            b_az_start = b_start
            b_az_end   = round(b_end -(1-azFraction)*(b_end - b_start))
            
        b_range_start  = round(nRangeSamples/2 * (1-rangeFraction))
        b_range_end    = round(nRangeSamples/2 * (1+rangeFraction))
        
        img.append(img_array[b_az_start:b_az_end, b_range_start:b_range_end])
        
    img = np.concatenate(img, axis=0)  
    img = np.real(np.sqrt(img*np.conjugate(img)))
    with plt.style.context("dark_background"):
        plt.figure(figsize=(20, 20))
        plt.title(f"SLC IW  {SAFE_dir}")
        plt.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 1000), aspect = '10')
    
    plt.xlabel("Range Samples")
    plt.ylabel("Azimuth Lines)")
    plt.show()
                
        

#
# Function to initially plot a single burst and place boundary boxes around ships located in burst
# Accepts SLC image data, burst(s) to plot, burst boundaries for all bursts, bounding boxes for all
# ships (shib_bounding_boxes) and the indices (ship_burst_indices) for determining which ships are 
# in what bursts 
def plotBurstsII(img_array, burstList, burst_boundaries, ship_bounding_boxes, ship_burst_indices, azFraction=1, rangeFraction=1):

    # Get bounding rectangles for ships
    img                    = []
    nRangeSamples          = img_array.shape[1]
    number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1
    burst_bounding_boxes   = np.array(ship_bounding_boxes)

    if nRangeSamples != number_of_range_pixels:
        print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information') 

    #with plt.style.context("dark_background"):
    #    plt.figure(figsize=(20, 20))
    #    plt.title(f"SLC IW  {SAFE_dir}")
    mpl.rcParams['figure.figsize'] = [14.0, 8.0]
    fig,(ax1, ax2) = plt.subplots(1,2)

    azimuth_offset = 0
    range_offset   = 0
    rec            = []
    for burst_number in burstList:
        # Get bounding boxes for ships only in this burst
        bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
        # Get burst boundary
        burst_boundary = burst_boundaries[burst_number-1]
        b_start        = burst_boundary[0][0]
        b_end          = burst_boundary[1][0]
        b_az_start     = round(b_start + (1-azFraction)*(b_end - b_start))
        b_az_end       = round(b_end   - (1-azFraction)*(b_end - b_start))
        b_range_start  = round(nRangeSamples/2 * (1-rangeFraction))
        b_range_end    = round(nRangeSamples/2 * (1+rangeFraction))
        # Extract burst image and copy to img
        img.append(img_array[b_az_start:b_az_end, b_range_start:b_range_end])
        img = np.concatenate(img, axis=0)
        img = np.real(np.sqrt(img*np.conjugate(img)))
        ax1.imshow(img,norm=colors.LogNorm(vmin=20,vmax = 700), aspect = '10')
        mask = np.zeros(img.shape, dtype=np.uint8)
        # Loop over each ship in burst and generate coordinates for bounding rectangle 
        # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
        # If plotting random bursts must subtract burst start position as offset for rectangles
        # Add end row of previous burst to position
        for j in range (0, bounding_boxes.shape[0]):
            row            = bounding_boxes[j][0][0] - b_az_start  -85
            column         = bounding_boxes[j][0][1]               -20
            h              = bounding_boxes[j][1][0] - bounding_boxes[j][0][0]
            w              = bounding_boxes[j][3][1] - bounding_boxes[j][0][1]
            # Check that ship locations are within viewing field set by azFraction and rangeFraction
            # If not do not plot them
            #if row in range(b_az_start,b_az_end) and column in range(b_range_start, b_range_end):
            if 0:
                print(f' In check for row col out of bounds')
                row+= azimuth_offset
                rec = patches.Rectangle((row, column),height = h, width =w,  edgecolor='red', facecolor='none', linewidth=50)
                print(f' Rec is ?? for j = {j} of {bounding_boxes.shape[0]} ship rectangles row = {row} col={column} h={h} w = {w}')
                ax.add_patch(rec)
                #rec.append(patches.Rectangle((row, column),height = h, width =w,  edgecolor='red', facecolor='none', linewidth=10))
            roi=create_roi(img, row, row+h,column, column +w)
            mask[row:row+h,column:column+w]=roi

            ########################################
            # Try cv2 rectangle - issue with rasterio to opencv conversion format
            #                     OpenCV doesn't like img 
            #start_point=(row,column)
            #end_point  =(row+h, column+w)
            #color=(0,0,255)
            #thickness = 10
            #img = cv2.rectangle(img, start_point, end_point, color, thickness)

        #img=img*mask

        # If plotting multiple bursts need to ensure ship bounding rectangles move with offset introduced by each successive burst 
        azimuth_offset = b_az_end - b_az_start
        range_offset   = 0
        ax1.imshow(img,norm=colors.LogNorm(vmin=20,vmax = 700), aspect = '10')

        img=img*mask
        ax2.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 1000), aspect = '10')

    plt.xlabel("Range Samples")
    plt.ylabel("Azimuth Lines)")
    plt.show()





#
# Function to initially plot a single burst and place boundary boxes around ships located in burst
# Accepts SLC image data, burst(s) to plot, burst boundaries for all bursts, bounding boxes for all
# ships (shib_bounding_boxes) and the indices (ship_burst_indices) for determining which ships are 
# in what bursts 
def plotBurstsIII(img_array, burstList, burst_boundaries, ship_bounding_boxes, ship_burst_indices, chip_border_size):

    # Get bounding rectangles for ships
    img                    = []
    nRangeSamples          = img_array.shape[1]
    number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1
    burst_bounding_boxes   = np.array(ship_bounding_boxes)

    if nRangeSamples != number_of_range_pixels:
        print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information') 

    mpl.rcParams['figure.figsize'] = [14.0, 8.0]
#    fig,(ax1, ax2) = plt.subplots(1,2)

    azimuth_offset = 0
    range_offset   = 0
    rec            = []
    for burst_number in burstList:
    #burst=burstList
    #if 1:
        # Get bounding boxes for ships only in this burst
        bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
        # Get burst boundary
        burst_boundary = burst_boundaries[burst_number-1]
        print(f' In plotBurstsIII   ...... burst boundary is : {burst_boundary}')
        print(f' Number of ships in this burst #{burst_number} is {bounding_boxes.shape}')
        az_start       = burst_boundary[0][0]
        az_end         = burst_boundary[1][0]
        range_start    = burst_boundary[0][1]
        range_end      = burst_boundary[3][1]
        # Extract burst image and copy to img
        img.append(img_array[az_start:az_end, range_start:range_end])
        img = np.concatenate(img, axis=0)
        img = np.real(np.sqrt(img*np.conjugate(img)))
        mask = np.zeros(img.shape, dtype=np.uint8)
        # Loop over each ship in burst and generate coordinates for bounding rectangle 
        # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
        # If plotting random bursts must subtract burst start position as offset for rectangles
        # Add end row of previous burst to position

        amplitude_img=np.abs(img_array[az_start:az_end, range_start:range_end])
        #ship_loc_img,binary_image_ground_truth, ship_chips, binary_ship_chips= generate_burst_binary_mask(img_array[az_start:az_end, range_start:range_end], burst_boundary, bounding_boxes, chip_border_size)
        ship_loc_img,binary_image_ground_truth, ship_chips, binary_ship_chips= generate_burst_binary_mask(amplitude_img, burst_boundary, bounding_boxes, chip_border_size)
        fig, ax = plt.subplots(4,1)
        ax[0].imshow(ship_chips.T,norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '1')
        ax[1].imshow(binary_ship_chips.T,cmap='Greys_r', aspect = '1')
        ax[2].imshow(binary_image_ground_truth,cmap='Greys_r', aspect = '10')
        plt.show()
        
        fig1, ax1 = plt.subplots()
        ax1.imshow(ship_loc_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
        plt.show()

        return ship_loc_img,binary_image_ground_truth, ship_chips, binary_ship_chips
#######################################################################################

def get_chips_and_masks(img_array,slc_id, burstList, burst_boundaries, 
                        ship_bounding_boxes, ship_burst_indices, 
                        box_size, chip_border_size):

    mpl.rcParams['figure.figsize'] = [14.0, 8.0]
    nRangeSamples          = img_array.shape[1]
    number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1
    if nRangeSamples != number_of_range_pixels:
        print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match '
              f'number of range samples: {number_of_range_pixels}, '
              f'calculated from burst boundary information')

    img                    = []
    burst_bounding_boxes   = np.array(ship_bounding_boxes)

    chip_row_size          = box_size + 2*chip_border_size
    chip_col_size          = box_size + 2*chip_border_size

    # Need to know the maximum number of ships per burst to set size of chips_from_all_bursts
    # to ensure successive bursts can be appended. Determine form incidence of bursts in 
    # ship_burst_indices     Note: need to import collections
    # Need to also take care of case where no ships in burst and 
    # 'ship_burst_indices' is an empty list
    counter                = collections.Counter(ship_burst_indices)
    most_common_items      = counter.most_common(1)
    if most_common_items:
        max_ships_per_burst = most_common_items[0][1]
    else:
        max_ships_per_burst = 1 #Assume at least 1 to give arrays correct size
        
    #max_ships_per_burst    = collections.Counter(ship_burst_indices).most_common(1)[0][1]
    chips_from_all_bursts  = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    chips_from_one_burst   = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    binary_chips_from_all_bursts  = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    binary_chips_from_one_burst   = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    first_time             = True
    for burst_number in burstList:
        #print(f"burst_number: {burst_number}, type: {type(burst_number)}")
        bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
        burst_boundary = burst_boundaries[burst_number-1]
        burst_img      = create_roi_from_polygon(img_array, burst_boundary)

        ship_loc_img, binary_image_ground_truth, ship_chips, binary_ship_chips= generate_burst_binary_mask(burst_img, burst_boundary, bounding_boxes, box_size, chip_border_size)
        ship_chips_transposed        = ship_chips.T
        binary_ship_chips_transposed = binary_ship_chips.T

        print(f'ship_chips_transposed has shape: {ship_chips_transposed.shape}'
              f' which is row and col size')
        ship_chips_row_size  = ship_chips_transposed.shape[0]
        ship_chips_col_size  = ship_chips_transposed.shape[1]
        chips_from_one_burst = chips_from_one_burst * 0
        binary_chips_from_one_burst = binary_chips_from_one_burst * 0
        print(f'chips_from_one_burst shape : {chips_from_one_burst.shape}')
        chips_from_one_burst[0:ship_chips_row_size, 0:ship_chips_col_size] = ship_chips_transposed
        binary_chips_from_one_burst[0:ship_chips_row_size, 0:ship_chips_col_size] = binary_ship_chips_transposed
        if first_time:
            chips_from_all_bursts = chips_from_one_burst
            binary_chips_from_all_bursts = binary_chips_from_one_burst
        else:
            chips_from_all_bursts = np.concatenate((chips_from_all_bursts, 
                                                    chips_from_one_burst), 
                                                    axis=0)
            binary_chips_from_all_bursts = np.concatenate(
                                                    (binary_chips_from_all_bursts, 
                                                     binary_chips_from_one_burst), 
                                                     axis=0)

        first_time = False
    plot_chips=False
    if plot_chips == True:
        fig1, ax1 = plt.subplots(1,2)
        ax1[0].imshow(chips_from_all_bursts, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
        ax1[1].imshow(binary_chips_from_all_bursts, cmap='Greys_r', aspect = '10')
        plt.show()
    
        fig, ax = plt.subplots()
        ax.imshow(chips_from_all_bursts, norm=colors.LogNorm(vmin=10,vmax = 300), 
                                                                    aspect = '1')
        plt.show()
    
        fig2, ax2 = plt.subplots(1,2)
        ax2[0].imshow(ship_loc_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
        ax2[1].imshow(binary_image_ground_truth,cmap='Greys_r', aspect = '10')
        plt.show()

    outputs=[]
    outputs.append(chips_from_all_bursts)
    outputs.append(binary_chips_from_all_bursts)
    outputs.append(ship_loc_img)
    outputs.append(binary_image_ground_truth)
    outputs.append(ship_chips)
    outputs.append(binary_ship_chips)

    return outputs 



def resample(arr, scale_factor):
    """
    Function to re-sample a 2D array along each row to a 
    multiplier of the current sampling frequency.
    Will result in new 2D array with same number of rows but
    number of columns changed to ncols*f_multiplier
    """
    nrows = arr.shape[0]
    ncols = arr.shape[1]
    t     = np.linspace(0,1,ncols)
    t_new = np.linspace(0,1, scale_factor)
    print(f'nrows = {arr.shape[0]}')
    new_arr=[]
    for row in range(0,nrows):
         line         = arr[row,:]
         interpolator = interp1d(t, line, kind='linear', fill_value='extrapolate')
         # Binary mask so need to round to keep binary
         interp_line  = np.round(interpolator(t_new))
         new_arr.append(interp_line)
    new_arr=np.array(new_arr)
    print(f'Number of rows in re-sampled mask is : {new_arr.shape[0]}')
    print(f'Number of cols in re-sampled mask is : {new_arr.shape[1]}')
    return(new_arr)
 

         
       
   
###############################################################################
# Function to generate binary mask for raw bursts and save mask
# Data must be re-sampled to same range sampling frequency as used in raw IW mode 
# All that is required then is to furthur divide each datafile into useful chips
# of size 3072 x 512 which is performed with xxxxx.py
#
def save_SLC_binary_burst_training_data(img_array,slc_id, burstList,
                                        burst_boundaries, ship_bounding_boxes, 
                                        ship_burst_indices, box_size, 
                                        chip_border_size, frequency_multiplier, iw_mode_index):

    nRangeSamples          = img_array.shape[1]
    number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1
    if nRangeSamples != number_of_range_pixels:
        print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match '
              f'number of range samples: {number_of_range_pixels}, '
              f'calculated from burst boundary information')

    img                    = []
    burst_bounding_boxes   = np.array(ship_bounding_boxes)
    for burst_number in burstList:
        bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
        burst_boundary = burst_boundaries[burst_number-1]
        burst_img      = create_roi_from_polygon(img_array, burst_boundary)
        try:
            ship_loc_img, binary_image_ground_truth, ship_chips, binary_ship_chips = generate_burst_binary_mask(
                                           burst_img, burst_boundary, bounding_boxes, box_size, chip_border_size)
        except Exception as e:
            print(f"Warning: generate_burst_binary_mask failed on burst#{burst_number} of {slc_id} with error: {e}")
            continue  # skip to the next burst

        # re-sample binary mask to sampling frequency of raw
        resampled_mask = resample(binary_image_ground_truth, frequency_multiplier) 
        np.save(f"{slc_id}_RAW_IW{iw_mode_index+1}_binary_mask_burst#{burst_number}", 
                   np.array(resampled_mask).astype(np.uint8))
        # 11-June_2025 ------------------------------------------
        # Added code to save slc burst, resampled to match raw
        # Save slc burst in slc_burst_directory
        slc_id_and_path = slc_id.replace('raw_burst_data','slc_burst_data')
        resampled_slc_burst = resample(burst_img, frequency_multiplier)
        np.save(f"{slc_id_and_path}_RAW_IW{iw_mode_index+1}_slc_burst#{burst_number}",
                   np.array(resampled_slc_burst))

    return(resampled_mask)


###############################################################################
# Function to generate binary mask for bursts and save mask and associated
# real and imaginary data.
# All that is required then is to furthur divide each datafile into useful chips
# of size 3072 x 512 which is performed with xxxxx.py
#
def save_burst_training_data(img_array,slc_id, burstList, burst_boundaries, 
                        ship_bounding_boxes, ship_burst_indices, 
                        box_size, chip_border_size):

    mpl.rcParams['figure.figsize'] = [14.0, 8.0]
    nRangeSamples          = img_array.shape[1]
    number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1
    if nRangeSamples != number_of_range_pixels:
        print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match '
              f'number of range samples: {number_of_range_pixels}, '
              f'calculated from burst boundary information')

    img                    = []
    burst_bounding_boxes   = np.array(ship_bounding_boxes)
    """
    chip_row_size          = box_size + 2*chip_border_size
    chip_col_size          = box_size + 2*chip_border_size

    # Need to know the maximum number of ships per burst to set size of chips_from_all_bursts
    # to ensure successive bursts can be appended. Determine form incidence of bursts in 
    # ship_burst_indices     Note: need to import collections

    max_ships_per_burst    = collections.Counter(ship_burst_indices).most_common(1)[0][1]
    chips_from_all_bursts  = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    chips_from_one_burst   = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    first_time             = True
    """
    for burst_number in burstList:
        bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
        burst_boundary = burst_boundaries[burst_number-1]
        burst_img      = create_roi_from_polygon(img_array, burst_boundary)

        ship_loc_img,binary_image_ground_truth, ship_chips, binary_ship_chips= generate_burst_binary_mask(burst_img, burst_boundary, bounding_boxes, box_size, chip_border_size)

        np.save(f"{slc_id}_real_burst#{burst_number}", 
                   np.real(burst_img).astype(np.int16))
        np.save(f"{slc_id}_imag_burst#{burst_number}", 
                   np.imag(burst_img).astype(np.int16))
        np.save(f"{slc_id}_binary_mask_burst#{burst_number}", 
                   np.array(binary_image_ground_truth).astype(np.uint8))


##########################################################################
# Code for thresholding ship chips
#
def set_inbetween_pixels(image):
    
    rows, cols = image.shape
    result = image.copy()
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if image[i, j] == 0:
                # Check horizontal neighbors
                if image[i, j-1] == 1 and image[i, j+1] == 1:
                    result[i, j] = 1
                # Check vertical neighbors
                elif image[i-1, j] == 1 and image[i+1, j] == 1:
                    result[i, j] = 1
    
    return result

# Miguel Alvarez's code for binary segmentation
# Modified by JC
def create_ship_binary_mask(ROI):
    
    min_ROI    = np.min(ROI)
    max_ROI    = np.max(ROI)
    avg_ROI    = np.mean(ROI)
    stddev_ROI = np.std(ROI)
    kernel1    = np.array([[1,1,1],
                           [1,1,1],
                           [1,1,1]])
    #rrrd = dilation(rrrd, kernel)
    
    threshold = avg_ROI + (1.0*stddev_ROI)
    #threshold = threshold_otsu(ROI)
    ROI2 = np.copy(ROI)
    if max_ROI - avg_ROI < 8 : 
        ROI_eq = exposure.equalize_hist(ROI)
        eq1av = np.mean(ROI_eq)
        ZQ = ROI_eq.copy()
        
        #ROI_eq[ROI_eq <= 0.99]=0
        #ROI_eq[ROI_eq >  0.99]=1
        
        for r in range(ROI.shape[0]):      # Iterate through rows in the ROI
            for c in range(ROI.shape[1]):  # Iterate through columns in the row
                if ROI_eq[r, c] > 0.99:
                    ROI_eq[r, c] = 1
                else:
                    ROI_eq[r, c] = 0
        ROI2 = ROI_eq 
        # apply erosion operation to remove noise, the dilate
        ROI2 = erosion(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        
    else:
        #AGREGAR THRESHOLD PARA CLAHE
        # Apply thresholding to create a binary mask
        #ROI[ROI <= threshold]=0
        #ROI[ROI >  threshold]=1
        for r in range(ROI.shape[0]):      # Iterate through rows in the ROI
            for c in range(ROI.shape[1]):  # Iterate through columns in the row        
                if ROI[r, c] > threshold:
                    ROI2[r, c] = 1
                else:
                    ROI2[r, c] = 0
        
        ROI2 = erosion(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        #ROI2 = dilation(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        ROI2 = dilation(ROI2, kernel1)
        ROI2 = erosion(ROI2, kernel1)
    return ROI2        
#filler           
#mask = set_inbetween_pixels(mask)
    
    
def create_roi(array, row_start, row_end, col_start, col_end):
    # Return the slice of the 2D array
    return array[row_start:row_end, col_start:col_end]

##########################################################################################################
# Function to take 1 burst image and create binary mask for burst
# img                 - image of burst already extracted from full SLC image,
#                       complex data
# burst_boundaries    - [row,col] co-ordinates of burst vertices in full SLC 
#                       image co-ordinates
# ship_bounding_boxes - [row,col] vertices of box around ship locations 
#                       relative to full SLC image vertices
# box_size            - size of bounding box around located ships
# border              - size of border to impose on bounding box mask
def generate_burst_binary_mask(image, burst_boundary, 
                               ship_bounding_boxes, box_size, border):

    # Convert complex image to amplitude only
    img                    = np.abs(image)
    img_mask               = np.zeros(img.shape,dtype=np.uint8)
    img_rectangle_mask     = np.zeros(img.shape,dtype=np.uint8)
    binary_img_mask        = np.zeros(img.shape,dtype=np.uint8)

    number_of_ships        = ship_bounding_boxes.shape[0]

    chip_size              = [box_size + 2*border, 
                              box_size + 2*border]
    print(f'generate_burst_binary_mask :: Chip size (chip_size) is : {chip_size}')
    ship_chip              = np.ones(chip_size, dtype= np.uint8)
    ship_chip_mask         = np.ones(chip_size, dtype= np.uint8)
    binary_ship_chip       = np.ones(chip_size, dtype= np.uint8)
    binary_ship_chip_mask  = np.ones(chip_size, dtype= np.uint8)
    rectangle_mask         = np.zeros(chip_size, dtype= np.uint8)
    dilated_rectangle_mask = np.ones(chip_size, dtype= np.uint8)

    burst_start_row       = burst_boundary[0][0]
    burst_end_row         = burst_boundary[1][0]
    burst_start_col       = burst_boundary[0][1]
    burst_end_col         = burst_boundary[2][1]
    row_offset            = round(border)
    col_offset            = round(border)

    kernel = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])

    for j in range (0, number_of_ships):
        row     = ship_bounding_boxes[j][0][0] - burst_start_row #-90 
        # WHY IS THIS OFFSET NEEDED? To do with overlap? ##
        if row < 0:
            print(f'generate_burst_binary_mask:: ERROR: ship #{j} at location [{ship_bounding_boxes[j][0][0]}]'
                  f',{ship_bounding_boxes[j][0][1]} is out of bounds with row '
                  f'= {row}')
            row = 0

        column  = ship_bounding_boxes[j][0][1]  - burst_start_col #- 25
        if column < 0:
            print(f'generate_burst_binary_mask:: ERROR: ship #{j} at location [{ship_bounding_boxes[j][0][0]}]'
                  f',{ship_bounding_boxes[j][0][1]} is out of bounds with column = {column}')
            column = 0

        h          = ship_bounding_boxes[j][1][0] - ship_bounding_boxes[j][0][0]
        w          = ship_bounding_boxes[j][3][1] - ship_bounding_boxes[j][0][1]
        roi        = create_roi(img,row, row+h, column, column+w)
        binary_roi = create_ship_binary_mask(roi)  
        binary_img_mask[row:row+h, column:column+w]  = np.copy(binary_roi)

        print(f' #################################################################################################')
        print(f' #################################################################################################')
        print(f' generate_burst_binary_mask:: roi shape for ship#{j+1} is {roi.shape}'   )
        img_mask[row:row+h, column:column+w] = np.copy(roi)

        # Paint roi into chip but if roi smaller than default size (which
        # implies it was within half the default size of the edge) then 
        # need to centre in the chip
        if (h < box_size) and (ship_bounding_boxes[j][0][0] == burst_start_row):
            chip_start_row = ship_chip.shape[0]-row_offset-h
            chip_end_row   = ship_chip.shape[0]-row_offset
        else:
            chip_start_row = row_offset
            chip_end_row   = row_offset + h

        if (w < box_size) and (ship_bounding_boxes[j][0][1] == burst_start_col):
            chip_start_col = ship_chip.shape[0]-col_offset-w
            chip_end_col   = ship_chip.shape[0]-col_offset
        else:
            chip_start_col = col_offset
            chip_end_col   = col_offset + w

        ship_chip[chip_start_row:chip_end_row, 
                  chip_start_col:chip_end_col]  = np.copy(roi)
        binary_ship_chip[chip_start_row:chip_end_row, 
                        chip_start_col:chip_end_col] = np.copy(binary_roi)

        if j == 0:
            ship_chip_mask         = np.copy(ship_chip.T)             # Is this 'if' statement really needed?
            binary_ship_chip_mask  = np.copy(binary_ship_chip.T)
        else:
            ship_chip_mask         = np.concatenate((ship_chip_mask, ship_chip.T),0)
            binary_ship_chip_mask  = np.concatenate((binary_ship_chip_mask, binary_ship_chip.T),0)

        #ship_chips_mask         = np.concatenate((ship_chips_mask, ship_chip.T),0)
        #binary_ship_chips_mask  = np.concatenate((binary_ship_chips_mask, binary_ship_chip.T),0)
        # Check that ship locations are within viewing field set by azFraction and rangeFraction

        # Need to highlight positions of ships in a mask : use dilation and subtraction to generate
        # outline of boundary and place in mask image
        rectangle_mask[row_offset:row_offset + h, col_offset:col_offset + w] = 1
        number_of_dilations = 4
        dilated_rectangle_mask = dilation(rectangle_mask, kernel)
        for k in range(0,number_of_dilations):
            dilated_rectangle_mask = dilation(dilated_rectangle_mask, kernel)

        # Create outline; Give outline high value; Make inner pixels =1
        dilated_rectangle_mask = (dilated_rectangle_mask - rectangle_mask)*3e4 + 1

        #dilated_rectangle_mask = dilated_rectangle_mask * 3e4          
        #dilated_rectangle_mask = dilated_rectangle_mask + 1           
        print(f' generate_burst_binary_mask:: Value of row {row} - row_offset {row_offset} is : '
              f' {row - row_offset},  h is {h}; column {column}-col_offset{col_offset}'
              f' is: {column-col_offset} w is {w}; dilated_rectangle_mask.shape'
              f' is : {dilated_rectangle_mask.shape}')
        ## Adding a kluge to get over some issue with assigning boundary box and 
        ## forcing area to be 100x100 

        # NOTE: row_offset equivalent to border
        if (row >= row_offset) and (burst_end_row - (row + h)) >= row_offset:
            start_row      = row - row_offset
            end_row        = row + row_offset + h
            rect_start_row = 0
            rect_end_row   = h + 2*row_offset
        elif (row < row_offset):
            start_row      = 0
            end_row        = row + row_offset + h
            rect_start_row = row_offset - row
            rect_end_row   = h + 2*row_offset
        else:
            start_row      = row - row_offset
            end_row        = burst_end_row
            rect_start_row = 0
            rect_end_row   = row_offset + (burst_end_row - row)

        
        if (column >= col_offset) and (burst_end_col - (column + w)) >= col_offset:
            start_col      = column - col_offset
            end_col        = column + col_offset + w
            rect_start_col = 0
            rect_end_col   = w + 2*col_offset
        elif (column < col_offset):
            start_col      = 0
            end_col        = column + col_offset + w
            rect_start_col = col_offset - column
            rect_end_col   = w + 2*col_offset
        else:
            start_col      = column - col_offset
            end_col        = burst_end_col
            rect_start_col = 0
            rect_end_col   = col_offset + (burst_end_col - column)

        """
        print(f' generate_burst_binary_mask:: For ship#{j+1}: row is {row}, col is {column}. start_row is {start_row}, end row is {end_row}')
        print(f' generate_burst_binary_mask:: Start column is {start_col}, end column is {end_col}')
        print(f' generate_burst_binary_mask:: rect_start_row is {rect_start_row}, rect_end_row is {rect_end_row}')
        print(f' generate_burst_binary_mask:: dilated_rectangle_mask size is : {dilated_rectangle_mask.shape}')
        print(f' generate_burst_binary_mask:: burst_boundaries are: {burst_boundary}')
        print(f' generate_burst_binary_mask:: burst size is {burst_boundary[1][0]-burst_boundary[0][0]} rows x {burst_boundary[2][1]-burst_boundary[1][1]} cols')
        """

        burst_row_height = burst_boundary[1][0]-burst_boundary[0][0] 
        if end_row <= burst_row_height:
            img_rectangle_mask[start_row:end_row, start_col:end_col] = dilated_rectangle_mask[rect_start_row:rect_end_row, rect_start_col:rect_end_col]
        else:
            print(f' generate_burst_binary_mask:: ERROR : Ship#{j} not included as boundary box crosses merged burst boundary')

        print(f' generate_burst_binary_mask:: img_rectangle_mask size is : {img_rectangle_mask.shape}')
                        
    # Multiply image mask with roi around ships by mask with bounding boxes. 
    # Should end up with original image with bounding box around each ship
    img_rectangle_mask = img_rectangle_mask + img
    #img_rectangle_mask = img
    if number_of_ships == 0:
        ship_chip_mask        = ship_chip_mask
        binary_ship_chip_mask = binary_ship_chip_mask

    return img_rectangle_mask, binary_img_mask, ship_chip_mask, binary_ship_chip_mask


###################################################################
# Get SLC files in format needed for subsequent functions. It's not
# pretty!
#
def getSLCFilesFromSafeDir(path, swath_number, pol='vh'):
    #safe_dir_path   = os.listdir(f'{path}/SLC/')
    src_path        = f'{path}/measurement/'
    if not os.path.isdir(src_path):
        print(f'Error : {path}measurement directory not found')
    src             = glob.glob(f'{src_path}*iw{swath_number}*{pol}*')[0]
    SLC_paths       = glob.glob(f'{src_path}*iw*{pol}*')
    SLC_filenames   = []
    for name in SLC_paths:
        SLC_filenames.append(os.path.basename(name))
    SLC_filenames.sort()
    annotation_path = f'{path}/annotation/'
    annotation_file = glob.glob(f'{annotation_path}*iw{swath_number}*{pol}*')[0]
    return src, annotation_file, annotation_path, SLC_filenames


###################################################################

###################################################################
# Get SLC files in format needed for subsequent functions. It's not
# pretty!
#
def getSLCFilesFromRoot(path, swath_number):
    safe_dir_path   = os.listdir(f'{path}/SLC/')
    src_path        = f'{path}/SLC/{safe_dir_path[0]}/measurement/'
    print(f' SAFEDir is : {safe_dir_path[0]}')
    src             = glob.glob(f'{src_path}*iw{swath_number}*vh*')[0]
    SLC_paths       = glob.glob(f'{src_path}*iw*vh*')
    SLC_filenames   = []
    for name in SLC_paths:
        SLC_filenames.append(os.path.basename(name))
    SLC_filenames.sort()
    annotation_path = f'{path}/SLC/{safe_dir_path[0]}/annotation/'
    annotation_file = glob.glob(f'{annotation_path}*iw{swath_number}*vh*')[0]
    return src, annotation_file, annotation_path, SLC_filenames, safe_dir_path[0]


###################################################################
# Get SLC files in format needed for subsequent functions. 
# Added error handling to getSLCFilesFromRoot
def get_slc_related_filenames(path, swath_number, polarization):
    """
    Function to extract SLC tiff and annotation filenames from root 
    directory
    Returns:  1 if all tiff and annotation file found
              0 if none found
             -1 if annotation but not tiff found 
    """
    if not (polarization == 'vh' or polarization =='vv'):
        print(f'ERROR: Incorrection polarization entered. {polarization} not valid!')
    if swath_number <1 or swath_number >3 :
        print(f'ERROR: Incorrect swath number entered: {swath_number} not valid, must be 1, 2, or 3')
    status = 1
    if os.path.exists(f'{path}/SLC'):
        safe_dir_list   = os.listdir(f'{path}/SLC/')
        safe_dir        = safe_dir_list[0]
        slc_tiff_path   = f'{path}/SLC/{safe_dir}/measurement/'
        slc_tiff_list   = glob.glob(f'{slc_tiff_path}*iw*{polarization}*')
        # Check that all 3 IW vh files present
        if len(slc_tiff_list) == 3:
            slc_tiff_file = glob.glob(f'{slc_tiff_path}*iw{swath_number}*vh*')[0]
            slc_filenames   = []
            for name in slc_tiff_list:
                slc_filenames.append(os.path.basename(name))
            slc_filenames.sort()
        else:
            print(f'ERROR: Function getSLCFilesFromRoot could not find all IW{swath_number} tiff files for {safe_dir}')
            print(f'       May need to download and extract original')
            status = 0
            slc_tiff_file=[]
            slc_filenames=[]
        # Check if annotation files present
        annotation_path = f'{path}/SLC/{safe_dir}/annotation/'
        if len(glob.glob(f'{annotation_path}*iw{swath_number}*vh*'))>0:
            annotation_file = glob.glob(f'{annotation_path}*iw{swath_number}*vh*')[0]
        else:
            print(f'ERROR: Function getSLCFilesFromRoot could not find IW{swath_number} annotation files for {safe_dir}')
            print(f'       May need to download and extract original')
            annotation_file =[]
            annotation_path=[]
            status = -1
    else:
        print(f'ERROR: Function getSLCFilesFromRoot -> Path to {path}/SLC/ : Not found. ') 
        status = 0
        annotation_file = []
        annotation_path = []
        slc_tiff_file   = []
        slc_filenames   = []
        safe_dir        = []

    all_file_info=[]
    all_file_info.append(slc_tiff_file)
    all_file_info.append(annotation_file)
    all_file_info.append(annotation_path)
    all_file_info.append(slc_filenames)
    all_file_info.append(safe_dir)
    all_file_info.append(status)
    return all_file_info

##############################################################################
# Get non-vessel co-ordinates & confidence for each burst in swath from label file
#
#
def getNonShipCoordinatesConfidence(label_fname, swath, SLC_ID):
    df1         = pd.read_csv(label_fname)
    ship_long   = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==False)
                 & (df1['SLC_product_identifier']== SLC_ID)]['detect_lon']
    ship_lat    = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==False)
                 & (df1['SLC_product_identifier']==SLC_ID)]['detect_lat']
    confidence  = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==False)
                 & (df1['SLC_product_identifier']==SLC_ID)]['confidence'].to_list()
    ship_coords = [ship_long, ship_lat]
    ship_coords = np.transpose(ship_coords)
    ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
    return ship_coords, confidence


##############################################################################
# Get ship co-ordinates & confidence for each burst in swath from label file
#
#
def getNonShipCoordinatesLengthConfidence(label_fname, swath, SLC_ID):
    df1         = pd.read_csv(label_fname)
    ship_long   = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==False)
                 & (df1['SLC_product_identifier']== SLC_ID)]['detect_lon']
    ship_lat    = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==False)
                 & (df1['SLC_product_identifier']==SLC_ID)]['detect_lat']
    confidence  = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==False)
                 & (df1['SLC_product_identifier']==SLC_ID)]['confidence'].to_list()
    length  = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==False)
                 & (df1['SLC_product_identifier']==SLC_ID)]['vessel_length_m'].to_list()
    ship_coords = [ship_long, ship_lat]
    ship_coords = np.transpose(ship_coords)
    ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
    return ship_coords, confidence, length


##############################################################################
# Get ship co-ordinates & length & confidence for each burst in swath from label file
#
#
def getShipCoordinatesLengthConfidence(label_fname, swath, SLC_ID):
    df1         = pd.read_csv(label_fname)
    ship_long   = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']== SLC_ID)]['detect_lon']
    ship_lat    = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']==SLC_ID)]['detect_lat']
    confidence  = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']==SLC_ID)]['confidence'].to_list()
    length  = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']==SLC_ID)]['vessel_length_m'].to_list()
    ship_coords = [ship_long, ship_lat]
    ship_coords = np.transpose(ship_coords)
    ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
    return ship_coords, confidence, length


def getShipCoordinatesConfidence(label_fname, swath, SLC_ID):
    df1         = pd.read_csv(label_fname)
    ship_long   = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']== SLC_ID)]['detect_lon']
    ship_lat    = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']==SLC_ID)]['detect_lat']
    confidence  = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']==SLC_ID)]['confidence'].to_list()
    ship_coords = [ship_long, ship_lat]
    ship_coords = np.transpose(ship_coords)
    ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
    return ship_coords, confidence


##############################################################################
# Get ship co-ordinates for each burst in swath from label file
#
#
def getShipCoordinates(label_fname, swath, SLC_ID):
    df1         = pd.read_csv(label_fname)
    ship_long   = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']== SLC_ID)]['detect_lon']
    ship_lat    = df1[(df1['swath_index'] == swath)
                 & (df1['is_vessel']==True)
                 & (df1['SLC_product_identifier']==SLC_ID)]['detect_lat']
    ship_coords = [ship_long, ship_lat]
    ship_coords = np.transpose(ship_coords)
    ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
    return ship_coords



##############################################################################
# Get ship bounding boxes
# Based on ship coordinates setup a bonding box centered on each ship
# that will be used to identify ship in image test data and also as template
# for binary image.
# Bounding box will be box_size x box_size in dimensions EXCEPT where ship is 
# located within box_size/2 of edge of burst and in these cases the size will be
# truncated to the distance of the ship to the edge + box_size/2 in row and 
# column space

def getShipBoundingBoxes(ship_coordinates, burst_boundaries,
                         burst_gps_boundaries, burst_grid_boundaries, 
                         burst_grid_gps_boundaries, box_size):
    # clearance around ship centroid for bounding box
    good_ship_coords = np.array(getGoodShipCoords(ship_coordinates,
                                burst_grid_gps_boundaries, 
                                burst_gps_boundaries)
                                )
    ship_coords=[]
    ship_coords = good_ship_coords
    print(f'Number of ships in swath is      : {ship_coordinates.shape}')
    print(f'Number of good ships in swath is : {good_ship_coords.shape}')
    number_of_ships_in_swath = ship_coords.shape[0]
    interp_ship_coords= []
    burst_indices     = []
    bounding_boxes    = []
    half_box_size     = box_size//2

    for i in range(0, number_of_ships_in_swath):
        burst_index   = getGrid(ship_coords[i,:], 
                                burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i,:], 
                                burst_grid_gps_boundaries[burst_index])

        if (burst_index == -1) or (subgrid_index == -1) :
            print(f'ERROR: Ship location {ship_coords[i]} :burst_index'
                  f' = {burst_index}, subgrid_index = {subgrid_index} '
                  f' not found correctly - not adding to ship locations')
        else:
            interp_row, interp_col = getShipInterpolatedPositionInGrid(
                                     ship_coords[i,:], 
                                     burst_grid_gps_boundaries[burst_index][subgrid_index],
                                     burst_grid_boundaries[burst_index][subgrid_index]
                                     )
            ship_row      = round(interp_row)
            ship_col      = round(interp_col)
            interp_ship_coords.append([ship_row, ship_col])
            interp_coords = np.array(interp_ship_coords)
            burst_indices.append(burst_index)
            print(f'Ship Location = {ship_coords[i]} '
                  f'Interpolated Ship Position in Burst #{burst_index+1} '
                  f'and Grid #{subgrid_index+1:02d} is: {interp_coords[i]} ')

            burst_start_row = burst_boundaries[burst_index][0][0]
            burst_end_row   = burst_boundaries[burst_index][1][0] 
            burst_start_col = burst_boundaries[burst_index][0][1]
            burst_end_col   = burst_boundaries[burst_index][2][1]

            # Check if ship close to edges of burst, 
            # if not give full bounding box size,
            # if within half box size of edge only extend bounding box to edge
            if ((ship_row - burst_start_row > half_box_size)
                and (burst_end_row - ship_row > half_box_size)):
                box_start_row = ship_row - half_box_size + 1
                box_end_row   = ship_row + half_box_size

            elif ((ship_row - burst_start_row <= half_box_size)
                and (burst_end_row - ship_row > half_box_size)):
                box_start_row = burst_start_row
                box_end_row   = ship_row + half_box_size
                       
            elif ((ship_row - burst_start_row > half_box_size)
                and (burst_end_row - ship_row <= half_box_size)):
                box_start_row = ship_row - half_box_size + 1
                box_end_row   = burst_end_row

            else:
                print(f'ERROR: getShipBoundingBoxes -> ship location issue.'
                      f' Burst size too small')

            if ((ship_col - burst_start_col > half_box_size)
                and (burst_end_col - ship_col > half_box_size)):
                box_start_col = ship_col - half_box_size + 1
                box_end_col   = ship_col + half_box_size

            elif ((ship_col - burst_start_col <= half_box_size)
                and (burst_end_col - ship_col > half_box_size)):
                box_start_col = burst_start_col
                box_end_col   = ship_col + half_box_size
                       
            elif ((ship_col - burst_start_col > half_box_size)
                and (burst_end_col - ship_col <= half_box_size)):
                box_start_col = ship_col - half_box_size + 1
                box_end_col   = burst_end_col

            else:
                print(f'ERROR: getShipBoundingBoxes -> ship location issue.'
                      f' Burst size too small?')


        bounding_box =np.array([[box_start_row, box_start_col],
                                [box_end_row,   box_start_col],
                                [box_end_row,   box_end_col],
                                [box_start_row, box_end_col]])
        bounding_boxes.append(bounding_box)
    return bounding_boxes, burst_indices
    


    # Have bounding box for each ship in swath in np.array


####################################################################################
# Code to plot bursts : 22-Jan-2025



def plot_IW2_RAW_SLC(SLC_ID, burst_to_plot):

    scene_id = get_scene_id(SLC_ID)
    if scene_id:
        print(f'Scene_id is: {scene_id}')
    else:
        print(f'scene_id failed')
        sys.exit()
    swath_of_interest = 2
    polarization      ='vh'
    swath_of_interest_index = swath_of_interest - 1
    IW_mode = swath_of_interest_index

    # Get SLC file to match raw based on mapping file
    scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"

    #scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_train.csv'
    mapping_df         = pd.read_csv(scene_mapping_file)

    #raw_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Train"
    #slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_TrainingData"
    raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
    slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"


    #save_path = "/mnt/smbshare/tempTest/Data/Burst_Training_Data/"
    #save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/raw_burst_training/"
    save_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/"

    if not (os.path.exists(raw_path)):
	    print(f' Path to {raw_path} inaccessible: Exiting!')
	    sys.exit()
    else:
	    print(f' Path to {raw_path} found')

    for raw_directory in [scene_id]:
        raw_root_file_path   = f'{raw_path}/{raw_directory}/RAW/'
        if os.listdir(raw_root_file_path) and contains_SAFE_file(raw_root_file_path):
            raw_safe_dir  = os.listdir(raw_root_file_path)
            raw_src_path  = f'{raw_root_file_path}{raw_safe_dir[0]}/'
            raw_filenames = glob.glob(f'{raw_src_path}*{polarization}*.dat')
            # Need to sort filenames to access correct .dat file
            raw_filenames.sort(reverse=True)
            raw_filename = raw_filenames[0]
            if not (os.path.exists(raw_filename[0])):
                print(f'Error: Filename {raw_filename} not found. Continuing to next raw file ....')
                continue
            else:
                raw_iw_data       = get_raw_burst_info(raw_filename)
                print(f' File {raw_filename} found. ')
        else:
            print(f'Error: Raw file path {raw_root_file_path} not found or does not contain .SAFE directory. Continuing to next raw file ......')
            continue
    	########################################################################### 
    	### Have matching SLC file
        slc_id = SLC_ID
        SLC_root_file_path = f'{slc_path}/{slc_id}'
        SLC_root_file_path = SLC_root_file_path.replace("__","_")
        #p1,p2,p3,p4,p5    = getSLCFilesFromRoot(, swath_of_interest)
        slc_file_info     = get_slc_related_filenames(SLC_root_file_path, swath_of_interest, polarization)
        slc_src_file      = slc_file_info[0]
        annotation_file   = slc_file_info[1]
        annotation_path   = slc_file_info[2]
        SLC_fnames        = slc_file_info[3]
        SAFE_dir          = slc_file_info[4]
        get_slc_info_status = slc_file_info[5]
    
        if get_slc_info_status != 1:
    	    print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')
    	    continue
    
        print(f' Annotation file is: {annotation_file}')
        print(f' Annotation_dir  is: {annotation_path}')
        print(f' SLC_fname(s) are  : {SLC_fnames}')
        print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
       	Annotation_dir = annotation_path
       	SLC_safe_id, _ = os.path.splitext(SAFE_dir)
       	label_file     = scene_mapping_file
       	#############################################################################
       	if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir)
       					   and os.path.exists(slc_src_file)):
       	    print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
       	else:
       	    print(f'ERROR: {annotation_file} or {Annotation_dir} '
       		  f' or {slc_src_file} Not found. Exiting')
       	    continue
       	    #exit(0)
       
       	# Have SLC src and annotation file and raw filename
       	###########################################################################
       
       	###########################################################################
       	# get binary masks for SLC bursts
       	ship_coords = getShipCoordinates(label_file, swath_of_interest, SLC_safe_id)
       	burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
       	IW_gps_boundary           = burst_information[0]
       	burst_boundaries          = burst_information[1]
       	burst_gps_boundaries      = burst_information[2]
       	burst_grid_boundaries     = burst_information[3]
       	burst_grid_gps_boundaries = burst_information[4]
       	number_of_bursts          = len(burst_boundaries)
       	bounding_box_size         = 100
       	border                    = 25
       	swath_bounding_boxes, burst_indices = getShipBoundingBoxes(ship_coords,
       						 burst_boundaries,
       						 burst_gps_boundaries,
       						 burst_grid_boundaries,
       						 burst_grid_gps_boundaries,
       						 bounding_box_size)
       	# Choose burst of interest
       	n = len(burst_boundaries)
       	burst_counts = [burst_indices.count(i) for i in range(0, n)]
       	# Print the table
       	print("Burst Number:\t" ,  "\t".join(map(str, range(1, n + 1))))
       	print("#Ships:      \t",  "\t".join(map(str, burst_counts)))
       	burst_of_interest = burst_counts.index(max(burst_counts)) + 1
       	#slc_burst         = burst_of_interest
       	slc_burst         = burst_to_plot
       	############################################################################
       	# Load real and imaginary components of burst
       	burst_real_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_real_burst#{slc_burst}.npy"
       	burst_imag_file = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_imag_burst#{slc_burst}.npy"
       	burst_bin_file  = f"{save_path}{SLC_safe_id}_RAW_IW{IW_mode+1}_binary_mask_burst#{slc_burst}.npy"
       	if os.path.exists(burst_real_file) and os.path.exists(burst_imag_file) and os.path.exists(burst_bin_file):
       	    burst_real = np.load(burst_real_file)
       	    burst_imag = np.load(burst_imag_file)
       	    burst_bin  = np.load(burst_bin_file)
       	    print(f'All burst files found')
       	else:
       	    print(f'ERROR : At least one raw burst data file not found in {save_path}. Exiting')
       	    exit(0)
       	###########################################################################
       
       
       	
       	if os.path.exists(slc_src_file):
       	    print(f"{SLC_fnames[swath_of_interest_index]} found")
       	else:
       	    print(f"ERROR: {slc_src_file} Not found. Exiting")
       	    #exit(0)
       
       	src            = rasterio.open(slc_src_file)
       	img            = src.read(1)
       
       	bursts_to_plot = [slc_burst]
       	output=get_chips_and_masks(img, SLC_safe_id,
       				 bursts_to_plot, burst_boundaries,
       				 np.array(swath_bounding_boxes),
       				 burst_indices, bounding_box_size, border
       				)
       
       
       
        burst_chips        = output[0]
        binary_burst_chips = output[1]
        final_burst_img    = output[2]
        final_binary_img   = output[3]
        final_burst_chips  = output[4]
        final_burst_binary_chips = output[5]
    
        print(f'All data available for plotting ..... is it plotting?   ..... IS IT????')
    
        br_min  = burst_real.min()
        br_max  = burst_real.max()
        br_mean = burst_real.mean()
        bi_min  = burst_imag.min()
        bi_max  = burst_imag.max()
        bi_mean = burst_imag.mean()
    
        fig1, ax1 = plt.subplots(1,2)
        ax1[0].imshow(burst_real, vmin=br_min,vmax = br_max, aspect = '10')
        #ax1[0].imshow(burst_real, norm=colors.LogNorm(vmin=br_min,vmax = br_max), aspect = '10')
        ax1[1].imshow(burst_imag, vmin=bi_min, vmax = bi_max, aspect = '10')
        ax1[0].set_title(f'Real Component of Burst #{slc_burst}')
        ax1[1].set_title(f'Imaginary Component of Burst #{slc_burst}')
        fig1.suptitle(f'{SLC_safe_id}', y=0.8)
        plt.show()
    
    
    
        fig2, ax2 = plt.subplots(1,2)
        ax2[0].imshow(final_burst_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
        ax2[1].imshow(burst_bin, cmap='Greys_r', aspect = '10')
        ax2[0].set_title(f'SLC Amplitude #{slc_burst} Max = {final_burst_img.max():.2f}')
        ax2[1].set_title(f'Binary Mask for  Burst #{slc_burst}')
        fig2.suptitle(f'{SLC_safe_id}', y=0.8)
    
        plt.show()
    
    
        amplitude = np.sqrt(burst_real*burst_real + burst_imag*burst_imag)
        phase    = np.arctan2(burst_imag, burst_real)
        fig3, ax3 = plt.subplots(1,2)
        ax3[0].imshow(amplitude, norm=colors.LogNorm(vmin=1,vmax = 20), aspect = '10')
        ax3[1].imshow(phase, cmap='Greys_r',vmin=-3.5, vmax=3.5, aspect = '10')
        ax3[0].set_title(f'Amplitude for Burst #{slc_burst} Max={amplitude.max():.2f}')
        ax3[1].set_title(f'Phase for Burst #{slc_burst}')
    
        fig3.suptitle(f'{SLC_safe_id}', y=0.85)
        plt.show()
    
        fig4, ax4 = plt.subplots(2,2)
        ax4[0,0].imshow(amplitude, norm=colors.LogNorm(vmin=1,vmax = 20), aspect='auto')
        ax4[0,1].imshow(phase, cmap='Greys_r',vmin=-3.5, vmax=3.5, aspect='auto')
        ax4[0,0].set_title(f'Amplitude for Burst #{slc_burst} Max={amplitude.max():.2f}')
        ax4[0,1].set_title(f'Phase for Burst #{slc_burst}')
    
        ax4[1,0].imshow(final_burst_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect='auto')
        ax4[1,1].imshow(burst_bin, cmap='Greys_r', aspect='auto')
        ax4[1,0].set_title(f'SLC Amplitude #{slc_burst} Max = {final_burst_img.max():.2f}')
        ax4[1,1].set_title(f'Binary Mask for  Burst #{slc_burst}')
        fig4.suptitle(f'{SLC_safe_id}', y=0.95)
        plt.show()
        print(f'Real Avg     = {burst_real.mean():.3f}')
        print(f'Real Std Dev = {burst_real.std():.3f}')
        print(f'Real Max     = {burst_real.max():.3f}')
        print(f'Real Min     = {burst_real.min():.3f}')
        print(f'Imag Avg     = {burst_imag.mean():.3f}')
        print(f'Imag Std Dev = {burst_imag.std():.3f}')
        print(f'Imag Max     = {burst_imag.max():.3f}')
        print(f'Imag Min     = {burst_imag.min():.3f}')
        print(f'Amplitude Max     = {amplitude.max():.3f}')
        print(f'Amplitude Min     = {amplitude.min():.3f}')
        print(f'Amplitude Mean    = {amplitude.mean():.3f}')
        print(f'Amplitude Std Dev = {amplitude.std():.3f}')


###############################################################################
# Function to return single burst of SLC data
# Need SLC_Product_ID and Burst Number, 
# Assumes IW2 and 'vh' polarization unless otherwise passed in
# Assumes SLC SAFE directory is in /home/joe/Misc/tempTest/Data/SLC_Data

def get_single_burst_image_data(SLC_ID, burst_number, iw_mode=2, pol='vh'):

    
    ##swath_of_interest = 2
    #polarization      ='vh'
    
    polarization      = pol
    swath_of_interest_index = iw_mode - 1
    IW_mode = swath_of_interest_index


    scene_mapping_file = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
    #scene_mapping_file = '/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/SLC_train.csv'
 

    #raw_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Train"
    #slc_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_TrainingData"
    raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data"
    slc_path = "/home/joe/Misc/tempTest/Data/SLC_Data"


    #save_path = "/mnt/smbshare/tempTest/Data/Burst_Training_Data/"
    #save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/raw_burst_training/"
    save_path = "/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts/"

 
    ########################################################################### 
    ### Have matching SLC file
    slc_id = SLC_ID
    SLC_root_file_path = f'{slc_path}/{slc_id}'
    SLC_root_file_path = SLC_root_file_path.replace("__","_")
    slc_file_info     = get_slc_related_filenames(SLC_root_file_path, iw_mode, polarization)
    slc_src_file      = slc_file_info[0]
    annotation_file   = slc_file_info[1]
    annotation_path   = slc_file_info[2]
    SLC_fnames        = slc_file_info[3]
    SAFE_dir          = slc_file_info[4]
    get_slc_info_status = slc_file_info[5]

    if get_slc_info_status != 1:
        print(f'ERROR : Issue with {slc_id}: Missing tiff or annotation file')


    print(f' Annotation file is: {annotation_file}')
    print(f' Annotation_dir  is: {annotation_path}')
    print(f' SLC_fname(s) are  : {SLC_fnames}')
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    Annotation_dir = annotation_path
    SLC_safe_id, _ = os.path.splitext(SAFE_dir)
    label_file     = scene_mapping_file
    #############################################################################
    if (os.path.exists(annotation_file) and os.path.exists(Annotation_dir)
                                       and os.path.exists(slc_src_file)):
        print(f'{annotation_file} and {Annotation_dir} and {slc_src_file} found')
    else:
        print(f'ERROR: {annotation_file} or {Annotation_dir} '
              f' or {slc_src_file} Not found. Exiting')
        #exit(0)

    # Have SLC src and annotation file and raw filename
    ###########################################################################

    ###########################################################################
    # Get burst locations in SLC image data
    burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
    burst_boundaries  = burst_information[1]
    
    src            = rasterio.open(slc_src_file)
    img            = src.read(1)
    burst_index    = burst_number - 1
    burst_boundary = burst_boundaries[burst_index]
    burst_img      = create_roi_from_polygon(img, burst_boundary)
    return burst_img


def get_slc_range_blackfill_indices(slc_burst, threshold_level=0.5):
    """
    Parameters
    ----------
    slc_burst : TYPE
        Script to find where in the range direction blackfill starts and ends
        Sums along azimuth direction to find average starting and ending index
        Then sets start and end as index where threshold exceeded
    Returns
    -------
    Indices of start, end and imblance on either side

    """
    range_sum      = np.sum(slc_burst, axis=0)
    range_sum_mean = range_sum.mean()
    threshold      = range_sum_mean*threshold_level
    start_index    = np.nonzero(range_sum>threshold)[0][0]
    end_index      = np.nonzero(range_sum>threshold)[0][-1]
    imbalance      = (slc_burst.shape[1] - end_index) - start_index
    return start_index, end_index, imbalance



############################################################################
# Functions for plotting slc and raw files and location therein

def find_raw_safe_dir(slc_safe_dir, raw_dir):
   #get start time of slc file
   time_format = '%Y%m%dT%H%M%S'
   #split and extract data and time 
   _dir, fname = os.path.split(slc_safe_dir)
   if fname =='':
       date_time   = _dir.split('_')[5]
   else:
       date_time   = fname.split('_')[5]
   slc_date_time = datetime.strptime(date_time, time_format)
   
   #find raw safe files that start less than 4s earlier than slc
   #making assumption that only 1 raw file can exist within 5s of start of slc
   # Ensure only parsing '.SAFE' files 
   for f in os.listdir(raw_dir):
       if 'SAFE' in f:
           match_found = False
           date_time = f.split('_')[5]
           raw_date_time = datetime.strptime(date_time, time_format)
           #Ensure SLC and RAW files within 5 seconds of each other
           if (np.abs((slc_date_time - raw_date_time).total_seconds()) < 5):
               match_found = True
               break
   if match_found == False:
       print(f'Error: No raw SAFE dir found matching SLC start time for {slc_safe_dir}')
       sys.exit(0)
   else:
       raw_safe_dir = f'{raw_dir}/{f}'
       return raw_safe_dir 
    
   
def get_raw_file_in_safe_dir(safe_dir, pol):
    file_found = False 
    for f in os.listdir(safe_dir):
        if pol in f and 'index' not in f and 'annot' not in f:
            file_found = True
            break

    if file_found == False:
        print(f'Error: No raw raw file matching criteria found in {safe_dir}')
    else:
        raw_file = f'{safe_dir}/{f}'
        return raw_file 

    
def get_raw_filename_matching_slc_filename_new(slc_safe_dir, raw_dir, pol):
    # Modified 21 March 2026 to deal with directory structure that
    # has RAW SAFE and SLC SAFE directories at same level 
    # raw_safe_dir = find_raw_safe_dir(slc_safe_dir, raw_dir)
    raw_safe_dir = raw_dir
    raw_filename = get_raw_file_in_safe_dir(raw_safe_dir, pol)
    return raw_filename
   
def get_raw_filename_matching_slc_filename(slc_safe_dir, raw_dir, pol):
    
    raw_safe_dir = find_raw_safe_dir(slc_safe_dir, raw_dir)
    raw_filename = get_raw_file_in_safe_dir(raw_safe_dir, pol)
    return raw_filename
   
    
def get_raw_slc_annotation_filenames(raw_dir, slc_safe_dir, location, pol):
        
    # Get slc image and annotation file names and directories
    # Use getSLCFilesFromSafeDir function even though don't yet know which swath
    # is of interest. Using '2' as a dummy placeholder.
    slc_file_info   = getSLCFilesFromSafeDir(slc_safe_dir,2,pol)
    annotation_path = slc_file_info[2]
    slc_path        = annotation_path.replace('annotation','measurement')
    slc_filenames   = slc_file_info[3]

    #raw_safe_dir = f'{raw_dir}'
    #raw_file = get_raw_filename_matching_slc_filename(slc_safe_dir, raw_dir, pol)
    raw_file = get_raw_filename_matching_slc_filename_new(slc_safe_dir, raw_dir, pol)
    
    # Get gps boundaries of all swaths
    
    swath_gps_boundaries = getAllSwathBoundaries(slc_filenames, 
                                                 annotation_path
                                                 )
    # Find swath location is in
    swath_index = getGrid(location, swath_gps_boundaries)
    # Get SLC src and annotation file names
    slc_src_file    = slc_path + slc_filenames[swath_index]
    annotation_file = annotation_path+change_extension(slc_filenames[swath_index],
                                                        "xml"
                                                       )
    # 8th Jan 2026: For completeness ... find burst as well..
    burst_info = getSwathBurstGridGPSBoundaries(annotation_file)
    burst_gps_boundaries = burst_info[2]
    burst_index = getGrid(location, burst_gps_boundaries)
  
    return raw_file, slc_src_file, annotation_file, swath_index, burst_index
    
# Function mainly used to plot GPS posiitons of ships in raw and slc files
# Amended 14-12-2025 to input ship information -name, speed, bearing, orbital 
# node 
def plot_raw_slc_data(fnames_and_swath, location, pol, info=[], save_fig=0, save_path =''):
    
    raw_file        = fnames_and_swath[0]
    slc_src_file    = fnames_and_swath[1]
    annotation_file = fnames_and_swath[2]
    swath_index     = fnames_and_swath[3]
    
    # Find burst in swath containing location 
    burst_info           = getSwathBurstGridGPSBoundaries(annotation_file)
    burst_boundaries     = burst_info[1]
    burst_gps_boundaries = burst_info[2]
    burst_index          = getGrid(location, burst_gps_boundaries)
    
    if burst_index == -1:
        raise ValueError(f'*** Error: Location {location} not found in '
                         f'{slc_src_file} with boundaries '
                         f'{burst_info[0]} ... Exiting ****')
        
    
    burst_grid_boundaries     = burst_info[3] 
    burst_grid_gps_boundaries = burst_info[4]
    grid_index                = getGrid(location, 
                                        burst_grid_gps_boundaries[burst_index]
                                        ) 
    if grid_index == -1:
        raise ValueError(f'*** Error: Location {location} not found in '
                         f' {slc_src_file} burst#{burst_index+1} with '
                         f'burst boundaries '
                         f'{burst_gps_boundaries[burst_index]}'
                         f' ... Exiting ****'
                         )
    
    # find [row, col] location in swath for gps location using burst gps boundaries
    interp_row, interp_col = getShipInterpolatedPositionInBurst(
                              location,
                              burst_gps_boundaries[burst_index],
                              burst_boundaries[burst_index]
                              )
    burst_based_row = round(interp_row)
    burst_based_col = round(interp_col)
    
    
    # find [row, col] location in swath for gps location using grid gps boundaries
    interp_row, interp_col = getShipInterpolatedPositionInGrid(
                              location,
                              burst_grid_gps_boundaries[burst_index][grid_index],
                              burst_grid_boundaries[burst_index][grid_index]
                              )
    grid_based_row = round(interp_row)
    grid_based_col = round(interp_col)
    
    # Convert swath based [row,col] to burst co-ordinates by subtracting starting 
    # row and starting column
    burst_start_row = burst_boundaries[burst_index][0][0]
    burst_start_col = burst_boundaries[burst_index][0][1]
    location1       = [burst_based_row - burst_start_row,
                       burst_based_col - burst_start_col
                       ]
    location2       = [grid_based_row  - burst_start_row,
                       grid_based_col  - burst_start_col
                       ]
    # Correct for range sampling frequency difference between RAW and SLC
    sampling_freq_mul= freq_multipliers[swath_index]
    location1        = [int(location1[0]), int(location1[1]*sampling_freq_mul)]
    location2        = [int(location2[0]), int(location2[1]*sampling_freq_mul)]
    
    
    # Are they the same??
    # No - why not? Check burst_boundary values!!!!
    # Get slc burst
    burst_number   = burst_index + 1
    slc_burst_data = get_slc_burst_from_files(slc_src_file, annotation_file, 
                                              burst_number
                                              )
    non_resampled_slc_burst = slc_burst_data[0]
    
    # Is this slc burst re-sampled to match raw sampling rate?
    scaling_factor = int(sampling_freq_mul * non_resampled_slc_burst.shape[1])
    slc_burst      = resample(non_resampled_slc_burst, scaling_factor)
    #It is now!
    
    # plot raw and slc burst
    
    raw_burst_data = get_slc_matched_raw_burst_from_file(raw_file,
                                annotation_file,
                                burst_number, swath_index +1, pol
                                )
    slc_matched_raw_burst = raw_burst_data[0]
    raw_burst    = raw_burst_data[1]
    az_offset    = raw_burst_data[2]
    az_end       = raw_burst_data[3]
    range_offset = raw_burst_data[4]
    range_end    = raw_burst_data[5]
    meta_data    = raw_burst_data[6]
    p_l0file     = raw_burst_data[7]
    range_length   = range_end-range_offset
    azimuth_length = az_end-az_offset
    roi = np.array([range_offset, range_length, az_offset, azimuth_length])
    


    img_name =(f"{slc_src_file.split('/')[9].split('.')[0]}"
               f"_IW{swath_index+1}_Burst#{burst_index +1}")
  

    #### Try range compression
    range_padding = 1000
    az_padding    = 200
    padded_raw_burst = np.pad(slc_matched_raw_burst, ((az_padding,az_padding),
                               (range_padding,range_padding)),
                               mode='constant', constant_values=0
                               )
    rc_padded_burst = range_compress_burst(padded_raw_burst,
                                           iw_mode = swath_index+1
                                           )
    resized_rc_padded_burst=rc_padded_burst[
                        az_padding:az_padding+slc_matched_raw_burst.shape[0],
                        range_padding:range_padding+slc_matched_raw_burst.shape[1]
                        ]
    del rc_padded_burst
    rc_padded_burst = resized_rc_padded_burst
    #plot_2d_array(intensity(rc_padded_burst),'rc_padded_burst')
   
    #SLC_ID = slc_src_file.split('/')[11].split('.')[0]
    safe_dir    = slc_src_file.split('/')[9]
    clean_name  = safe_dir.replace('.SAFE', '')
    # 3. Split by underscore and slice the last two items
    # [-2:] means "take the last two items"
    parts       = clean_name.split('_')
    SLC_ID      = "_".join(parts[-2:])
    date_T_time = parts[5]
    satellite   = parts[0]
    
    #Plot RAW and SLC zoomed in around location
    slc_img_name = f'{satellite}_{SLC_ID}_{date_T_time}_SLC'
    raw_img_name = f'{satellite}_{SLC_ID}_{date_T_time}_RAW'
    #slc_fig      = plot_2d_array(np.square(np.abs(slc_burst)),slc_img_name)
    #raw_fig      = plot_2d_array(np.real(slc_matched_raw_burst),raw_img_name,
    #                             'linear'
    #                             )    
    
    
    
    
    fig, axs  = plt.subplots(3, 1, figsize=(16,9),constrained_layout=True) 
    if not info:
        plot_fig_name = f'{satellite}_{SLC_ID}_[{location[0]:.3f},{location[1]:.3f}]_{date_T_time}'
    else:
        plot_fig_name = f'{satellite}_{SLC_ID}_[{location[0]:.3f},{location[1]:.3f}]_Name_{info[0]}_speed_{info[1]}knph_bearing_{info[2]}Deg_TravelTime_{info[3]:.2f}hours_{info[4]}_IW{swath_index+1}_Burst#{burst_number}_{date_T_time}'
    fig.suptitle(f'{plot_fig_name}')
    font_size =8
    
    img_name=['SLC', 'RAW', 'RC_RAW']
    plot_type=['log_normal', 'linear', 'log_normal']
    sar_data_to_plot=[intensity(slc_burst), np.real(slc_matched_raw_burst), 
                      intensity(rc_padded_burst)
                      ]
    az_sgn = -1 if location[0] < azimuth_length / 2 else 1
    range_sgn = -1 if location[1] > range_length / 2 else 1
    
    for i in range(0,3):
    
        grid_colour='gray' if i==0 or i==2 else 'red'
        plot_2d_array_on_ax(axs[i], sar_data_to_plot[i],img_name[i],
                            plot_type[i],font_size, grid_on=1, 
                            grid_colour=grid_colour
                            )
        if i==0 or i==2:
            # Highlight ship:  Place circle around ship
            text_offset=[az_sgn*200,range_sgn*2000]
            #add_text_to_plot(axs[i], f'Burst based Object Labelled @ [{location1[0]},'
            #             f'{location1[1]}]',
            #             location1,text_offset, colour='white'
            #             )
            #circle = patches.Circle((location1[1], location1[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
            #axs[i].add_patch(circle)
            # Add text for detected position of ship
            text_offset=[text_offset[0]*-1,text_offset[1]+100]
            add_text_to_plot(axs[i], f'Grid Based Object labelled @ [{location2[0]:.0f},'
                         f'{location2[1]:.0f}]',
                         location2,text_offset,
                         colour='white', arrow_colour='green'
                         )
            circle = patches.Circle((location2[1],
                                     location2[0]),
                                    radius=50, edgecolor='green',
                                    facecolor='none', linewidth=1
                                    )
            axs[i].add_patch(circle)
    
                
    
    # Create roi for magnifed area around ship
    roi_rows_cols = [600,1200]
    roi_centre    = location2
    roi_row_start,roi_col_start = place_roi(slc_burst,roi_rows_cols, roi_centre)
    slc_roi = create_roi(slc_burst,
                         roi_row_start, 
                         roi_row_start + roi_rows_cols[0],
                         roi_col_start,
                         roi_col_start + roi_rows_cols[1],
                         )
    raw_roi = create_roi(slc_matched_raw_burst,
                         roi_row_start, 
                         roi_row_start + roi_rows_cols[0],
                         roi_col_start,
                         roi_col_start + roi_rows_cols[1],
                         )
    
    fig1, axs1 = plt.subplots(1,2, figsize=(16,9),constrained_layout=True) 
    fig1.suptitle(f"{slc_img_name.replace('_SLC','_SLC_&_RAW ')}")
    plot_2d_array_on_ax(axs1[0], np.square(np.abs(slc_roi)),slc_img_name,
                        'log_normal', font_size, grid_on=1, 
                        grid_colour=grid_colour
                        )
    
    plot_2d_array_on_ax(axs1[1], np.real(raw_roi),raw_img_name,
                        'linear', font_size, grid_on=1, 
                        grid_colour=grid_colour
                        )
    
    if save_fig:
        print(f'Saving {plot_fig_name} to {save_path}')
        fig.savefig(f'{save_path}{plot_fig_name}', format='jpg', dpi=600)
        fig1.savefig(f"{save_path}{slc_img_name.replace('_SLC','_SLC_&_RAW ')}.jpg", format='jpg', dpi=600)
    
    
    
    print(f'**** {satellite}_{SLC_ID}_{date_T_time} Successfully processed *******')    
    return fig, fig1, burst_number    



def intensity(arr):
    return np.square(np.abs(arr))

def place_roi(A: np.ndarray, roi_shape: tuple, p: tuple):
    """
    Place a region of interest (ROI) of given shape centrally at location p in 2D numpy array A.
    If ROI extends beyond array edges, position it to include the edge fully.

    Args:
        A (np.ndarray): 2D numpy array
        roi_shape (tuple): (num_cols, num_rows) size of region of interest
        p (tuple): (row, col) coordinate where ROI is to be centered

    Returns:
        starting_row (int): starting row index of ROI within A
        starting_col (int): starting column index of ROI within A
        new_centre        : location in crop of where desired point ends up
    """

    num_rows, num_cols = roi_shape  # note: roi_shape is (num_cols, num_rows) as per your problem
    total_rows, total_cols = A.shape

    row_c, col_c = p  # The location p is (row, col)

    # Calculate half sizes (use integer division)
    half_rows = num_rows // 2
    half_cols = num_cols // 2

    # Starting row and col with +1 as in your example
    starting_row = row_c - half_rows + 1
    starting_col = col_c - half_cols + 1
    new_centre_r = half_rows
    new_centre_c = half_cols

    # Correct start if ROI extends beyond top/left edges
    if starting_row < 0:
        starting_row = 0
        new_centre_r = row_c
    if starting_col < 0:
        starting_col = 0
        new_centre_c = col_c

    # Correct start if ROI extends beyond bottom/right edges
    if starting_row + num_rows > total_rows:
        starting_row = total_rows - num_rows
        new_centre_r = row_c - starting_row
    if starting_col + num_cols > total_cols:
        starting_col = total_cols - num_cols
        new_centre_c = col_c - starting_col

    # Ensure no negative indices after correction (in case ROI bigger than array)
    starting_row = max(0, starting_row)
    starting_col = max(0, starting_col)

    return int(starting_row), int(starting_col), [new_centre_r, new_centre_c]









def place_roi_old(A: np.ndarray, roi_shape: tuple, p: tuple):
    """
    Place a region of interest (ROI) of given shape centrally at location p in 2D numpy array A.
    If ROI extends beyond array edges, position it to include the edge fully.

    Args:
        A (np.ndarray): 2D numpy array
        roi_shape (tuple): (num_cols, num_rows) size of region of interest
        p (tuple): (row, col) coordinate where ROI is to be centered

    Returns:
        starting_row (int): starting row index of ROI within A
        starting_col (int): starting column index of ROI within A
    """

    num_rows, num_cols = roi_shape  # note: roi_shape is (num_cols, num_rows) as per your problem
    total_rows, total_cols = A.shape

    row_c, col_c = p  # The location p is (row, col)

    # Calculate half sizes (use integer division)
    half_rows = num_rows // 2
    half_cols = num_cols // 2

    # Starting row and col with +1 as in your example
    starting_row = row_c - half_rows + 1
    starting_col = col_c - half_cols + 1

    # Correct start if ROI extends beyond top/left edges
    if starting_row < 0:
        starting_row = 0
    if starting_col < 0:
        starting_col = 0

    # Correct start if ROI extends beyond bottom/right edges
    if starting_row + num_rows > total_rows:
        starting_row = total_rows - num_rows
    if starting_col + num_cols > total_cols:
        starting_col = total_cols - num_cols

    # Ensure no negative indices after correction (in case ROI bigger than array)
    starting_row = max(0, starting_row)
    starting_col = max(0, starting_col)

    return int(starting_row), int(starting_col)


# Timerelated functions
def convert_string_to_datetime(dt_string: str) -> datetime:
    """
    Converts a datetime string in 'YYYYMMDDTHHMMSS' format 
    (e.g., '20250925T181446') into a Python datetime object.
    
    Args:
        dt_string: The input string.
        
    Returns:
        A datetime.datetime object.
    """
    # Define the format code that exactly matches the input string structure.
    # %Y - Year (4 digits)
    # %m - Month (01-12)
    # %d - Day (01-31)
    # T  - The literal character 'T'
    # %H - Hour (24-hour clock)
    # %M - Minute
    # %S - Second
    format_code = "%Y%m%dT%H%M%S"
    
    # Use datetime.strptime() to parse the string according to the format
    datetime_object = datetime.strptime(dt_string, format_code)
    
    return datetime_object

def round_to_nearest_second(dt: datetime) -> datetime:
    # Rounds date_time object to nearest second
    # Microseconds cutoff for rounding up is 500,000 (half a second) 
    if dt.microsecond >= 500000:
        # Round up: Add 1 second and then set microseconds to 0
        dt_rounded = dt + timedelta(seconds=1)
        # Use .replace(microsecond=0) to reset the microseconds to zero
        return dt_rounded.replace(microsecond=0)
    else:
        # Round down: Just set microseconds to 0
        return dt.replace(microsecond=0)

def get_acq_time_from_SLC_SAFE_filename(f):
    """ 
    Determine acquisiiton time from time stamps in
    SLC SAFE Dir filename. Use mid-point.
    Must have SAFE name format as input argument
    Eg:S1A_IW_SLC__1SDV_20250925T181446_20250925T181513_061148_079F93_E9BB.SAFE
        
    Returns mid-time of acquisition rounded to nearest second
    """
    start_time_str  = f.split('_')[5]
    end_time_str    = f.split('_')[6]
    start_date_time = convert_string_to_datetime(start_time_str)
    end_date_time   = convert_string_to_datetime(end_time_str)
    half_time_diff  = (end_date_time - start_date_time)/2
    mid_date_time   = start_date_time + half_time_diff
    acq_time        = round_to_nearest_second(mid_date_time)
    return acq_time


#Function to perform polynomial interpolation based on least squares
#minimization of error
def get_polynomial_fit(ais_data_file_path, obsv_time, order):
    #Open AIS data csv file and convert to dataframe
    if not os.path.exists(ais_data_file_path):
        print(f'File not found: {ais_data_file_path}')
        print(f'Exiting!')
        sys.exit(1)
    else:        
        df = pd.read_csv(ais_data_file_path)
        df['freshness_dt'] = pd.to_datetime(df['freshness'], utc=True)
        #Localize acq_time to UTC for a proper comparison
        # The 'Z' on the column means it's UTC, so acq_time must also be UTC-aware for comparison.
        acq_time_tz = pytz.utc.localize(obsv_time)
        # Filter for entries strictly GREATER THAN acq_time
        condition = df['freshness_dt'] > acq_time_tz
        try:
            target_index = df[condition].index[0]
        except IndexError:
            print("Error: Could not find any row where 'freshness' is strictly greater than 'acq_time'.")
            sys.exit(1)
        
        # --- Main Extraction Steps ---
        
        # Get the integer position (iloc) of the target index label
        target_iloc = df.index.get_loc(target_index)
        
        # Calculate the slice boundaries (3 rows before to 2 rows after 
        # so 6 in total) 
        # start_iloc: 3 positions before the target, 
        # ensuring it doesn't go below 0
        start_iloc = max(0, target_iloc - 3)
        # end_iloc: two positions after the target. For iloc slicing, 
        # the end is exclusive, so we use +3
        end_iloc = min(len(df), target_iloc + 3)
        
        # Define the columns to extract
        columns_to_extract = ['freshness', 'latitude', 'longitude',
                              'speed', 'course'
                             ]
        
        # Use .iloc to slice the DataFrame by integer position and then 
        # select columns
        result_slice = df.iloc[start_iloc:end_iloc][columns_to_extract]
        times         = pd.to_datetime(result_slice['freshness'].iloc[0:6],
                                       utc=True
                                       )
        time_deltas  = times - times.iloc[0]
        times_hours  = time_deltas.dt.total_seconds().to_numpy()/3600
        latitudes    = result_slice['latitude'].to_numpy()
        latitudes    = np.reshape(latitudes,(len(latitudes),1))
        longitudes   = result_slice['longitude'].to_numpy()
        longitudes   = np.reshape(longitudes,(len(longitudes),1))
        col1         = np.ones((len(latitudes),1))
        A            = np.zeros((len(latitudes),order))
        for i in range(order):
            col=times_hours**i
            A[:,i] = col
        
        b1 = latitudes
        b2 = longitudes
        
        # Perform Singular Value Decomposition on A
        U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

        # Compute the pseudo-inverse of Sigma
        Sigma_inv = np.diag(1 / Sigma)

        # Compute the coefficients using SVD
        lat_coeffs = Vt.T @ Sigma_inv @ U.T @ b1
        lon_coeffs = Vt.T @ Sigma_inv @ U.T @ b2

        #Get interpolated position P(t)= c0t^0+c1t^1+c2t^2+c3t^3
        #Convert obsv_time to hours
        obsv_time_obj = (acq_time_tz 
                      - pd.to_datetime(result_slice['freshness'].iloc[0],
                                       utc=True)
                      )
        obsv_time_hours = obsv_time_obj.total_seconds()/3600
        
        
        obsv_lat=0
        obsv_lon = 0
        for i in range(order):
            obsv_lat += lat_coeffs[i]*obsv_time_hours**i
            obsv_lon += lon_coeffs[i]*obsv_time_hours**i
            
            
    return obsv_lat.tolist(), obsv_lon.tolist()
           




# Function to return cartesian coordinates with a roi (burst) based
# on gps location and grids defining boundaries

def get_cartesian_location_in_burst(gps_loc, burst, annotation_file):

    burst_info           = getSwathBurstGridGPSBoundaries(annotation_file)
    burst_boundaries     = burst_info[1]
    burst_gps_boundaries = burst_info[2]
    burst_index          = getGrid(gps_loc, burst_gps_boundaries)
    swath_index          = int(os.path.basename(annotation_file).split('-')[1][2]) - 1
    
    if burst_index == -1:
        raise ValueError(f'*** Error: Location {location} not found in '
                         f'{slc_src_file} with boundaries '
                         f'{burst_info[0]} ... Exiting ****')
        
    
    burst_grid_boundaries     = burst_info[3] 
    burst_grid_gps_boundaries = burst_info[4]
    grid_index                = getGrid(gps_loc, 
                                        burst_grid_gps_boundaries[burst_index]
                                        ) 
    if grid_index == -1:
        raise ValueError(f'*** Error: Location {location} not found in '
                         f' {slc_src_file} burst#{burst_index+1} with '
                         f'burst boundaries '
                         f'{burst_gps_boundaries[burst_index]}'
                         f' ... Exiting ****'
                         )
    
    # find [row, col] location in swath for gps location using burst 
    # gps boundaries
    interp_row, interp_col = getShipInterpolatedPositionInBurst(
                              gps_loc,
                              burst_gps_boundaries[burst_index],
                              burst_boundaries[burst_index]
                              )
    burst_based_row = round(interp_row)
    burst_based_col = round(interp_col)
    
    
    # find [row, col] location in swath for gps location using grid gps boundaries
    interp_row, interp_col = getShipInterpolatedPositionInGrid(
                              gps_loc,
                              burst_grid_gps_boundaries[burst_index][grid_index],
                              burst_grid_boundaries[burst_index][grid_index]
                              )
    grid_based_row = round(interp_row)
    grid_based_col = round(interp_col)
    
    # Convert swath based [row,col] to burst co-ordinates by subtracting starting 
    # row and starting column
    burst_start_row = burst_boundaries[burst_index][0][0]
    burst_start_col = burst_boundaries[burst_index][0][1]
    location1       = [burst_based_row - burst_start_row,
                       burst_based_col - burst_start_col
                       ]
    location2       = [grid_based_row  - burst_start_row,
                       grid_based_col  - burst_start_col
                       ]
    # Correct for range sampling frequency difference between RAW and SLC
    sampling_freq_mul= freq_multipliers[swath_index]
    location1        = [int(location1[0]), int(location1[1]*sampling_freq_mul)]
    location2        = [int(location2[0]), int(location2[1]*sampling_freq_mul)] 
    
    # NOTE: location2 based on sub-grids within burst generally more accurate
    #       so return it.
    return location2





def plot_I_Q_stats(raw_burst, slc_burst, ship_location, crc_burst, save_path):

    do_visual_check = False
    burst_crop_data = []
    crop_size      = [512, 512]
    ship_crop_size = [150, 150]
    kernel1         = np.array( [[1,1,1],
                                 [1,1,1],
                                 [1,1,1]
                                 ]
                                 ) 
    raw_real_burst      = np.real(raw_burst)
    raw_imag_burst      = np.imag(raw_burst)
    slc_intensity       = intensity(slc_burst)
    slc_intensity_filtered = exclude_high_outliers(slc_intensity,10)
    r,c, ship_loc_in_crop = place_roi(raw_real_burst, crop_size, ship_location)

    cropped_raw_real_burst = raw_real_burst[r:r + crop_size[0],
                                            c:c + crop_size[1]]
    cropped_raw_imag_burst = raw_imag_burst[r:r + crop_size[0],
                                            c:c + crop_size[1]]
    #SLC is complex, 
    cropped_slc_burst = slc_burst[r:r + crop_size[0],
                                 c:c + crop_size[1]]
    #Get crop around ship of 150x150, ship located in crop at ship_loc_in_crop
    #crop_centre = [crop_size[0]//2, crop_size[1]//2]
    r,c, pos_in_crop= place_roi(cropped_raw_real_burst, ship_crop_size,
                                 ship_loc_in_crop
                                 )

    ship_crop_raw_real = cropped_raw_real_burst[r:r + ship_crop_size[0],
                                 c:c + ship_crop_size[1]]
    ship_crop_raw_imag = cropped_raw_imag_burst[r:r + ship_crop_size[0],
                                 c:c + ship_crop_size[1]]
    ship_crop_slc      = cropped_slc_burst[r:r + ship_crop_size[0],
                                 c:c + ship_crop_size[1]]
    ship_crop_slc_I    = np.real(ship_crop_slc)
    ship_crop_slc_Q    = np.real(ship_crop_slc)
    ship_crop_slc_intensity = intensity(ship_crop_slc)


    #Get pixels of crop outside of ship crop
    #Create mask to exclude ship crop
    #Use of mask flattens array so crop_without_ship* arrays are 1D
    mask = np.ones((crop_size[0], crop_size[1]), dtype=bool)
    mask[r:r+ship_crop_size[0], c:c+crop_size[1]] = False

    crop_without_ship_slc      = cropped_slc_burst[mask]
    crop_without_ship_raw_real = cropped_raw_real_burst[mask]
    crop_without_ship_raw_imag = cropped_raw_imag_burst[mask]
    crop_without_ship_slc_intensity = intensity(crop_without_ship_slc)

    #Get pixels associated with ship
    # Set threshold to mean + X std dev of signal in area without ship
    ship_detection_threshold = ( crop_without_ship_slc_intensity.mean()
                             + 3*crop_without_ship_slc_intensity.std()
                                )
    #Create mask for identifying ship pixels
    ship_crop_slc_above_threshold = ship_crop_slc_intensity > ship_detection_threshold
    # Process mask map to remove isolated pixels using erosion/dilation
    processed_ship_crop_mask = dilation(erosion(ship_crop_slc_above_threshold))
    ship_pixels = ship_crop_slc_intensity[processed_ship_crop_mask]



    #burst & crop statistics            
    raw_burst_I_stats = get_statistics(raw_real_burst)
    raw_burst_Q_stats = get_statistics(raw_imag_burst)
    slc_burst_I_stats = get_statistics(np.real(slc_burst))
    slc_burst_Q_stats = get_statistics(np.imag(slc_burst))

    slc_burst_intensity_stats = get_statistics(slc_intensity)
    slc_burst_intensity_filtered_stats = get_statistics(slc_intensity_filtered)

    raw_crop_I_stats  = get_statistics(cropped_raw_real_burst)
    raw_crop_Q_stats  = get_statistics(cropped_raw_imag_burst)
    slc_crop_I_stats  = get_statistics(np.real(cropped_slc_burst))
    slc_crop_Q_stats  = get_statistics(np.imag(cropped_slc_burst))
    crop_slc_intensity_stats = get_statistics(intensity(cropped_slc_burst))

    crop_without_ship_slc_intensity_stats = get_statistics(
                                              crop_without_ship_slc_intensity
                                                )
    raw_ship_crop_I_stats = get_statistics(ship_crop_raw_real)
    raw_ship_crop_Q_stats = get_statistics(ship_crop_raw_imag)
    slc_ship_crop_I_stats = get_statistics(ship_crop_slc_I)
    slc_ship_crop_Q_stats = get_statistics(ship_crop_slc_Q)
    slc_ship_crop_intensity_stats = get_statistics(ship_crop_slc_intensity)

    #Get ship statistics, size and brightness
    ship_stats  = get_statistics(ship_pixels)
    ship_size   = len(ship_pixels)
    ship_rcs    = ship_size * ship_pixels.mean()
    #Plot graphs
    # Plot SLC and equivalent RAW
    """
    img_name = f'SLC_{crc_burst}'
    f=plot_2d_array(np.square(np.abs(slc_burst)), img_name)
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    # Plot RAW burst
    img_name = f'SLC_matched_Raw_Burst_Real_{crc_burst}'
    f=plot_2d_array(raw_real_burst,img_name,'linear')
    f.savefig(f'{img_path}/{img_name}.jpg', format='jpg', dpi=600)
    """

    #Plot multiple graphs on single plot
    fig, axs  = plt.subplots(3, 6, figsize=(16,9),constrained_layout=True)
    num_raw_bins  = 20
    num_bins  = 100
    threshold = 10
    # For example assign the function's plot to subplot 2
    plot_fig_name = f'{crc_burst}_ship_{ship_location}_RAW_SLC_Histograms'
    fig.suptitle(plot_fig_name)
    img_name  = f"Raw_Burst_Real"
    plot_hist_with_fit_on_ax(axs[0, 0], raw_real_burst, num_raw_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"Raw_Imag_Real"
    plot_hist_with_fit_on_ax(axs[0, 1], raw_imag_burst,num_raw_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"SLC_Burst_Real"
    plot_hist_with_fit_on_ax(axs[0, 2], np.real(slc_burst),num_bins,
                             threshold, 'gaussian',img_name)
    img_name = f"SLC_Burst_Imag"
    plot_hist_with_fit_on_ax(axs[0, 3], np.imag(slc_burst),num_bins,
                             threshold, 'gaussian',img_name)

    img_name = f"SLC_Burst_Amplitude"
    plot_hist_with_fit_on_ax(axs[0, 4], np.abs(slc_burst),num_bins,
                             threshold, 'gamma',img_name)
    img_name =f"SLC_Burst_Intensity"
    plot_hist_with_fit_on_ax(axs[0, 5], np.square(np.abs(slc_burst)),num_bins,
                             threshold, 'weibull',img_name)
    #Plot Crop data with ship at centre ; crop size = crop_size

    img_name  = f"Raw_Real_Crop"
    plot_hist_with_fit_on_ax(axs[1, 0], cropped_raw_real_burst, num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"Raw_Imag_Crop"
    plot_hist_with_fit_on_ax(axs[1, 1], cropped_raw_imag_burst,num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"SLC_Real_Crop"
    plot_hist_with_fit_on_ax(axs[1, 2], np.real(cropped_slc_burst),num_bins,
                             threshold, 'gaussian',img_name)
    img_name = f"SLC_Imag_Crop"
    plot_hist_with_fit_on_ax(axs[1, 3], np.imag(cropped_slc_burst),num_bins,
                             threshold, 'gaussian',img_name)

    img_name = f"SLC_Crop_Amplitude"
    plot_hist_with_fit_on_ax(axs[1, 4], np.abs(cropped_slc_burst),num_bins,
                             threshold, 'gamma',img_name)
    img_name =f"SLC_Crop_Intensity"
    plot_hist_with_fit_on_ax(axs[1, 5], np.square(np.abs(cropped_slc_burst)),num_bins,
                             threshold, 'weibull',img_name)

    #Plot data for ship only with crop size = ship_crop_size
    #In this data outliers are not excluded (threshold == 0)

    threshold = 0
    img_name  = f"Raw_Real_Ship_Crop"
    plot_hist_with_fit_on_ax(axs[2, 0], ship_crop_raw_real, num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"Raw_Imag_Ship_Crop"
    plot_hist_with_fit_on_ax(axs[2, 1], ship_crop_raw_imag,num_bins,
                             threshold,'gaussian',img_name
                             )
    img_name = f"SLC_Real_Ship_Crop"
    plot_hist_with_fit_on_ax(axs[2, 2], np.real(ship_crop_slc),num_bins,
                             threshold, 'gaussian',img_name)
    img_name = f"SLC_Imag_Ship_Crop"
    plot_hist_with_fit_on_ax(axs[2, 3], np.imag(ship_crop_slc),num_bins,
                             threshold, 'gaussian',img_name)

    img_name = f"SLC_Ship_Crop_Amp"
    plot_hist_with_fit_on_ax(axs[2, 4], np.abs(ship_crop_slc),num_bins,
                             threshold, '',img_name)
    img_name =f"SLC_Ship_Crop_Intensity"
    plot_hist_with_fit_on_ax(axs[2, 5], intensity(ship_crop_slc),1000,
                             threshold, '',img_name)
    # #Plot multiple graphs on single plot
    fig1, axs1  = plt.subplots(2, 3, figsize=(16,9),constrained_layout=True)

    font_size =8
    plot_name = f'{crc_burst}_ship_{ship_location}_RAW_SLC'
    fig1.suptitle(plot_name)
    img_name  = f"SLC"
    plot_2d_array_on_ax(axs1[0, 0], slc_intensity,img_name,'log_normal',font_size)
    # Highlight ship:  Place circle around ship
    text_offset=[200,200]
    add_text_to_plot(axs1[0,0], f'Ship @ [{ship_location[0]},'
                     f'{ship_location[1]}]',
                     ship_location,text_offset
                     )
    circle = patches.Circle((ship_location[1], ship_location[0]), radius=50, edgecolor='red', facecolor='none', linewidth=1)
    axs1[0,0].add_patch(circle)
    img_name = f"SLC_crop_for_ship_{ship_location}"
    plot_2d_array_on_ax(axs1[0, 1], intensity(cropped_slc_burst),img_name,'log_normal',font_size)
    img_name = f"RAW_crop_{crc_burst}"
    plot_2d_array_on_ax(axs1[1, 0], cropped_raw_real_burst,img_name,'linear',font_size)
    img_name = f"Range_compressed_Raw_crop"
    plot_2d_array_on_ax(axs1[1, 1],
                        intensity(range_compress_burst(cropped_raw_real_burst)),
                        img_name,'log_normal',font_size
                        )

    img_name = f"Ship_crop"
    plot_2d_array_on_ax(axs1[0, 2],
                       ship_crop_slc_intensity,
                        img_name,'log_normal',font_size
                        )
    img_name =f'Ship_only_pixels_after_masking'
    plot_2d_array_on_ax(axs1[1, 2],
                       ship_crop_slc_intensity*processed_ship_crop_mask +1,
                        img_name,'log_normal',font_size
                        )



    big_crop_size = [1000,1000]
    r,c, pos_in_crop = place_roi(raw_real_burst, big_crop_size, ship_location)

    cropped_raw_real_burst2 = raw_real_burst[r:r + big_crop_size[0], c:c + big_crop_size[1]]
    img_name = f"Large_Raw_crop"
    #plot_2d_array_on_ax(axs1[2, 0], cropped_raw_real_burst2,img_name,'linear')

    #plt.show()   

    #plt.pause(1)
    if do_visual_check:
        while True:
            try:
                response = int(input(f'What is strength of interferogram on'
                                     f' scale of 0 to 2; where 0 indicates'
                                     f' non-visibility, 1 weakly present and'
                                     f' 2 strongly present: '
                ))
                if response in (0, 1, 2):
                    signal_strength = response
                    break
                else:
                    print('Invalid response. Please enter 0, 1, or 2.')
            except ValueError:
                print('Input must be an integer. Please try again.')
    else:
        response=-1

    fig.savefig(f'{save_path}{plot_fig_name}.jpg', format='jpg',
             bbox_inches='tight', dpi=600
             )

    fig1.savefig(f'{save_path}{plot_name}.jpg', format='jpg',
            bbox_inches='tight', dpi=600
            )

    #Assign rows to statistics dataframe

    # Close all figures 
    fig.clf()
    fig1.clf()
    plt.close(fig)
    plt.close(fig1)

    burst_crop_data.append({

        "Burst Name"       : crc_burst,
       # "Ship Length"      : ship_length,
        "Ship Row"         : ship_location[0],
        "Ship Col"         : ship_location[1],
        "Signal Strength"  : response,
        "Burst Raw I Mean" : raw_burst_I_stats[0],
        "Burst Raw Q Mean" : raw_burst_Q_stats[0],
        "Burst Raw I StDev": raw_burst_I_stats[1],
        "Burst Raw Q StDev": raw_burst_Q_stats[1],
        "Burst SLC I Mean" : slc_burst_I_stats[0],
        "Burst SLC Q Mean" : slc_burst_Q_stats[0],
        "Burst SLC I StDev": slc_burst_I_stats[1],
        "Burst SLC Q StDev": slc_burst_Q_stats[1],
        "Burst SLC Intensity Mean"  : slc_burst_intensity_stats[0],
        "Burst SLC Intensity StDev" : slc_burst_intensity_stats[1],
        "Crop Raw I Mean"  : raw_crop_I_stats[0],
        "Crop Raw Q Mean"  : raw_crop_Q_stats[0],
        "Crop Raw I StDev" : raw_crop_I_stats[1],
        "Crop Raw Q StDev" : raw_crop_Q_stats[1],
        "Crop SLC I Mean"  : slc_crop_I_stats[0],
        "Crop SLC Q Mean"  : slc_crop_Q_stats[0],
        "Crop SLC I StDev" : slc_crop_I_stats[1],
        "Crop SLC Q StDev" : slc_crop_Q_stats[1],
        "Crop SLC Intensity Mean"  : crop_slc_intensity_stats[0],
        "Crop SLC Intensity StDev" : crop_slc_intensity_stats[1],


        "Ship Crop Raw I Mean"  : raw_ship_crop_I_stats[0],
        "Ship Crop Raw Q Mean"  : raw_ship_crop_Q_stats[0],
        "Ship Crop Raw I StDev" : raw_ship_crop_I_stats[1],
        "Ship Crop Raw Q StDev" : raw_ship_crop_Q_stats[1],
        "Ship Crop SLC I Mean"  : slc_ship_crop_I_stats[0],
        "Ship Crop SLC Q Mean"  : slc_ship_crop_Q_stats[0],
        "Ship Crop SLC I StDev" : slc_ship_crop_I_stats[1],
        "Ship Crop SLC Q StDev" : slc_ship_crop_Q_stats[1],
        "Ship Crop SLC Intensity Mean"  : slc_ship_crop_intensity_stats[0],
        "Ship Crop SLC Intensity StDev" : slc_ship_crop_intensity_stats[1],
        "CWS Mean"      : crop_without_ship_slc_intensity_stats[0],
        "CWS StdDev"    : crop_without_ship_slc_intensity_stats[1],
        "Ship Mean RCS" : ship_stats[0],
        "Ship StDev"    : ship_stats[1],
        "Ship RCS"      : ship_rcs,
        "Ship Area"     : ship_size

    })


    return pd.DataFrame(burst_crop_data)





def return_annotation_file(safe_dir, swath_index, pol):        
    # Get slc image and annotation file names and directories
    # Use getSLCFilesFromSafeDir function even though don't yet know which swath
    # is of interest. Using '2' as a dummy placeholder.
    slc_file_info   = getSLCFilesFromSafeDir(safe_dir,2,pol)
    annotation_path = slc_file_info[2]
    slc_path        = annotation_path.replace('annotation','measurement')
    slc_filenames   = slc_file_info[3]
    # Get SLC src and annotation file names
    slc_src_file    = slc_path + slc_filenames[swath_index]
    annotation_file = annotation_path+change_extension(slc_filenames[swath_index],
                                                        "xml"
                                                        )
    return annotation_file

    
    
# Function to locate swath and burst that a gps location is in
# and return the raw and slc burst. The slc burst is 'de-bursted'
# ensuring no overlap of contiguous bursts and the raw burst has 
# matched timings. The slc burst is then re-sampled (if neccessary) 
# to the same range sampling frequency as the raw burst.
def return_raw_slc_matched_bursts_for_gps_location(raw_dir,
                                                   slc_safe_dir,
                                                   gps_long_lat,
                                                   pol
                                                   ):
    

    fnames_swath = get_raw_slc_annotation_filenames(raw_dir, 
                                                    slc_safe_dir,
                                                    gps_long_lat, 
                                                    pol
                                                    )
    raw_file        = fnames_swath[0]
    slc_src_file    = fnames_swath[1]
    annotation_file = fnames_swath[2]
    swath_index     = fnames_swath[3]
    burst_index     = fnames_swath[4]
    
    
    radar_data      = get_slc_matched_raw_burst_from_file(raw_file, 
                                                          annotation_file, 
                                                          burst_index +1, 
                                                          swath_index +1, 
                                                          pol
                                                          ) 
    raw_burst      = radar_data[0]
    slc_burst_data = get_slc_burst_from_files(slc_src_file,
                                              annotation_file,
                                              burst_index +1
                                              )
    non_resampled_slc_burst = slc_burst_data[0]

    # Is this slc burst re-sampled to match raw sampling rate?
    sampling_freq_mul = freq_multipliers[swath_index]
    scaling_factor    = int(sampling_freq_mul*non_resampled_slc_burst.shape[1])
    slc_burst         = resample(non_resampled_slc_burst, scaling_factor)

    return raw_burst, slc_burst, swath_index, burst_index    



def get_statistics(roi):    
    mean = roi.mean()
    std  = roi.std()
    SNR  = mean/std
    max_roi  = roi.max()
    # Get mean of 10 brightest pixels
    avg10_max = np.sort(roi)[-10:].mean()
    return mean, std, SNR, max_roi, avg10_max


def intensity(arr):
    return np.square(np.abs(arr))

