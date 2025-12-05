#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:57:34 2025

@author: joe
"""

from SAR_code_module import *

"""
def get_single_raw_burst_from_files(raw_file, slc_annotation_file, burst_to_get, swath_of_interest=2, polarization='vh'):

##
# Function to return a raw burst based on the SLC burst number
# Requires access to raw data file and SLC annotation file
# Assumes these are in standard RAW and SLC directories
#
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
        print('ERROR: IW mode of SLC annotation file does not match expected swath or there is a polarization mis-match !')
        print('Exiting ....')
        sys.exit(0)
    
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
    
    bursts_not_to_save = []
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
    
    return radar_data



def get_slc_range_blackfill_indices(slc_burst, threshold_level=0.5):
    """"""
    Parameters
    ----------
    slc_burst : TYPE
        Script to find where in the range direction blackfill starts and ends
        Sums along azimuth direction to find average starting and ending index
        Then sets start and end as index where threshold exceeded
    Returns
    -------
    Indices of start and end

    """"""
    range_sum      = np.sum(slc_burst, axis=0)
    range_sum_mean = range_sum.mean()
    threshold      = range_sum_mean*threshold_level
    start_index    = np.nonzero(range_sum>threshold)[0][0]
    end_index      = np.nonzero(range_sum>threshold)[0][-1]
    imbalance      = (slc_burst.shape[1] - end_index) - start_index
    return start_index, end_index, imbalance



"""
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

# Test function



raw_file            = '/home/joe/Misc/tempTest/Data/RAW_Data/f9fd0c2000d1a632p/RAW/S1B_IW_RAW__0SDV_20200421T054807_20200421T054840_021238_0284D5_1452.SAFE/s1b-iw-raw-s-vh-20200421t054807-20200421t054840-021238-0284d5.dat'
slc_annotation_file = '/home/joe/Misc/tempTest/Data/SLC_Data/S1B_IW_SLC_1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96/SLC/S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96.SAFE/annotation/s1b-iw2-slc-vh-20200421t054811-20200421t054837-021238-0284d5-002.xml'
slc_src_file        = '/home/joe/Misc/tempTest/Data/SLC_Data/S1B_IW_SLC_1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96/SLC/S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96.SAFE/measurement/s1b-iw2-slc-vh-20200421t054811-20200421t054837-021238-0284d5-002.tiff'


#raw_file            = '/home/joe/Misc/tempTest/Data/RAW_Data/aad0cc0c9ff85960p/RAW/S1A_IW_RAW__0SDV_20200720T054903_20200720T054935_033534_03E2C6_C4C6.SAFE/s1a-iw-raw-s-vh-20200720t054903-20200720t054935-033534-03e2c6.dat'
#slc_annotation_file = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200720T054905_20200720T054933_033534_03E2C6_2D45/SLC/S1A_IW_SLC__1SDV_20200720T054905_20200720T054933_033534_03E2C6_2D45.SAFE/annotation/s1a-iw2-slc-vh-20200720t054907-20200720t054932-033534-03e2c6-002.xml'
#slc_src_file        = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200720T054905_20200720T054933_033534_03E2C6_2D45/SLC/S1A_IW_SLC__1SDV_20200720T054905_20200720T054933_033534_03E2C6_2D45.SAFE/measurement/s1a-iw2-slc-vh-20200720t054907-20200720t054932-033534-03e2c6-002.tiff'


#raw_file            = '/home/joe/Misc/tempTest/Data/RAW_Data/fb8dc6ae0258a3afp/RAW/S1A_IW_RAW__0SDV_20200906T054840_20200906T054913_034234_03FA57_DFC7.SAFE/s1a-iw-raw-s-vh-20200906t054840-20200906t054913-034234-03fa57.dat'
#slc_annotation_file = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200906T054843_20200906T054910_034234_03FA57_A7BB/SLC/S1A_IW_SLC__1SDV_20200906T054843_20200906T054910_034234_03FA57_A7BB.SAFE/annotation/s1a-iw2-slc-vh-20200906t054844-20200906t054910-034234-03fa57-002.xml'
#slc_src_file        = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200906T054843_20200906T054910_034234_03FA57_A7BB/SLC/S1A_IW_SLC__1SDV_20200906T054843_20200906T054910_034234_03FA57_A7BB.SAFE/measurement/s1a-iw2-slc-vh-20200906t054844-20200906t054910-034234-03fa57-002.tiff'

#raw_file            = '/home/joe/Misc/tempTest/Data/RAW_Data/4bf6c82ef5130dabt/RAW/S1A_IW_RAW__0SDV_20200116T051408_20200116T051441_030821_038933_4F07.SAFE/s1a-iw-raw-s-vh-20200116t051408-20200116t051441-030821-038933.dat'
#slc_annotation_file = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200116T051410_20200116T051438_030821_038933_CD1B/SLC/S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B.SAFE/annotation/s1a-iw2-slc-vh-20200116t051411-20200116t051437-030821-038933-002.xml'
#slc_src_file        = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200116T051410_20200116T051438_030821_038933_CD1B/SLC/S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B.SAFE/measurement/s1a-iw2-slc-vh-20200116t051411-20200116t051437-030821-038933-002.tiff'   

burst = 3
swath = 2

slc_id= slc_annotation_file.split('/SLC/')[1].split('.SAFE')[0]
slc_id_crc = slc_id[-11:]


####################################################
# Test get_raw_burst and get_single_raw_burst_from_files
# plot single raw burst of non cropped raw data
raw_data = get_single_raw_burst_from_files(raw_file, slc_annotation_file, burst, swath)
plot_2d_array(np.real(raw_data[0]), f'{slc_id_crc} RAW Burst#{burst}Not Cropped to SLC: Size= {raw_data[0].shape}', 'linear')
print(f' RAW Burst #{burst} has size of {raw_data[0].shape}')


####################################################
# Test get_raw_burst and get_single_raw_burst_from_files
# plot singe raw burst of burst cropped to size time ranges of SLC burst
raw_burst = get_raw_burst(slc_id, burst)
raw_cropped=plot_2d_array(np.real(raw_burst), f'{slc_id_crc} Burst#{burst} RAW Cropped-to_SLC: Size ={raw_burst.shape} ', 'linear')
print(f' RAW Cropped-to-SLC  Burst #{burst} has size of {raw_burst.shape}')


#plot single de-bursted slc burst from SLC_ID
#slc_id = 'S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B'
slc_burst = get_slc_burst_with_ships(slc_id, burst)
slc_ship_burst = slc_burst[1]
plot_2d_array(slc_ship_burst**2, f'{slc_id_crc} Burst#{burst} Revised code: Size = {slc_ship_burst.shape}')
print(f' SLC Burst #{burst} has size {slc_ship_burst.shape} ')

#plot single de-bursted slc burst from files
burst_data = get_slc_burst_from_files(slc_src_file, slc_annotation_file, burst)
plot_2d_array(np.abs(burst_data[0])**2, f'{slc_id_crc} Burst#{burst} -Revised getSwath code')
print(f' SLC Burst #{burst} has size of {burst_data[0].shape}')

# Get single non-debursted burst of SLC data
non_deburst_burst_image = get_single_non_debursted_slc_burst(slc_src_file, slc_annotation_file, burst)
non_deburst_burst_intensity = np.abs(non_deburst_burst_image)**2
plot_2d_array(non_deburst_burst_intensity, f'Non Debursted {slc_id_crc} Burst#{burst}: Burst Size = {non_deburst_burst_image.shape}')
print(f' Non Debursted SLC Burst #{burst} has size of {non_deburst_burst_image.shape}')

print(f' RAW Burst Not Cropped to SLC #{burst} has size of {raw_data.shape}')
print(f' RAW Burst Cropped-to-SLC #{burst} has size of {raw_burst.shape}')
print(f' SLC Debursted Burst with ships  #{burst} has size {slc_ship_burst.shape} ')
print(f' SLC Debursted Burst #{burst} has size of {burst_data[0].shape}')
print(f' SLC Non-Deburst Burst #{burst} has size of {non_deburst_burst_image.shape}')

number_raw_samples = raw_burst.shape[1]
resampled_slc = resample(slc_ship_burst, number_raw_samples)
resampled_slc_fig=plot_2d_array(resampled_slc**2, f'{slc_id_crc} Resampled SLC Burst#{burst}: Size = {resampled_slc.shape}')
print(f' Re-sampled SLC Burst #{burst} has size {resampled_slc.shape} ')


slc_burst_range_sum =np.sum(non_deburst_burst_intensity,axis =0)
plot_1d_array(slc_burst_range_sum,f'{slc_id_crc} Burst#{burst} Blackfill')

indices=get_slc_range_blackfill_indices(non_deburst_burst_intensity)