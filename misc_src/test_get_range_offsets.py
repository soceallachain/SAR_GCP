#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:16:08 2026

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
from SAR_code_module import *
from SAR_focusing_module import *
from scipy.ndimage import zoom
#filepath = "../data/sao_paulo/"
#filename = "s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat"
raw_filepath = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_RAW__0SDH_20180801T001745_20180801T001811_012062_01635C_9872.SAFE/"
#filepath = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/S1A_S6_RAW_0SDV_20210226T214354_20210226T214424_036766_045287_FD68/RAW/S1A_S6_RAW__0SDV_20210226T214354_20210226T214424_036766_045287_FD68.SAFE/"

#filename = "s1a-s6-raw-s-vv-20210226t214354-20210226t214424-036766-045287.dat"
raw_filename  = "s1b-s3-raw-s-hv-20180801t001745-20180801t001811-012062-01635c.dat"
raw_file = raw_filepath + raw_filename

annot_filepath = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A.SAFE/annotation/"
annot_filename = "s1b-s3-slc-hv-20180801t001746-20180801t001810-012062-01635c-002.xml"
slc_annotation_file= annot_filepath + annot_filename

slc_filepath = "/home/joe/Work/Projects/RSAR/OtherSARData/Houston/S1B_S3_SLC__1SDH_20180801T001746_20180801T001810_012062_01635C_6E9A.SAFE/measurement/"
slc_filename = f's1b-s3-slc-hv-20180801t001746-20180801t001810-012062-01635c-002.tiff'
slc_file     = slc_filepath + slc_filename


slc_burst_info  = get_SM_GridPoints(slc_annotation_file)
raw_slc_offsets = get_SM_raw_slc_offsets(raw_file, slc_annotation_file)
raw_burst_info  = get_SM_data(raw_file)

raw_data = raw_burst_info[2]
swath_number = int(raw_filename.split('-')[1][1]) - 1

filtered_df= raw_data.packet_metadata[
        (raw_data.packet_metadata['Swath Number'] == swath_number) &
        (raw_data.packet_metadata['Signal Type'] == 0)
    ]

# Find packets that match with SLC grid lines
# Use az_offset betwen slc and raw and add to Packet Number 
# Generate dataframe with same number of grid lines as slc grid
# Get list of lines i.e. where pixel == 0
result=filtered_df.reset_index()[['Azimuth Block Number','Packet Number', 'Coarse Time',
                          'Fine Time', 'ECC Number','Signal Type',
                          'Swath Number', 'Number of Quads']]

result['Burst Start Time'] = pd.to_datetime(result.apply(lambda row:
                                 gps_to_utc(row['Coarse Time'],
                                         row['Fine Time']), axis=1))


slc_grid_line_info = slc_burst_info.loc[slc_burst_info['pixel'] == '0', 
                                        ['line', 'azimuth_time', 
                                         'slant_range_time']
                                        ]
az_slc_raw_offset = raw_slc_offsets[0]


# Extract these packets from the raw data and merge with slc_grid_line_info
index_of_first_packet = result['Packet Number'].iloc[0]
raw_packet_numbers    = (slc_grid_line_info['line'].to_numpy().astype(int) 
                       + az_slc_raw_offset + index_of_first_packet)
matched_slc_raw_lines = result.query("`Packet Number` in @raw_packet_numbers")
#matched_slc_raw_lines['azimuth time'] = slc_grid_line_info['azimuth_time'].values
#matched_slc_raw_lines['line'] = slc_grid_line_info['line'].values
matched_slc_raw_lines[['azimuth time', 'line', 'slant range time']] = slc_grid_line_info[
                                                     ['azimuth_time',
                                                      'line',
                                                      'slant_range_time']
                                                    ].values
# Get difference between azimuth time(slc) and Burst Start Time(raw)
# Add .dt before total_seconds()
matched_slc_raw_lines['start_diff'] = (
    matched_slc_raw_lines['Burst Start Time'] - matched_slc_raw_lines['azimuth time']
).dt.total_seconds() 

RGDEC                = raw_burst_info[0].["Range Decimation"].unique()[0]
#RGDEC                = selection["Range Decimation"].unique()[0]
range_sample_freq    = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
 
# Get parameters from annotation file
tree = etree.parse(slc_annotation_file)
root = tree.getroot()
    
# SLC parameters
#n_slc_az_lines      = int(root.find('.//numberOfLines').text)
#azTimeInterval      = float(root.find('.//azimuthTimeInterval').text
zeroDopMinusAcqTime      = float(root.find('.//zeroDopMinusAcqTime').text
t_SWL = raw_burst_info[0]["SWL"].unique()[0]
t_SWST = raw_burst_info[0]["SWST"].unique()[0]
ts = result['Burst Start Time'].iloc[0]
sensingStartTime = ts.second + (ts.microsecond / 1e6) + (ts.nanosecond / 1e9)


firstLineSLCStartTime=sensingStartTime + (zeroDopMinusAcqTime//pri)*pri
slc_pixel0_time = slc_grid_line_info['azimuth_time'].loc[0]
slc_pixel0_time_seconds = (slc_pixel0_time.second 
                        + (slc_pixel0_time.microsecond / 1e6) 
                        + (slc_pixel0_time.nanosecond / 1e9)
                          )
                  
slc_mid_pixel_time_seconds = productFirstLineUtcTime





 matched_slc_raw_lines['start_delta'] = (
     (  matched_slc_raw_lines['azimuth time'] 
      - matched_slc_raw_lines['Burst Start Time'] 
     ).dt.total_seconds()
     +  matched_slc_raw_lines['Number of Quads'].astype(int) /range_sample_freq
                                          )