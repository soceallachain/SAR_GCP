#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:27:19 2025

@author: joe
"""
from SAR_code_module import *



def get_valid_corners_of_bursts(annotation_file):
    """

    Parameters
    ----------
    annotation_file : TYPE
        DESCRIPTION.
        Reads annotation file to determine the valid lines and range samples
        in SLC data.
        
    Returns
    -------
    valid_corners_of_bursts : TYPE
        DESCRIPTION.

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
    
    # Create 'valid_burst_corners' as list of tuples per row
    df['valid_burst_corners'] = df.apply(lambda row: [(row['first_valid_line'], row['first_valid_range_pixels']),
                                                      (row['last_valid_line'], row['first_valid_range_pixels']),
                                                      (row['last_valid_line'], row['last_valid_range_pixels']),
                                                      (row['first_valid_line'], row['last_valid_range_pixels'])], axis=1)
    
    #valid_corners_of_bursts = np.array(df['valid_burst_corners'].tolist())
    valid_corners_of_bursts = df['valid_burst_corners'].tolist()
    return valid_corners_of_bursts


# Test functions
raw_path = "/home/joe/Misc/tempTest/Data/RAW_Data/f9fd0c2000d1a632p/RAW/S1B_IW_RAW__0SDV_20200421T054807_20200421T054840_021238_0284D5_1452.SAFE/"
raw_file_name='s1b-iw-raw-s-vh-20200421t054807-20200421t054840-021238-0284d5.dat'
raw_file=f'{raw_path}{raw_file_name}'

slc_id = 'S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96'
slc_id_crc = slc_id[-11:]

burst_number = 3
annotation_file='/home/joe/Misc/tempTest/Data/SLC_Data/S1B_IW_SLC_1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96/SLC/S1B_IW_SLC__1SDV_20200421T054810_20200421T054837_021238_0284D5_2C96.SAFE/annotation/s1b-iw2-slc-vh-20200421t054811-20200421t054837-021238-0284d5-002.xml'
print(get_valid_corners_of_bursts(annotation_file))


burst_info = getSwathBurstGridGPSBoundaries(annotation_file)
print(burst_info[1])

pre_post_info=get_pre_post_mid_times(annotation_file)

raw_data=get_raw_burst(slc_id, burst_number)
fig_handles=plot_raw_burst(raw_data, slc_id, burst_number)


raw_burst_info=get_raw_burst_info(raw_file)



slc_iw_data=[]
slc_iw_data.append(pre_post_info)
iw_mode = 1
burst_of_interest=burst_number
raw_iw_data=raw_burst_info
matching_burst    = slc_iw_data[0][slc_iw_data[0]['burst_number']==str(burst_of_interest)]
index_of_burst    = find_raw_start_burst(raw_iw_data[iw_mode], matching_burst)

l0file           = sentinel1decoder.Level0File(raw_file)
raw_burst_l0file_info   = l0file.get_burst_metadata(index_of_burst)

valid_corners=get_valid_corners_of_bursts(annotation_file)

#Check non-debursted burt boundatries
non_debursted_corners = getBurstBoundaries(annotation_file)

################################################### test df's

def getSwathValidMergedBurstsGPSGrid(annotation_file):
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
    
    
    
    #Add code to get valid lines and pixels
    valid_burst_corners             = get_valid_corners_of_bursts(annotation_file)
    number_valid_burst_lines        = get_number_valid_lines_in_slc_burst(valid_burst_corners)
    number_valid_burst_range_pixels = get_number_valid_range_pixels_in_slc_burst(valid_burst_corners)
    invalid_burst_start_lines       = get_number_invalid_start_lines_in_slc_burst(valid_burst_corners)
    invalid_burst_start_pixels      = get_number_invalid_start_range_pixels_in_slc_burst(valid_burst_corners)
    
    
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
    for i in range(0,len(burst_start_times)-1):
        # Add burst interval time to start time
        t = valid_burst_start_times[i] + timedelta(seconds=(number_valid_burst_lines[i]*azTimeInterval))
        # Subtract end time of burst i from start of next burst i+1
        overlap_time  = t - valid_burst_start_times[i+1]
        #Round down (why?)
        overlap_lines = int(overlap_time.total_seconds() / azTimeInterval)
        burst_end_time.append(t)
        number_overlap_lines.append(overlap_lines)
    
    
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
        if i==0:
            #+ round(np.mean(overlap_offset_lines))  
            # NOT SURE ABOUT THIS, but may be source of offset 
            merged_start_line = valid_burst_start_line + round(overlap_offset_lines[i])
            merged_end_line   = (valid_burst_end_line
                                - round(overlap_offset_lines[i]))
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
    
        non_overlapped_end_line = valid_burst_end_line - number_overlap_lines[i]
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
                                       burst_end_long_lat[i,:]]
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




b_orig=getSwathBurstGridGPSBoundaries_OriginalAndBest(annotation_file)


b_mod = getSwathBurstGridGPSBoundaries(annotation_file)




slc_burst = get_slc_burst_with_ships(slc_id, burst)
slc_ship_burst = slc_burst[1]
plot_2d_array(slc_ship_burst**2, f'{slc_id_crc} Burst#{burst}: Size = {slc_ship_burst.shape}')
print(f' SLC Burst #{burst} has size {slc_ship_burst.shape} ')


raw_burst_data = get_raw_burst(slc_id, burst)
plot_2d_array(np.real(raw_burst_data), 'Raw Data Burst#{burst} of {slc_id_crc}', 'linear')