## getSLCBurstInfo.py
##
## Code to extract burst information for SLC file


from lxml import etree
# system modules
import os
import io
import re
import sys
import random
import shutil
# data manipulation
import shapely
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import skimage.measure as skm
import matplotlib.gridspec as gridspec
from matplotlib import colors
from datetime import datetime, timedelta
from shapely import LineString, Point, Polygon


pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.precision', 12)

def change_extension(file_name, new_extension):
    base_name, _ = os.path.splitext(file_name)
    new_file_name = base_name + "." + new_extension
    return new_file_name

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

#
# Function to get GPS coordinates of a swath pointed to by 'annotation file'
#
def getSwathGPSBoundaries(annotation_file):
    
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

        azimuth_burst_end_time = datetime.strptime(azimuth_time,'%Y-%m-%dT%H:%M:%S.%f') + timedelta(seconds = (nBurstLines * azTimeInterval))
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
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),(nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    return gps_corners

"""

    
""" 


#
# Modification of getSwathBurstGPSBoundaries to now also return the 
# sub-grids within each burst in both [row,col] and gps(long,lat) formats
#
def getSwathBurstGridGPSBoundaries(annotation_file):
    
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

        azimuth_burst_end_time = datetime.strptime(azimuth_time,'%Y-%m-%dT%H:%M:%S.%f') + timedelta(seconds = (nBurstLines * azTimeInterval))
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
    
    pixel_value = '0'
    burst_start_times = df[df['pixel'] == pixel_value]['azimuth_time']
    burst_end_time = []
    number_overlap_lines = []
    for i in range(0,len(burst_start_times)-1):
        # Convert string to datetime object
        t = datetime.strptime(burst_start_times.iloc[i], '%Y-%m-%dT%H:%M:%S.%f')
        # Add burst interval time to start time
        t = t + timedelta(seconds=(nBurstLines*azTimeInterval))
        # Subtract end time of burst i from start of next burst i+1
        overlap_time = t- datetime.strptime(burst_start_times.iloc[i+1], '%Y-%m-%dT%H:%M:%S.%f')
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
    
    burst_ends           = gridPoints[(gridPoints['pixel'] == str(nRangeSamples-1))]
    burst_starts         = gridPoints[(gridPoints['pixel'] == str(0))]
    burst_end_long_lat   = burst_ends[['longitude','latitude']].to_numpy().astype(float)
    burst_start_long_lat = burst_starts[['longitude','latitude']].to_numpy().astype(float)
    
    burst_grid_corners  = []
    burst_grid_gps      = []

    for i in range(0,nBursts):
        burst_start_line  = gridPoints[gridPoints['burst_number'] == str(i+1)]['line'].to_numpy().astype(int)[0]
        burst_end_line    = gridPoints[gridPoints['burst_number'] == str(i+2)]['line'].to_numpy().astype(int)[0]
        if i==0:
            merged_start_line = burst_start_line
            merged_end_line   = burst_end_line   - round(overlap_offset_lines[i])
        else:
            merged_start_line = burst_start_line + round(overlap_offset_lines[i-1])
            merged_end_line   = burst_end_line   - round(overlap_offset_lines[i])
            
        gridPoints.loc[gridPoints['burst_number'] == str(i+1), 'merged_burst_start_line'] = merged_start_line
        gridPoints.loc[gridPoints['burst_number'] == str(i+1), 'merged_burst_end_line']   = merged_end_line
        burst_corners.append([(merged_start_line,0),(merged_end_line,0),(merged_end_line,nRangeSamples-1), (merged_start_line,nRangeSamples-1)])
        burst_gps_corners.append([burst_start_long_lat[i,:], burst_start_long_lat[i+1,:], burst_end_long_lat[i+1,:], burst_end_long_lat[i,:]])
        
        ##### Add code for sub-grids
        boi1           = gridPoints[(gridPoints['burst_number'] == str(i+1))]
        boi2           = gridPoints[(gridPoints['burst_number'] == str(i+2))]
        #long_lat_start = np.array(boi1[['longitude','latitude']].to_numpy().astype(float))
        #long_lat_end   = np.array(boi2[['longitude','latitude']].to_numpy().astype(float))
        # Check why burst coords work with getGrid but burst_grid coords don't
        long_lat_start = boi1[['longitude','latitude']].to_numpy().astype(float)
        long_lat_end   = boi2[['longitude','latitude']].to_numpy().astype(float)
        range_col      = boi1[['pixel']].to_numpy().astype(int)    
        grid_corners   = []
        grid_gps       = []
        for j in range(0,len(boi1)-1):
            grid_corners.append([(merged_start_line,range_col[j]), (merged_end_line, range_col[j]), (merged_end_line, range_col[j+1]), (merged_start_line, range_col[j+1])])
            grid_gps.append([long_lat_start[j,:], long_lat_end[j,:], long_lat_end[j+1,:], long_lat_start[j+1,:]])
            #print(f'long_lat_start[{j},:] = {long_lat_start[j,:]}')
        burst_grid_corners.append(grid_corners)
        burst_grid_gps.append(grid_gps)
        
        #print(f"Burst {i} - grid_corners length: {len(burst_grid_corners[i])}, grid_gps length: {len(burst_grid_gps[i])}")

    #Define boundaries of SLC file 
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),(nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    

    return gps_corners, burst_corners, burst_gps_corners, burst_grid_corners, burst_grid_gps






def getAllSwathBoundaries(SLC_file_names, path):
    IW_gps=[]
    for fname in SLC_file_names:
        annotationFile = path + change_extension(fname,"xml")
        swath_gps=getSwathGPSBoundaries(annotationFile)
        IW_gps.append(swath_gps)
    return(IW_gps)


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
# Function to determine which of  aseries of grids a point belongs to
# gps_location ->(latitude, longitude)
# gps_grid     -> array of grid gps corners in format [(sw[lat,long], nw(lat,long),ne(lat,long),se(lat,long)] 
def getGrid(gps_location, gps_grid):
    count = 0
    for i in range(0,len(gps_grid)):   
        if isPointInROI(gps_location, gps_grid[i]):
            burst = i
            count +=1    
    if count != 1:
        print(f'ERROR: getGrid has found {count} bursts containing location :{gps_location}')
        return -1
    else:
        return burst


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
    reshaped_bgb        = np.array([[x[0], x[1][0]] for x in burst_grid_boundary]) 
    ship_row, ship_col  = gps_to_cartesian_svd(location, reshaped_bggb, reshaped_bgb)
    
    # Check that interpolated value is within boundary and set to boundary if not
    if ship_row  < reshaped_bgb[0][0]:
        ship_row = reshaped_bgb[0][0]
    if ship_row  > reshaped_bgb[1][0]:
        ship_row = reshaped_bgb[1][0]
    if ship_col  < reshaped_bgb[0][1]:
        ship_col = reshaped_bgb[0][1]
    if ship_col  > reshaped_bgb[3][1]:
        ship_col = reshaped_bgb[3][1]
    
    return ship_row, ship_col







# function to accept 2 contiguous bursts and plot them
# based on burst start and end line in df
# burst_list -> list of 2 string values for burst numbers
# src -> image file
# azFraction -> controls fraction of azimuth extent to plot; centered on boundary of bursts:
#            1-> displays all of azimuth extent
#          0.5-> displays half of azimuth extent
# rangeFraction -> similar
#
def plotBursts(img_array, df, burstList, azFraction, rangeFraction):
    
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
                
        
