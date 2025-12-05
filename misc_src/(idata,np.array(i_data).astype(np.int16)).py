# coding: utf-8
from lxml import etree
# system modules
import os
import io
import re
import sys
import random
import shutil
# data manipulation
import pandas as pd
import math
import numpy as np
import rasterio
import shapely
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.measure as skm
from skimage.morphology import erosion, dilation
from skimage import exposure
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2
import time

from matplotlib import colors
from datetime import datetime, timedelta
from shapely import LineString, Point, Polygon






pd.options.mode.chained_assignment = None  # default='warn'

#pd.set_option('precision', 12)
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
        #result = res.to_numpy().astype(float)
        if not result.empty:
            longitude = result['longitude'].values[0]
            latitude = result['latitude'].values[0]
            coordinates.append((float(longitude), float(latitude)))
        else:
            coordinates.append((None, None))  # Append (None, None) if no match is found
    
    return coordinates

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
# Function returns the gps coordinates of each swath boundary
# Also returns the de-bursted boundaries for each burst in [row,col]
# and gps format. The [row,col] indices are with respect to the original 
# Sentinel-1 SLC iwx file. So for example while Burst #1 might have an azimuth 
# extent of 0->1517 in the original SLC, several of these lines are due to 
# overlap with Burst #2 and so Burst #1 actually comprises 0-> 14xx where the 
# other lines comprising half the overlap are ignored. Similarly while Burst 
# #2 in the original would have started at line 1518 its start location is moved
# upwards, due to the overlap to something like 16xx.
#
def getSwathBurstGPSBoundaries(annotation_file):
    
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

    #Define boundaries of SLC file 
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),(nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    return gps_corners, burst_corners, burst_gps_corners



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

    
    
def getGridII(gps_location, gps_grid):
    count = 0
    for i in range(0,len(gps_grid)):   
        grid = np.array(gps_grid[i]).reshape(4,2)
        roi=[(grid[0][0],grid[0][1]), (grid[1][0],grid[1][1]), (grid[2][0],grid[2][1]), (grid[3][0],grid[3][1])]
        print(f'roi[{i}] is            : {roi}')
        print(f' gps_location : is {isPointInROI(gps_location,roi)} that {gps_location} is in {roi}')
        if isPointInROI(gps_location, roi):
            print(f' Made it inside if statement !!!! Count = {count}')
            sub_grid   = i
            count +=1
            print(f' Made it inside if statement !!!! Count = {count}')
     
    if count != 1:
        print(f'ERROR: getGrid has found {count} grids containing location :{gps_location}')
        return -1
    else:
        return sub_grid




def getShipInterpolatedPositionInBurst(location, burst_gps, burst_boundary):
    reshaped_burst_gps      = np.array(burst_gps).reshape(4,2)
    reshaped_burst_boundary = np.array(burst_boundary).reshape(4,2)
    ship_row, ship_col      = gps_to_cartesian(location, reshaped_burst_gps, reshaped_burst_boundary)
    
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
    ship_row, ship_col  = gps_to_cartesian(location, reshaped_bggb, reshaped_bgb)
    
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
os.getcwd()
##############################################################################################
## Set path information

path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
#GRD_dir   ="Dublin"
#SAFE_dir  ="S1A_IW_SLC__1SDV_20240107T063927_20240107T063954_051997_06488E_177F.SAFE"

#SLC_fnames =["s1a-iw1-slc-vh-20240107t063928-20240107t063953-051997-06488e-001.tiff",
#             "s1a-iw2-slc-vh-20240107t063929-20240107t063954-051997-06488e-002.tiff",
#             "s1a-iw3-slc-vh-20240107t063927-20240107t063952-051997-06488e-003.tiff"]

#Cornwall scene
#GRD_dir   ="S1B_IW_GRDH_1SDV_20200715T063117_20200715T063142_022478_02AA98_34D9"
GRD_dir    ="Cornwall"
SAFE_dir  ="S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E.SAFE"
SLC_fnames =["s1b-iw1-slc-vh-20200715t063117-20200715t063142-022478-02aa98-001.tiff",
             "s1b-iw2-slc-vh-20200715t063118-20200715t063143-022478-02aa98-002.tiff",
             "s1b-iw3-slc-vh-20200715t063116-20200715t063141-022478-02aa98-003.tiff"]

#SLC_fnames=["s1a-iw2-slc-vh-20240105t180707-20240105t180732-051975-0647cf-002.tiff"]
#GRD_dir   ="S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53"
#SAFE_dir  ="S1A_IW_SLC__1SDV_20200421T051411_20200421T051438_032221_03BA16_C5BE.SAFE"

################################################################################################
# swath of interest is IW2
swath_of_interest = 2
swath_of_interest_index = swath_of_interest - 1

Annotation_dir  = path + GRD_dir +'/SLC/' + SAFE_dir + '/annotation/'
annotation_file = Annotation_dir + change_extension(SLC_fnames[swath_of_interest_index],"xml")
src_file        = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fnames[swath_of_interest_index]

SLC_safe_id, _ = os.path.splitext(SAFE_dir)
print(SLC_safe_id)


label_file     = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
#################################################################################################
if os.path.exists(annotation_file):
    print(f"{annotation_file} found")
else:
    print(f"ERROR: {annotation_file} Not found. Exiting")
    exit(0)
base_name,_ = os.path.splitext(SAFE_dir)
base_name
IW_gps=getAllSwathBoundaries(SLC_fnames, Annotation_dir)
df1         = pd.read_csv(label_file)
ship_long   = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lon']
ship_lat    = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lat']
#ship_long   = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lon']
#ship_lat    = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lat']
ship_coords = [ship_long, ship_lat]
ship_coords = np.transpose(ship_coords)
ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
print(f' First 5 ship locations for file {SLC_safe_id} are: {ship_coords[0:4,:]}')


print(f'Output of isPointInROI: {isPointInROI(ship_coords[0,:],IW_gps[2])}')


IW_gps=getAllSwathBoundaries(SLC_fnames, Annotation_dir)
IW_gps_boundary,burst_boundaries,burst_gps_boundaries,burst_grid_boundaries,burst_grid_gps_boundaries=getSwathBurstGridGPSBoundaries(annotation_file)

ship_location= ship_coords[2,:]
print(f'Ship at {ship_location} is in IW{swath_of_interest} is {isPointInROI(ship_location,IW_gps[swath_of_interest_index])}')
print(f'******************************')
print(f'burst_gps_boundaries is:    {burst_gps_boundaries}')

print(f'******************************')


burst_index    = getGrid(ship_location,burst_gps_boundaries)
subgrid_index  = getGrid(ship_location,burst_grid_gps_boundaries[burst_index])




burst_index=getGrid(ship_location,burst_gps_boundaries)
print(f'Ship at {ship_location} is in Burst # {burst_index}')
subgrid_index=getGrid(ship_location,burst_grid_gps_boundaries[burst_index])
#burst_gps      = burst_gps_boundaries[burst_index]
#burst_boundary = burst_boundaries[burst_index]

srr, scc = getShipInterpolatedPositionInBurst(ship_location,burst_gps_boundaries[burst_index] , burst_boundaries[burst_index])
print(f'Ship at {ship_location} is in Burst # {burst_index} and Sub-grid {subgrid_index} is located at row: {srr} and column: {scc} in global burst grid co-ordinates' )

#burst_grid_boundary     = burst_grid_boundaries[burst_index][subgrid_index]
#burst_grid_gps_boundary = burst_grid_gps_boundaries[burst_index][subgrid_index]

srr1, scc1 = getShipInterpolatedPositionInGrid(ship_location, burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
print(f'Ship at {ship_location} is in Burst # {burst_index} and Sub-grid {subgrid_index} is located at row: {srr1} and column: {scc1} in local grid co-ordinates' )
ship_coords[0,:]
number_of_ships_in_swath = ship_coords.shape[0]
number_of_ships_in_swath
bggpsb=np.array(burst_grid_gps_boundaries[burst_index])
bggpsb[burst_index]
subgrid_index = getGrid(ship_coords[1,:], bggpsb[burst_index])
subgrid_index_index
# Function to extract only those ships that can be idenified as definitely 
# belonging to a burst and a grid within the burst
# Unsure why a ship would be in a burst but not its associated grid
# Note: Find out!
def getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries):
    good_ship_coords=[]
    for i in range (0, ship_coords.shape[0]):
        burst_index   = getGrid(ship_coords[i,:], burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index])
        if (burst_index !=-1) and (subgrid_index != -1):
                    good_ship_coords.append(ship_coords[i])
                    
    
def getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries):
    good_ship_coords = []
    
    for i in range(ship_coords.shape[0]):
        burst_index = getGrid(ship_coords[i, :], burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i, :], burst_grid_gps_boundaries[burst_index])
        
        if (burst_index != -1) and (subgrid_index != -1):
            good_ship_coords.append(ship_coords[i])
    
    return good_ship_coords  # Return the list of good ship coordinates
print(f'Number of ships in swath is: {ship_coords.shape}')

good_ship_coords = np.array(getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries))
print(f'Number of good ships in swath is: {good_ship_coords.shape}')
number_of_ships_in_swath = ship_coords.shape[0]
# clearance around ship centroid for bounding box
clearance  = 50

interp_ship_coords= []
burst_indices     = []
bounding_boxes    = []

for i in range(0, number_of_ships_in_swath):
    burst_index            = getGrid(ship_coords[i,:], burst_gps_boundaries)
    subgrid_index          = getGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index])
    print(f' subgrid_index = {subgrid_index}')
    interp_row, interp_col = getShipInterpolatedPositionInGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
    r_row                  = round(interp_row)
    r_col                  = round(interp_col)
    interp_ship_coords.append([r_row, r_col])
    burst_indices.append(burst_index)
    #print(f' Ship Location = {ship_coords[i]}  , Interpolated Ship Position in Burst #{burst_index+1} and Grid #{subgrid_index+1} is: {interp_ship_coords[i]}') 
    

    if r_row - burst_boundaries[burst_index][0][0] > clearance:
        box_start_row = r_row-clearance + 1
    else:
        box_start_row = burst_boundaries[burst_index][0][0]

    if burst_boundaries[burst_index][1][0] - r_row > clearance:
        box_end_row = r_row + clearance
    else:
        box_end_row = burst_boundaries[burst_index][1][0]

    if r_col - burst_boundaries[burst_index][0][1] > clearance:
        box_start_col = r_col - clearance + 1
    else:
        box_start_col = burst_boundaries[burst_index][0][1]

    if burst_boundaries[burst_index][2][1] - r_col > clearance:
        box_end_col = r_col + clearance
    else:
        box_end_col = burst_boundaries[burst_index][2][1]


    bounding_box =np.array([[box_start_row, box_start_col],
                            [box_end_row,   box_start_col],
                            [box_end_row,   box_end_col],
                            [box_start_row, box_end_col]])
    bounding_boxes.append(bounding_box)

    #bounding_img =src(box_start_row:box_end_row,box_start_col:box_end_col)

swath_bounding_boxes = np.array(bounding_boxes)
ss=np.array(burst_indices)
bool_ind = ss ==5
burst_ship_coordinates=ship_coords[np.array(burst_indices)==3]
burst_ship_coordinates.shape[0]
np.array(burst_gps_boundaries[6])
print(f'{isPointInROI(([-4.7831939, 49.41320054]), burst_grid_gps_boundaries[6][4])}')
print(f'{isPointInROI(([-4.7831939, 49.41320054]), burst_gps_boundaries[6])}')

np.array(burst_grid_gps_boundaries[6][5])
getGrid(([-4.7831939, 49.41320054]), burst_grid_gps_boundaries[6])
burst_bounding_boxes=np.array(bounding_boxes)
burst_boxes = burst_bounding_boxes[np.array(burst_indices)==5]
burst_boxes[0][0][0]
interp_coords=np.array(interp_ship_coords)
cart_coords=interp_coords[np.array(burst_indices)==5]
cart_coords.shape
swath_of_interest
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
src_file
ship_bounding_boxes = swath_bounding_boxes
ship_burst_indices =burst_indices
img_array      = img

nRangeSamples          = img_array.shape[1]
number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1 
burst_bounding_boxes   = np.array(ship_bounding_boxes)

if nRangeSamples != number_of_range_pixels:
    print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')
ship_bounding_boxes.shape
azimuth_offset = 0 
range_offset   = 0 
rec            = []

burst_number=4
import matplotlib.patches as patches

A = np.random.rand(100, 100) 
bounding_box_dim    = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0]) 
bounding_box_dim

roi     = create_roi(img,row, row+h, column, column+w)
roi.shape
aaa=img[100:200, 300:400]
img.shape
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
    
    threshold = avg_ROI + (1*stddev_ROI)
    #threshold = threshold_otsu(ROI)*1
    #print(f'Otsu Threshold = {threshold}  min = {np.min(ROI)}, max = {np.max(ROI)} mean = {np.mean(ROI)}, std = {np.std(ROI)}')
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
        #ROI2 = ROI
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
burst_list=[4]
img                    = []
nRangeSamples          = img_array.shape[1]
number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1 
burst_bounding_boxes   = np.array(ship_bounding_boxes)
azFraction =1
rangeFraction=1

if nRangeSamples != number_of_range_pixels:
    print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')

#with plt.style.context("dark_background"):
#    plt.figure(figsize=(20, 20))
#    plt.title(f"SLC IW  {SAFE_dir}")
mpl.rcParams['figure.figsize'] = [14.0, 8.0]
fig, ax = plt.subplots(3,1)

azimuth_offset = 0 
range_offset   = 0 
rec            = []
bounding_box_border = 25   # Should be clearance/2
bounding_box_dim    = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0]) # +1?
ship_chips_mask     = np.ones([bounding_box_dim + 2*bounding_box_border, bounding_box_dim + 2*bounding_box_border], dtype= np.uint8)
binary_ship_chips_mask =np.ones(ship_chips_mask.shape, dtype = np.uint8)
for burst_number in burst_list:
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
    img = np.real(np.sqrt((img*np.conjugate(img))))
    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 1000), aspect = '10')
    #create a mask same size as burst image
    img_mask            = np.zeros(img.shape,dtype=np.uint8)
    binary_img_mask     = np.zeros(img.shape,dtype=np.uint8)
    ship_chip           = np.ones(ship_chips_mask.shape, dtype= np.uint8)
    binary_ship_chip    = np.ones(ship_chips_mask.shape, dtype= np.uint8)

    row_offset = round(bounding_box_border/2)
    col_offset = round(bounding_box_border/2)
    # Loop over each ship in burst and generate coordinates for bounding rectangle 
    # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
    # If plotting random bursts must subtract burst start position as offset for rectangles
    # Add end row of previous burst to position
    for j in range (0, bounding_boxes.shape[0]):
    #for j in range (0, 6):
        row     = bounding_boxes[j][0][0] - b_az_start -90 # WHY IS THIS OFFSET NEEDED? To do with overlap?
        column  = bounding_boxes[j][0][1]              -25
        h       = bounding_boxes[j][1][0] - bounding_boxes[j][0][0]
        w       = bounding_boxes[j][3][1] - bounding_boxes[j][0][1]
        roi     = create_roi(img,row, row+h, column, column+w)
        #roi     =img[row:row+h, column:column+w]
        print(f' row ={row}, row_offset={row_offset}, h={h}, col={column}, w={w}')
        img_mask[row:row+h, column:column+w]                                   = np.copy(roi)
        ship_chip       [row_offset:row_offset + h, col_offset:col_offset + w] = np.copy(roi)
        binary_ship_chip[row_offset:row_offset + h, col_offset:col_offset + w] = create_ship_binary_mask(roi)
        binary_img_mask[row:row+h, column:column+w]                            = create_ship_binary_mask(roi)
        #mask4[row_offset:row_offset + h, col_offset:col_offset +w]      = binary_ship_mask
        if j==0:
            ship_chips_mask         = ship_chip
            binary_ship_chips_mask  = binary_ship_chip
        else:
            ship_chips_mask         = np.concatenate((ship_chips_mask, ship_chip.T),0)
            binary_ship_chips_mask  = np.concatenate((binary_ship_chips_mask, binary_ship_chip.T),0)
        # Check that ship locations are within viewing field set by azFraction and rangeFraction
        # If not do not plot them
        #if row in range(b_az_start,b_az_end) and column in range(b_range_start, b_range_end):
        # If plotting multiple bursts need to ensure ship bounding rectangles move with offset introduced by each successive burst 
    s_img=img*binary_img_mask
    

    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 700), aspect = '5')
    
    #plt.imshow(amp_img,norm=colors.LogNorm(vmin=10,vmax = 700),  aspect='15')
    #Use these with multiple burst processing 
    azimuth_offset = b_az_end - b_az_start
    range_offset   = 0 
    
ax[0].imshow(ship_chips_mask.T,norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '1')
ax[1].imshow(binary_ship_chips_mask.T,cmap='Greys_r', aspect = '1')
ax[2].imshow(binary_img_mask,cmap='Greys_r', aspect = '10')
#ax[3].imshow(np.real(np.sqrt(img_array*np.conjugate(img_array))),norm=colors.LogNorm(vmin=15,vmax = 800), aspect = '1')


plt.style.context("dark_background")
plt.tight_layout()
plt.xlabel("Range Samples")
plt.ylabel("Azimuth Lines)")
plt.show()
fig, ax1 = plt.subplots()
aaa=img_array[b_az_start:b_az_end, b_range_start:b_range_end]
ax1.imshow(np.real(np.sqrt(aaa*np.conjugate(aaa))),norm=colors.LogNorm(vmin=10,vmax = 500), aspect = '10')
plt.show()
ship_bounding_boxes.shape[1:2]
plt.imshow(binary_image_mask,cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(binary_img_mask,cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(binary_img_mask,cmap='gray', aspect = 10)
plt.axis('off')
plt.show()
ss=np.array(burst_indices)
bool_ind = ss ==5
burst_ship_coordinates=ship_coords[np.array(burst_indices)==3]
burst_ship_coordinates.shape[0]
burst_indices
ss=np.array(burst_indices)
bool_ind = ss ==5
burst_ship_coordinates=ship_coords[np.array(burst_indices)==3]
burst_ship_coordinates.shape[0]
burst_indices
counter = collections.Counter(burst_indices)
import collections
counter = collections.Counter(burst_indices)
import collections
counter = collections.Counter(burst_indices)
counter.shape
import collections
counter = collections.Counter(burst_indices)
counter
import collections
counter = collections.Counter(burst_indices)
counter[0]
import collections
counter = collections.Counter(burst_indices)
counter
import collections
counter = collections.Counter(burst_indices)
counter[1]
import collections
counter = collections.Counter(burst_indices)
np.max(counter)
import collections
counter = collections.Counter(burst_indices)
np.max(counter).[0]
import collections
counter = collections.Counter(burst_indices)
np.max(counter)[0]
import collections
counter = collections.Counter(burst_indices)
np.max(counter)[1]
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(0)
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(2)
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)[1]
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)[0][1]
ss=np.array(burst_indices)
bool_ind = ss ==5
burst_ship_coordinates=ship_coords[np.array(burst_indices)==3]
burst_ship_coordinates.shape[0]
burst_indices
ss=np.array(burst_indices)
bool_ind = ss ==5
burst_ship_coordinates=ship_coords[np.array(burst_indices)==3]
burst_ship_coordinates.shape[0]
burst_indices.most_common()
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)[0][1]
max_ships_per_burst = collections.Counter(burst_indices).counter.most_common(1)[0][1]
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)[0][1]
max_ships_per_burst = collections.Counter(burst_indices).most_common(1)[0][1]
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)[0][1]
max_ships_per_burst = collections.Counter(burst_indices).most_common(1)[0][1]
print(max_ships_per_burst)
ship_bounding_boxes[0].shape
ship_bounding_boxes[0].shape
ship_bounding_box[0][0][0]
ship_bounding_boxes[0].shape
tt=ship_bounding_box[0]
tt[0,0]
ship_bounding_boxes[0].shape
tt=ship_bounding_boxes[0]
tt[0,0]
ship_bounding_boxes[0].shape
tt=ship_bounding_boxes[0]
tt[1,0]-tt[0,0]
ship_bounding_boxes[0].shape
tt=ship_bounding_boxes[0]
tt[2,1]-tt[0,1]
ship_bounding_boxes[0].shape
tt=ship_bounding_boxes[0]
tt[2,1]-tt[0,1]
ship_bounding_boxes[0,2,1]-ship_bounding_boxes[0,0,1]
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
hh.dtype
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
np.floor(1.2)
np.ceiling(1.2)
np.round(1.2)
np.sum(np.round(hh)-np.floor(hh)
np.sum(np.round(hh)-np.floor(hh))
np.sum(hh-np.floor(hh))
np.sum(hh-np.round(hh))
ff=hh.astype(uint16)
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
2^16
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh, dtype=int16)
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
sys.getsizeof(hh)
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
sys.getsizeof(jj)
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
sys.getsizeof(jj)
jj.shape
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
sys.getsizeof(jj)
#jj.shape
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
sys.getsizeof(jj)
jj.shape
real_data.shape
from lxml import etree
# system modules
import os
import io
import re
import sys
import random
import shutil
# data manipulation
import pandas as pd
import math
import numpy as np
import rasterio
import shapely
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.measure as skm
from skimage.morphology import erosion, dilation
from skimage import exposure
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2
import time

from matplotlib import colors
from datetime import datetime, timedelta
from shapely import LineString, Point, Polygon






pd.options.mode.chained_assignment = None  # default='warn'

#pd.set_option('precision', 12)
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
        #result = res.to_numpy().astype(float)
        if not result.empty:
            longitude = result['longitude'].values[0]
            latitude = result['latitude'].values[0]
            coordinates.append((float(longitude), float(latitude)))
        else:
            coordinates.append((None, None))  # Append (None, None) if no match is found
    
    return coordinates

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
# Function returns the gps coordinates of each swath boundary
# Also returns the de-bursted boundaries for each burst in [row,col]
# and gps format. The [row,col] indices are with respect to the original 
# Sentinel-1 SLC iwx file. So for example while Burst #1 might have an azimuth 
# extent of 0->1517 in the original SLC, several of these lines are due to 
# overlap with Burst #2 and so Burst #1 actually comprises 0-> 14xx where the 
# other lines comprising half the overlap are ignored. Similarly while Burst 
# #2 in the original would have started at line 1518 its start location is moved
# upwards, due to the overlap to something like 16xx.
#
def getSwathBurstGPSBoundaries(annotation_file):
    
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

    #Define boundaries of SLC file 
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),(nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    return gps_corners, burst_corners, burst_gps_corners



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

    
    
def getGridII(gps_location, gps_grid):
    count = 0
    for i in range(0,len(gps_grid)):   
        grid = np.array(gps_grid[i]).reshape(4,2)
        roi=[(grid[0][0],grid[0][1]), (grid[1][0],grid[1][1]), (grid[2][0],grid[2][1]), (grid[3][0],grid[3][1])]
        print(f'roi[{i}] is            : {roi}')
        print(f' gps_location : is {isPointInROI(gps_location,roi)} that {gps_location} is in {roi}')
        if isPointInROI(gps_location, roi):
            print(f' Made it inside if statement !!!! Count = {count}')
            sub_grid   = i
            count +=1
            print(f' Made it inside if statement !!!! Count = {count}')
     
    if count != 1:
        print(f'ERROR: getGrid has found {count} grids containing location :{gps_location}')
        return -1
    else:
        return sub_grid




def getShipInterpolatedPositionInBurst(location, burst_gps, burst_boundary):
    reshaped_burst_gps      = np.array(burst_gps).reshape(4,2)
    reshaped_burst_boundary = np.array(burst_boundary).reshape(4,2)
    ship_row, ship_col      = gps_to_cartesian(location, reshaped_burst_gps, reshaped_burst_boundary)
    
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
    ship_row, ship_col  = gps_to_cartesian(location, reshaped_bggb, reshaped_bgb)
    
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
os.getcwd()
##############################################################################################
## Set path information

path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
#GRD_dir   ="Dublin"
#SAFE_dir  ="S1A_IW_SLC__1SDV_20240107T063927_20240107T063954_051997_06488E_177F.SAFE"

#SLC_fnames =["s1a-iw1-slc-vh-20240107t063928-20240107t063953-051997-06488e-001.tiff",
#             "s1a-iw2-slc-vh-20240107t063929-20240107t063954-051997-06488e-002.tiff",
#             "s1a-iw3-slc-vh-20240107t063927-20240107t063952-051997-06488e-003.tiff"]

#Cornwall scene
#GRD_dir   ="S1B_IW_GRDH_1SDV_20200715T063117_20200715T063142_022478_02AA98_34D9"
GRD_dir    ="Cornwall"
SAFE_dir  ="S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E.SAFE"
SLC_fnames =["s1b-iw1-slc-vh-20200715t063117-20200715t063142-022478-02aa98-001.tiff",
             "s1b-iw2-slc-vh-20200715t063118-20200715t063143-022478-02aa98-002.tiff",
             "s1b-iw3-slc-vh-20200715t063116-20200715t063141-022478-02aa98-003.tiff"]

#SLC_fnames=["s1a-iw2-slc-vh-20240105t180707-20240105t180732-051975-0647cf-002.tiff"]
#GRD_dir   ="S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53"
#SAFE_dir  ="S1A_IW_SLC__1SDV_20200421T051411_20200421T051438_032221_03BA16_C5BE.SAFE"

################################################################################################
# swath of interest is IW2
swath_of_interest = 2
swath_of_interest_index = swath_of_interest - 1

Annotation_dir  = path + GRD_dir +'/SLC/' + SAFE_dir + '/annotation/'
annotation_file = Annotation_dir + change_extension(SLC_fnames[swath_of_interest_index],"xml")
src_file        = path + GRD_dir +'/SLC/' + SAFE_dir + '/measurement/' + SLC_fnames[swath_of_interest_index]

SLC_safe_id, _ = os.path.splitext(SAFE_dir)
print(SLC_safe_id)


label_file     = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
#################################################################################################
if os.path.exists(annotation_file):
    print(f"{annotation_file} found")
else:
    print(f"ERROR: {annotation_file} Not found. Exiting")
    exit(0)
base_name,_ = os.path.splitext(SAFE_dir)
base_name
IW_gps=getAllSwathBoundaries(SLC_fnames, Annotation_dir)
df1         = pd.read_csv(label_file)
ship_long   = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lon']
ship_lat    = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lat']
#ship_long   = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lon']
#ship_lat    = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lat']
ship_coords = [ship_long, ship_lat]
ship_coords = np.transpose(ship_coords)
ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
print(f' First 5 ship locations for file {SLC_safe_id} are: {ship_coords[0:4,:]}')


print(f'Output of isPointInROI: {isPointInROI(ship_coords[0,:],IW_gps[2])}')


IW_gps=getAllSwathBoundaries(SLC_fnames, Annotation_dir)
IW_gps_boundary,burst_boundaries,burst_gps_boundaries,burst_grid_boundaries,burst_grid_gps_boundaries=getSwathBurstGridGPSBoundaries(annotation_file)

ship_location= ship_coords[2,:]
print(f'Ship at {ship_location} is in IW{swath_of_interest} is {isPointInROI(ship_location,IW_gps[swath_of_interest_index])}')
print(f'******************************')
print(f'burst_gps_boundaries is:    {burst_gps_boundaries}')

print(f'******************************')


burst_index    = getGrid(ship_location,burst_gps_boundaries)
subgrid_index  = getGrid(ship_location,burst_grid_gps_boundaries[burst_index])




burst_index=getGrid(ship_location,burst_gps_boundaries)
print(f'Ship at {ship_location} is in Burst # {burst_index}')
subgrid_index=getGrid(ship_location,burst_grid_gps_boundaries[burst_index])
#burst_gps      = burst_gps_boundaries[burst_index]
#burst_boundary = burst_boundaries[burst_index]

srr, scc = getShipInterpolatedPositionInBurst(ship_location,burst_gps_boundaries[burst_index] , burst_boundaries[burst_index])
print(f'Ship at {ship_location} is in Burst # {burst_index} and Sub-grid {subgrid_index} is located at row: {srr} and column: {scc} in global burst grid co-ordinates' )

#burst_grid_boundary     = burst_grid_boundaries[burst_index][subgrid_index]
#burst_grid_gps_boundary = burst_grid_gps_boundaries[burst_index][subgrid_index]

srr1, scc1 = getShipInterpolatedPositionInGrid(ship_location, burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
print(f'Ship at {ship_location} is in Burst # {burst_index} and Sub-grid {subgrid_index} is located at row: {srr1} and column: {scc1} in local grid co-ordinates' )
ship_coords[0,:]
number_of_ships_in_swath = ship_coords.shape[0]
number_of_ships_in_swath
bggpsb=np.array(burst_grid_gps_boundaries[burst_index])
bggpsb[burst_index]
subgrid_index = getGrid(ship_coords[1,:], bggpsb[burst_index])
subgrid_index_index
# Function to extract only those ships that can be idenified as definitely 
# belonging to a burst and a grid within the burst
# Unsure why a ship would be in a burst but not its associated grid
# Note: Find out!
def getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries):
    good_ship_coords=[]
    for i in range (0, ship_coords.shape[0]):
        burst_index   = getGrid(ship_coords[i,:], burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index])
        if (burst_index !=-1) and (subgrid_index != -1):
                    good_ship_coords.append(ship_coords[i])
                    
    
def getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries):
    good_ship_coords = []
    
    for i in range(ship_coords.shape[0]):
        burst_index = getGrid(ship_coords[i, :], burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i, :], burst_grid_gps_boundaries[burst_index])
        
        if (burst_index != -1) and (subgrid_index != -1):
            good_ship_coords.append(ship_coords[i])
    
    return good_ship_coords  # Return the list of good ship coordinates
print(f'Number of ships in swath is: {ship_coords.shape}')

good_ship_coords = np.array(getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries))
print(f'Number of good ships in swath is: {good_ship_coords.shape}')
number_of_ships_in_swath = ship_coords.shape[0]
# clearance around ship centroid for bounding box
clearance  = 50

interp_ship_coords= []
burst_indices     = []
bounding_boxes    = []

for i in range(0, number_of_ships_in_swath):
    burst_index            = getGrid(ship_coords[i,:], burst_gps_boundaries)
    subgrid_index          = getGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index])
    print(f' subgrid_index = {subgrid_index}')
    interp_row, interp_col = getShipInterpolatedPositionInGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
    r_row                  = round(interp_row)
    r_col                  = round(interp_col)
    interp_ship_coords.append([r_row, r_col])
    burst_indices.append(burst_index)
    #print(f' Ship Location = {ship_coords[i]}  , Interpolated Ship Position in Burst #{burst_index+1} and Grid #{subgrid_index+1} is: {interp_ship_coords[i]}') 
    

    if r_row - burst_boundaries[burst_index][0][0] > clearance:
        box_start_row = r_row-clearance + 1
    else:
        box_start_row = burst_boundaries[burst_index][0][0]

    if burst_boundaries[burst_index][1][0] - r_row > clearance:
        box_end_row = r_row + clearance
    else:
        box_end_row = burst_boundaries[burst_index][1][0]

    if r_col - burst_boundaries[burst_index][0][1] > clearance:
        box_start_col = r_col - clearance + 1
    else:
        box_start_col = burst_boundaries[burst_index][0][1]

    if burst_boundaries[burst_index][2][1] - r_col > clearance:
        box_end_col = r_col + clearance
    else:
        box_end_col = burst_boundaries[burst_index][2][1]


    bounding_box =np.array([[box_start_row, box_start_col],
                            [box_end_row,   box_start_col],
                            [box_end_row,   box_end_col],
                            [box_start_row, box_end_col]])
    bounding_boxes.append(bounding_box)

    #bounding_img =src(box_start_row:box_end_row,box_start_col:box_end_col)

swath_bounding_boxes = np.array(bounding_boxes)
ss=np.array(burst_indices)
bool_ind = ss ==5
burst_ship_coordinates=ship_coords[np.array(burst_indices)==3]
burst_ship_coordinates.shape[0]
burst_indices
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)[0][1]
max_ships_per_burst = collections.Counter(burst_indices).most_common(1)[0][1]
print(max_ships_per_burst)
np.array(burst_gps_boundaries[6])
print(f'{isPointInROI(([-4.7831939, 49.41320054]), burst_grid_gps_boundaries[6][4])}')
print(f'{isPointInROI(([-4.7831939, 49.41320054]), burst_gps_boundaries[6])}')

np.array(burst_grid_gps_boundaries[6][5])
getGrid(([-4.7831939, 49.41320054]), burst_grid_gps_boundaries[6])
burst_bounding_boxes=np.array(bounding_boxes)
burst_boxes = burst_bounding_boxes[np.array(burst_indices)==5]
burst_boxes[0][0][0]
interp_coords=np.array(interp_ship_coords)
cart_coords=interp_coords[np.array(burst_indices)==5]
cart_coords.shape
swath_of_interest
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
sys.getsizeof(jj)
jj.shape
ff=hh.astype(uint16)
src_file
ship_bounding_boxes = swath_bounding_boxes
ship_burst_indices =burst_indices
img_array      = img

nRangeSamples          = img_array.shape[1]
number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1 
burst_bounding_boxes   = np.array(ship_bounding_boxes)

if nRangeSamples != number_of_range_pixels:
    print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')
ship_bounding_boxes[0].shape
tt=ship_bounding_boxes[0]
tt[2,1]-tt[0,1]
ship_bounding_boxes[0,2,1]-ship_bounding_boxes[0,0,1]
azimuth_offset = 0 
range_offset   = 0 
rec            = []

burst_number=4
import matplotlib.patches as patches

A = np.random.rand(100, 100) 
bounding_box_dim    = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0]) 
bounding_box_dim

roi     = create_roi(img,row, row+h, column, column+w)
roi.shape
aaa=img[100:200, 300:400]
img.shape
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
    
    threshold = avg_ROI + (1*stddev_ROI)
    #threshold = threshold_otsu(ROI)*1
    #print(f'Otsu Threshold = {threshold}  min = {np.min(ROI)}, max = {np.max(ROI)} mean = {np.mean(ROI)}, std = {np.std(ROI)}')
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
        #ROI2 = ROI
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
burst_list=[4]
img                    = []
nRangeSamples          = img_array.shape[1]
number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1 
burst_bounding_boxes   = np.array(ship_bounding_boxes)
azFraction =1
rangeFraction=1

if nRangeSamples != number_of_range_pixels:
    print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')

#with plt.style.context("dark_background"):
#    plt.figure(figsize=(20, 20))
#    plt.title(f"SLC IW  {SAFE_dir}")
mpl.rcParams['figure.figsize'] = [14.0, 8.0]
fig, ax = plt.subplots(3,1)

azimuth_offset = 0 
range_offset   = 0 
rec            = []
bounding_box_border = 25   # Should be clearance/2
bounding_box_dim    = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0]) # +1?
ship_chips_mask     = np.ones([bounding_box_dim + 2*bounding_box_border, bounding_box_dim + 2*bounding_box_border], dtype= np.uint8)
binary_ship_chips_mask =np.ones(ship_chips_mask.shape, dtype = np.uint8)
for burst_number in burst_list:
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
    img = np.real(np.sqrt((img*np.conjugate(img))))
    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 1000), aspect = '10')
    #create a mask same size as burst image
    img_mask            = np.zeros(img.shape,dtype=np.uint8)
    binary_img_mask     = np.zeros(img.shape,dtype=np.uint8)
    ship_chip           = np.ones(ship_chips_mask.shape, dtype= np.uint8)
    binary_ship_chip    = np.ones(ship_chips_mask.shape, dtype= np.uint8)

    row_offset = round(bounding_box_border/2)
    col_offset = round(bounding_box_border/2)
    # Loop over each ship in burst and generate coordinates for bounding rectangle 
    # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
    # If plotting random bursts must subtract burst start position as offset for rectangles
    # Add end row of previous burst to position
    for j in range (0, bounding_boxes.shape[0]):
    #for j in range (0, 6):
        row     = bounding_boxes[j][0][0] - b_az_start -90 # WHY IS THIS OFFSET NEEDED? To do with overlap?
        column  = bounding_boxes[j][0][1]              -25
        h       = bounding_boxes[j][1][0] - bounding_boxes[j][0][0]
        w       = bounding_boxes[j][3][1] - bounding_boxes[j][0][1]
        roi     = create_roi(img,row, row+h, column, column+w)
        #roi     =img[row:row+h, column:column+w]
        print(f' row ={row}, row_offset={row_offset}, h={h}, col={column}, w={w}')
        img_mask[row:row+h, column:column+w]                                   = np.copy(roi)
        ship_chip       [row_offset:row_offset + h, col_offset:col_offset + w] = np.copy(roi)
        binary_ship_chip[row_offset:row_offset + h, col_offset:col_offset + w] = create_ship_binary_mask(roi)
        binary_img_mask[row:row+h, column:column+w]                            = create_ship_binary_mask(roi)
        #mask4[row_offset:row_offset + h, col_offset:col_offset +w]      = binary_ship_mask
        if j==0:
            ship_chips_mask         = ship_chip
            binary_ship_chips_mask  = binary_ship_chip
        else:
            ship_chips_mask         = np.concatenate((ship_chips_mask, ship_chip.T),0)
            binary_ship_chips_mask  = np.concatenate((binary_ship_chips_mask, binary_ship_chip.T),0)
        # Check that ship locations are within viewing field set by azFraction and rangeFraction
        # If not do not plot them
        #if row in range(b_az_start,b_az_end) and column in range(b_range_start, b_range_end):
        # If plotting multiple bursts need to ensure ship bounding rectangles move with offset introduced by each successive burst 
    s_img=img*binary_img_mask
    

    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 700), aspect = '5')
    
    #plt.imshow(amp_img,norm=colors.LogNorm(vmin=10,vmax = 700),  aspect='15')
    #Use these with multiple burst processing 
    azimuth_offset = b_az_end - b_az_start
    range_offset   = 0 
    
ax[0].imshow(ship_chips_mask.T,norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '1')
ax[1].imshow(binary_ship_chips_mask.T,cmap='Greys_r', aspect = '1')
ax[2].imshow(binary_img_mask,cmap='Greys_r', aspect = '10')
#ax[3].imshow(np.real(np.sqrt(img_array*np.conjugate(img_array))),norm=colors.LogNorm(vmin=15,vmax = 800), aspect = '1')


plt.style.context("dark_background")
plt.tight_layout()
plt.xlabel("Range Samples")
plt.ylabel("Azimuth Lines)")
plt.show()
fig, ax1 = plt.subplots()
aaa=img_array[b_az_start:b_az_end, b_range_start:b_range_end]
ax1.imshow(np.real(np.sqrt(aaa*np.conjugate(aaa))),norm=colors.LogNorm(vmin=10,vmax = 500), aspect = '10')
plt.show()
ship_bounding_boxes.shape[1:2]
array_normalized = ((binary_img_mask - np.min(binary_img_mask)) / (np.max(binary_img_mask) - np.min(binary_img_mask)) * 255).astype(np.uint8)
image_bin = Image.fromarray(array_normalized)
image_bin.save('Cornwall.bmp')

img1   = src.read(1)
real_data = np.real(img1[b_az_start:b_az_end, b_range_start:b_range_end])
imag_data = np.imag(img1[b_az_start:b_az_end, b_range_start:b_range_end])
np.save('Cornwall_real.npy', real_data)
np.save('Cornwall_imag.npy', imag_data)
img1.dtype
real_data.shape
sys.getsizeof(real_data)
sys.getsizeof(real_data)
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.sizeof(kk)}')
sys.getsizeof(real_data)
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data)
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data[0][0])
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data)
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data)
kk.nbytes
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data)
kk.nbytes
np.array(real_data).nbytes
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data)
kk.nbytes
print(f'np.array(real_data).nbytes = {np.array(real_data).nbytes}')
real_data.shape
real_data.nbytes
real_data.dtype
imag_data.nbytes
img1.nbytes
img1.shape
img1.dtype
real_h=np.array(real_data).astype(np.int16)
real_h=np.array(real_data).astype(np.int16)
real_h.nbytes
imag_h=np.array(imag_data).astype(np.int16)
imag_h=np.array(imag_data).astype(np.int16)
imag_h.nbytes
get_ipython().run_line_magic('pwd', '')
path
path
SAFE_dir
path
SAFE_dir
SLC_src_id
path
SAFE_dir
SLCsrc_id
path
SAFE_dir
print(path)
SAFE_dir
print(pwd)
print(path)
SAFE_dir
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('ls', '-alt *.csv')
np.savetxt('test.csv',np.array(real_data).astype(np.int16), delimiter =',')
get_ipython().run_line_magic('ls', '-alt *.csv')
np.savetxt('test.dat',np.array(real_h), delimiter =' ')
get_ipython().run_line_magic('ls', '-alt *.dat')
real_h.shape
real_h.nbytes
array_normalized = ((binary_img_mask - np.min(binary_img_mask)) / (np.max(binary_img_mask) - np.min(binary_img_mask)) * 255).astype(np.uint8)
image_bin = Image.fromarray(array_normalized)
image_bin.save('Cornwall.bmp')

img1   = src.read(1)
real_data = np.copy(np.real(img1[b_az_start:b_az_end, b_range_start:b_range_end]))
imag_data = np.copy(np.imag(img1[b_az_start:b_az_end, b_range_start:b_range_end]))
np.save('Cornwall_real.npy', real_data)
np.save('Cornwall_imag.npy', imag_data)
img1.dtype
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data)
kk.nbytes
print(f'np.array(real_data).nbytes = {np.array(real_data).nbytes}')
real_h=np.array(real_data).astype(np.int16)
real_h.nbytes
imag_h=np.array(imag_data).astype(np.int16)
imag_h.nbytes
imag_h=np.array(imag_data).astype(np.int16)
imag_h.nbytes
print(pwd)
print(path)
SAFE_dir
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '-alt *.csv')
np.savetxt('test.dat',np.array(real_h), delimiter =' ')
get_ipython().run_line_magic('ls', '-alt *.dat')
np.save('test', real_h)
get_ipython().run_line_magic('ls', '-alt test*')
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '-alt *.csv')
np.array(np.real(img1)).astype(np.int16).nbytes
np.array(np.real(img1)).nbytes
np.real(img1).nbytes
np.real(img1).astype(np.int16).nbytes
binary_image_mask.shape
plt.imshow(binary_img_mask,cmap='gray', aspect = 10)
plt.axis('off')
plt.show()
bin_image_filename=fnames[swath_of_interest] +
n.savetxt(SLC_src_id
IW_gps=getAllSwathBoundaries(SLC_fnames, Annotation_dir)
df1         = pd.read_csv(label_file)
ship_long   = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lon']
ship_lat    = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lat']
#ship_long   = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lon']
#ship_lat    = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lat']
ship_coords = [ship_long, ship_lat]
ship_coords = np.transpose(ship_coords)
ship_coords = np.reshape(ship_coords, (len(ship_coords),2))
print(f' First 5 ship locations for file {SLC_safe_id} are: {ship_coords[0:4,:]}')


print(f'Output of isPointInROI: {isPointInROI(ship_coords[0,:],IW_gps[2])}')


IW_gps=getAllSwathBoundaries(SLC_fnames, Annotation_dir)
IW_gps_boundary,burst_boundaries,burst_gps_boundaries,burst_grid_boundaries,burst_grid_gps_boundaries=getSwathBurstGridGPSBoundaries(annotation_file)

ship_location= ship_coords[2,:]
print(f'Ship at {ship_location} is in IW{swath_of_interest} is {isPointInROI(ship_location,IW_gps[swath_of_interest_index])}')
print(f'******************************')
print(f'burst_gps_boundaries is:    {burst_gps_boundaries}')

print(f'******************************')


burst_index    = getGrid(ship_location,burst_gps_boundaries)
subgrid_index  = getGrid(ship_location,burst_grid_gps_boundaries[burst_index])




burst_index=getGrid(ship_location,burst_gps_boundaries)
print(f'Ship at {ship_location} is in Burst # {burst_index}')
subgrid_index=getGrid(ship_location,burst_grid_gps_boundaries[burst_index])
#burst_gps      = burst_gps_boundaries[burst_index]
#burst_boundary = burst_boundaries[burst_index]

srr, scc = getShipInterpolatedPositionInBurst(ship_location,burst_gps_boundaries[burst_index] , burst_boundaries[burst_index])
print(f'Ship at {ship_location} is in Burst # {burst_index} and Sub-grid {subgrid_index} is located at row: {srr} and column: {scc} in global burst grid co-ordinates' )

#burst_grid_boundary     = burst_grid_boundaries[burst_index][subgrid_index]
#burst_grid_gps_boundary = burst_grid_gps_boundaries[burst_index][subgrid_index]

srr1, scc1 = getShipInterpolatedPositionInGrid(ship_location, burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
print(f'Ship at {ship_location} is in Burst # {burst_index} and Sub-grid {subgrid_index} is located at row: {srr1} and column: {scc1} in local grid co-ordinates' )
ship_coords[0,:]
number_of_ships_in_swath = ship_coords.shape[0]
number_of_ships_in_swath
bggpsb=np.array(burst_grid_gps_boundaries[burst_index])
bggpsb[burst_index]
subgrid_index = getGrid(ship_coords[1,:], bggpsb[burst_index])
subgrid_index_index
# Function to extract only those ships that can be idenified as definitely 
# belonging to a burst and a grid within the burst
# Unsure why a ship would be in a burst but not its associated grid
# Note: Find out!
def getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries):
    good_ship_coords=[]
    for i in range (0, ship_coords.shape[0]):
        burst_index   = getGrid(ship_coords[i,:], burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index])
        if (burst_index !=-1) and (subgrid_index != -1):
                    good_ship_coords.append(ship_coords[i])
                    
    
def getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries):
    good_ship_coords = []
    
    for i in range(ship_coords.shape[0]):
        burst_index = getGrid(ship_coords[i, :], burst_gps_boundaries)
        subgrid_index = getGrid(ship_coords[i, :], burst_grid_gps_boundaries[burst_index])
        
        if (burst_index != -1) and (subgrid_index != -1):
            good_ship_coords.append(ship_coords[i])
    
    return good_ship_coords  # Return the list of good ship coordinates
print(f'Number of ships in swath is: {ship_coords.shape}')

good_ship_coords = np.array(getGoodShipCoords(ship_coords, burst_grid_gps_boundaries, burst_gps_boundaries))
print(f'Number of good ships in swath is: {good_ship_coords.shape}')
number_of_ships_in_swath = ship_coords.shape[0]
# clearance around ship centroid for bounding box
clearance  = 50

interp_ship_coords= []
burst_indices     = []
bounding_boxes    = []

for i in range(0, number_of_ships_in_swath):
    burst_index            = getGrid(ship_coords[i,:], burst_gps_boundaries)
    subgrid_index          = getGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index])
    print(f' subgrid_index = {subgrid_index}')
    interp_row, interp_col = getShipInterpolatedPositionInGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
    r_row                  = round(interp_row)
    r_col                  = round(interp_col)
    interp_ship_coords.append([r_row, r_col])
    burst_indices.append(burst_index)
    #print(f' Ship Location = {ship_coords[i]}  , Interpolated Ship Position in Burst #{burst_index+1} and Grid #{subgrid_index+1} is: {interp_ship_coords[i]}') 
    

    if r_row - burst_boundaries[burst_index][0][0] > clearance:
        box_start_row = r_row-clearance + 1
    else:
        box_start_row = burst_boundaries[burst_index][0][0]

    if burst_boundaries[burst_index][1][0] - r_row > clearance:
        box_end_row = r_row + clearance
    else:
        box_end_row = burst_boundaries[burst_index][1][0]

    if r_col - burst_boundaries[burst_index][0][1] > clearance:
        box_start_col = r_col - clearance + 1
    else:
        box_start_col = burst_boundaries[burst_index][0][1]

    if burst_boundaries[burst_index][2][1] - r_col > clearance:
        box_end_col = r_col + clearance
    else:
        box_end_col = burst_boundaries[burst_index][2][1]


    bounding_box =np.array([[box_start_row, box_start_col],
                            [box_end_row,   box_start_col],
                            [box_end_row,   box_end_col],
                            [box_start_row, box_end_col]])
    bounding_boxes.append(bounding_box)

    #bounding_img =src(box_start_row:box_end_row,box_start_col:box_end_col)

swath_bounding_boxes = np.array(bounding_boxes)
ss=np.array(burst_indices)
bool_ind = ss ==5
burst_ship_coordinates=ship_coords[np.array(burst_indices)==3]
burst_ship_coordinates.shape[0]
burst_indices
import collections
counter = collections.Counter(burst_indices)
print(counter)
np.max(counter)
counter.most_common(1)[0][1]
max_ships_per_burst = collections.Counter(burst_indices).most_common(1)[0][1]
print(max_ships_per_burst)
np.array(burst_gps_boundaries[6])
print(f'{isPointInROI(([-4.7831939, 49.41320054]), burst_grid_gps_boundaries[6][4])}')
print(f'{isPointInROI(([-4.7831939, 49.41320054]), burst_gps_boundaries[6])}')

np.array(burst_grid_gps_boundaries[6][5])
getGrid(([-4.7831939, 49.41320054]), burst_grid_gps_boundaries[6])
burst_bounding_boxes=np.array(bounding_boxes)
burst_boxes = burst_bounding_boxes[np.array(burst_indices)==5]
burst_boxes[0][0][0]
interp_coords=np.array(interp_ship_coords)
cart_coords=interp_coords[np.array(burst_indices)==5]
cart_coords.shape
swath_of_interest
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
swath_of_interest
print(f'img dtype is {img.dtype}')
hh=np.real(img)
print(f' img real component dtype is {hh.dtype}')
print(f'max value of real part is {hh.max()}')
print(f'min value of real part is {hh.min()}')
np.power(2,16)
jj=np.array(hh).astype(np.int16)
sys.getsizeof(jj)
jj.shape
ff=hh.astype(uint16)
src_file
ship_bounding_boxes = swath_bounding_boxes
ship_burst_indices =burst_indices
img_array      = img

nRangeSamples          = img_array.shape[1]
number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1 
burst_bounding_boxes   = np.array(ship_bounding_boxes)

if nRangeSamples != number_of_range_pixels:
    print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')
ship_bounding_boxes[0].shape
tt=ship_bounding_boxes[0]
tt[2,1]-tt[0,1]
ship_bounding_boxes[0,2,1]-ship_bounding_boxes[0,0,1]
azimuth_offset = 0 
range_offset   = 0 
rec            = []

burst_number=4
import matplotlib.patches as patches

A = np.random.rand(100, 100) 
bounding_box_dim    = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0]) 
bounding_box_dim

roi     = create_roi(img,row, row+h, column, column+w)
roi.shape
aaa=img[100:200, 300:400]
img.shape
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
    
    threshold = avg_ROI + (1*stddev_ROI)
    #threshold = threshold_otsu(ROI)*1
    #print(f'Otsu Threshold = {threshold}  min = {np.min(ROI)}, max = {np.max(ROI)} mean = {np.mean(ROI)}, std = {np.std(ROI)}')
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
        #ROI2 = ROI
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
burst_list=[4]
img                    = []
nRangeSamples          = img_array.shape[1]
number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1 
burst_bounding_boxes   = np.array(ship_bounding_boxes)
azFraction =1
rangeFraction=1

if nRangeSamples != number_of_range_pixels:
    print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')

#with plt.style.context("dark_background"):
#    plt.figure(figsize=(20, 20))
#    plt.title(f"SLC IW  {SAFE_dir}")
mpl.rcParams['figure.figsize'] = [14.0, 8.0]
fig, ax = plt.subplots(3,1)

azimuth_offset = 0 
range_offset   = 0 
rec            = []
bounding_box_border = 25   # Should be clearance/2
bounding_box_dim    = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0]) # +1?
ship_chips_mask     = np.ones([bounding_box_dim + 2*bounding_box_border, bounding_box_dim + 2*bounding_box_border], dtype= np.uint8)
binary_ship_chips_mask =np.ones(ship_chips_mask.shape, dtype = np.uint8)
for burst_number in burst_list:
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
    img = np.real(np.sqrt((img*np.conjugate(img))))
    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 1000), aspect = '10')
    #create a mask same size as burst image
    img_mask            = np.zeros(img.shape,dtype=np.uint8)
    binary_img_mask     = np.zeros(img.shape,dtype=np.uint8)
    ship_chip           = np.ones(ship_chips_mask.shape, dtype= np.uint8)
    binary_ship_chip    = np.ones(ship_chips_mask.shape, dtype= np.uint8)

    row_offset = round(bounding_box_border/2)
    col_offset = round(bounding_box_border/2)
    # Loop over each ship in burst and generate coordinates for bounding rectangle 
    # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
    # If plotting random bursts must subtract burst start position as offset for rectangles
    # Add end row of previous burst to position
    for j in range (0, bounding_boxes.shape[0]):
    #for j in range (0, 6):
        row     = bounding_boxes[j][0][0] - b_az_start -90 # WHY IS THIS OFFSET NEEDED? To do with overlap?
        column  = bounding_boxes[j][0][1]              -25
        h       = bounding_boxes[j][1][0] - bounding_boxes[j][0][0]
        w       = bounding_boxes[j][3][1] - bounding_boxes[j][0][1]
        roi     = create_roi(img,row, row+h, column, column+w)
        #roi     =img[row:row+h, column:column+w]
        print(f' row ={row}, row_offset={row_offset}, h={h}, col={column}, w={w}')
        img_mask[row:row+h, column:column+w]                                   = np.copy(roi)
        ship_chip       [row_offset:row_offset + h, col_offset:col_offset + w] = np.copy(roi)
        binary_ship_chip[row_offset:row_offset + h, col_offset:col_offset + w] = create_ship_binary_mask(roi)
        binary_img_mask[row:row+h, column:column+w]                            = create_ship_binary_mask(roi)
        #mask4[row_offset:row_offset + h, col_offset:col_offset +w]      = binary_ship_mask
        if j==0:
            ship_chips_mask         = ship_chip
            binary_ship_chips_mask  = binary_ship_chip
        else:
            ship_chips_mask         = np.concatenate((ship_chips_mask, ship_chip.T),0)
            binary_ship_chips_mask  = np.concatenate((binary_ship_chips_mask, binary_ship_chip.T),0)
        # Check that ship locations are within viewing field set by azFraction and rangeFraction
        # If not do not plot them
        #if row in range(b_az_start,b_az_end) and column in range(b_range_start, b_range_end):
        # If plotting multiple bursts need to ensure ship bounding rectangles move with offset introduced by each successive burst 
    s_img=img*binary_img_mask
    

    #ax.imshow(img,norm=colors.LogNorm(vmin=30,vmax = 700), aspect = '5')
    
    #plt.imshow(amp_img,norm=colors.LogNorm(vmin=10,vmax = 700),  aspect='15')
    #Use these with multiple burst processing 
    azimuth_offset = b_az_end - b_az_start
    range_offset   = 0 
    
ax[0].imshow(ship_chips_mask.T,norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '1')
ax[1].imshow(binary_ship_chips_mask.T,cmap='Greys_r', aspect = '1')
ax[2].imshow(binary_img_mask,cmap='Greys_r', aspect = '10')
#ax[3].imshow(np.real(np.sqrt(img_array*np.conjugate(img_array))),norm=colors.LogNorm(vmin=15,vmax = 800), aspect = '1')


plt.style.context("dark_background")
plt.tight_layout()
plt.xlabel("Range Samples")
plt.ylabel("Azimuth Lines)")
plt.show()
fig, ax1 = plt.subplots()
aaa=img_array[b_az_start:b_az_end, b_range_start:b_range_end]
ax1.imshow(np.real(np.sqrt(aaa*np.conjugate(aaa))),norm=colors.LogNorm(vmin=10,vmax = 500), aspect = '10')
plt.show()
ship_bounding_boxes.shape[1:2]
array_normalized = ((binary_img_mask - np.min(binary_img_mask)) / (np.max(binary_img_mask) - np.min(binary_img_mask)) * 255).astype(np.uint8)
image_bin = Image.fromarray(array_normalized)
image_bin.save('Cornwall.bmp')

img1   = src.read(1)
real_data = np.copy(np.real(img1[b_az_start:b_az_end, b_range_start:b_range_end]))
imag_data = np.copy(np.imag(img1[b_az_start:b_az_end, b_range_start:b_range_end]))
np.save('Cornwall_real.npy', real_data)
np.save('Cornwall_imag.npy', imag_data)
img1.dtype
kk=np.array(real_data).astype(np.int16)
print(f' size of real data in 16 bit int is :{sys.getsizeof(kk)}')
sys.getsizeof(real_data)
kk.nbytes
print(f'np.array(real_data).nbytes = {np.array(real_data).nbytes}')
r_data=np.load('Cornwall_real.npy')
i_data=np.load('Cornwall_imag.npy')
#data = np.square(r_data) + np.square(i_data)
data = np.square(i_data)
fig, ax5 = plt.subplots()
#aaa=img_array[b_az_start:b_az_end, b_range_start:b_range_end]
#
ax5.imshow(data,norm=colors.LogNorm(vmin=100,vmax = 20000), aspect = '10')

plt.show()
s_img=img*binary_img_mask
fig, ax1 = plt.subplots()
#aaa=img_array[b_az_start:b_az_end, b_range_start:b_range_end]
#ax1.imshow(s_img,norm=colors.LogNorm(vmin=10,vmax = 500), aspect = '10')
ax1.imshow(s_img,cmap='gray', aspect = '10')
plt.show()
plt.imshow(binary_img_mask,cmap='gray', aspect = 10)
plt.axis('off')
plt.show()
binary_image_mask.dtype
binary_image_mask.shape
binary_image_mask.nbytes
np.array(binary_image_mask).nbytes
np.array(binary_img_mask).nbytes
np.array(binary_img_mask).dtype
np.array(binary_img_mask).nbytes
np.array(binary_img_mask).nbytes
i_data.bytes
np.array(binary_img_mask).nbytes
np.array(i_data).nbytes
np.array(binary_img_mask).nbytes
np.array(i_data).dtype
