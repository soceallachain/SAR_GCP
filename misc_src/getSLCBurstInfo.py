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
            merged_start_line = burst_start_line      #+ round(np.mean(overlap_offset_lines))   #### NOT SURE ABOUT THIS, but may be source of offset as initially had been 0
#            print(f' Added offset to first line of firsr burst of {np.mean(overlap_offset_lines)}  --REMEMBER TO REMOVE IF NOT NEEDED')
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
# Function to determine which of a series of grids a point belongs to
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
    return ship_row, ship_col

"""  
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
    return array[row_start:row_end, col_start:col_end]



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

def get_chips_and_masks(img_array, burstList, burst_boundaries, ship_bounding_boxes, ship_burst_indices, chip_border_size):

    mpl.rcParams['figure.figsize'] = [14.0, 8.0]
    nRangeSamples          = img_array.shape[1]
    number_of_range_pixels = burst_boundaries[0][3][1] - burst_boundaries[0][0][1] + 1
    if nRangeSamples != number_of_range_pixels:
        print(f'ERROR: Width of SLC Image is : {nRangeSamples} does not match number of range samples: {number_of_range_pixels}, calculated from burst boundary information')

    img                    = []
    burst_bounding_boxes   = np.array(ship_bounding_boxes)
    bounding_box_row_size  = burst_bounding_boxes[0,1,0]- burst_bounding_boxes[0,0,0]
    chip_row_size          = bounding_box_row_size + 2*chip_border_size
    bounding_box_col_size  = burst_bounding_boxes[0,3,1]- burst_bounding_boxes[0,0,1]
    chip_col_size          = bounding_box_row_size + 2*chip_border_size
    # Need to know the maximum number of ships per burst to set size of chips_from_all_bursts
    # to ensure successive bursts can be appended. Determine form incidence of bursts in 
    # ship_burst_indices     Note: need to import collections
    max_ships_per_burst    = collections.Counter(ship_burst_indices).most_common(1)[0][1]
    #chips_from_all_bursts  = []
    chips_from_all_bursts  = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    chips_from_one_burst   = np.zeros((chip_row_size, chip_col_size*max_ships_per_burst))
    first_time = True
    for burst_number in burstList:
        # Get bounding boxes for ships only in this burst
        bounding_boxes = burst_bounding_boxes[np.array(ship_burst_indices)==(burst_number -1)]
        # Get burst boundary
        burst_boundary = burst_boundaries[burst_number-1]
        print(f' In get_chips_and_masks   ...... burst boundary is : {burst_boundary}')
        print(f' Number of ships in this burst #{burst_number} is {bounding_boxes.shape}')
        az_start       = burst_boundary[0][0]
        az_end         = burst_boundary[1][0]
        range_start    = burst_boundary[0][1]
        range_end      = burst_boundary[3][1]
        # Extract burst image and copy to img
        #img.append(img_array[az_start:az_end, range_start:range_end])
        burst_img      = img_array[az_start:az_end, range_start:range_end]
        # Now save real and imaginary components of SLC file for this burst
        #np.savetxt(f"{SLC_safe_id}_real_burst#{burst_number}.csv", np.real(burst_img), delimiter = ",")
        #np.savetxt(f"{SLC_safe_id}_imag_burst#{burst_number}.csv", np.imag(burst_img), delimiter = ",")
        #img = np.concatenate(img, axis=0)
        #img = np.real(np.sqrt(img*np.conjugate(img)))
        #mask = np.zeros(img.shape, dtype=np.uint8)
        # Loop over each ship in burst and generate coordinates for bounding rectangle 
        # when plotted. Need to add offset for azimuth position if plotting multiple bursts.
        # If plotting random bursts must subtract burst start position as offset for rectangles
        # Add end row of previous burst to position

        ship_loc_img,binary_image_ground_truth, ship_chips, binary_ship_chips= generate_burst_binary_mask(burst_img, burst_boundary, bounding_boxes, chip_border_size)

        np.save(f"{SLC_safe_id}_real_burst#{burst_number}", np.real(burst_img).astype(np.int16))
        np.save(f"{SLC_safe_id}_imag_burst#{burst_number}", np.imag(burst_img).astype(np.int16))
        np.save(f"{SLC_safe_id}_binary_mask_burst#{burst_number}", np.array(binary_image_ground_truth).astype(np.uint8))

        # Determine size of ship_chips array and then assign it to a space of the same size in 
        # chips_from_one_burst
        ship_chips_transposed                                             = ship_chips.T

        print(f'ship_chips_transposed has shape: {ship_chips_transposed.shape} which is row and col size')
        ship_chips_row_size                                               = ship_chips_transposed.shape[0]
        ship_chips_col_size                                               = ship_chips_transposed.shape[1]
        chips_from_one_burst                                              = chips_from_one_burst * 0
        print(f'chips_from_one_burst shape : {chips_from_one_burst.shape}')
        chips_from_one_burst[0:ship_chips_row_size, 0:ship_chips_col_size] = ship_chips_transposed
        if first_time:
            chips_from_all_bursts = chips_from_one_burst
            #chips_from_all_bursts[0:ship_chips_row_size,0:ship_chips_col_size] = chips_from_one_burst[0:ship_chips_row_size, 0:ship_chips_col_size]
        else:
            chips_from_all_bursts = np.concatenate((chips_from_all_bursts, chips_from_one_burst), axis=0)
            #chips_from_all_bursts = np.concatenate(chips_from_one_burst, axis=0)


        fig1, ax1 = plt.subplots(1,2)
        ax1[0].imshow(chips_from_all_bursts, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
        ax1[1].imshow(binary_ship_chips.T, cmap='Greys_r', aspect = '10')
        plt.show()

        chips_from_one_burst[0:ship_chips_row_size,0:ship_chips_col_size] =  binary_ship_chips.T
        #chips_from_all_bursts = np.concatenate(chips_from_one_burst, axis=0)
        chips_from_all_bursts = np.concatenate((chips_from_all_bursts, chips_from_one_burst), axis=0)

        first_time = False
    print(f'Shape of chips_from_all_bursts is:{chips_from_all_bursts.shape}')
    fig, ax = plt.subplots()
    ax.imshow(chips_from_all_bursts, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '1')
    plt.show()

    fig2, ax2 = plt.subplots(1,2)
    ax2[1].imshow(binary_image_ground_truth,cmap='Greys_r', aspect = '10')
    ax2[0].imshow(ship_loc_img, norm=colors.LogNorm(vmin=10,vmax = 300), aspect = '10')
    plt.show()
    return chips_from_all_bursts, ship_loc_img, binary_image_ground_truth, ship_chips, binary_ship_chips



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
# img                 - image of burst already extracted from full SLC image, complex data
# burst_boundaries    - [row,col] co-ordinates of burst vertices in full SLC image co-ordinates
# ship_bounding_boxes - [row,col] vertices of box around ship locations relative to full SLC image vertices
# border              - size of border to impose on bounding box mask
def generate_burst_binary_mask(image, burst_boundary, ship_bounding_boxes, border):

    # Convert complex image to amplitude only
    img                      = np.abs(image)
    img_mask                 = np.zeros(img.shape,dtype=np.uint8)
    img_rectangle_mask       = np.zeros(img.shape,dtype=np.uint8)
    binary_img_mask          = np.zeros(img.shape,dtype=np.uint8)
    number_of_ships          = ship_bounding_boxes.shape[0]
    default_bounding_box_dim = 100
    if number_of_ships == 0:
        bounding_box_dim = default_bounding_box_dim
    else:
        bounding_box_dim       = (ship_bounding_boxes[0][1][0]-ship_bounding_boxes[0][0][0])
    chip_size              = [bounding_box_dim + 2*border, bounding_box_dim + 2*border]
    print(f'Chip bounding box size (chip_size) is : {chip_size}')
    ship_chip              = np.ones(chip_size, dtype= np.uint8)
    ship_chip_mask         = np.ones(chip_size, dtype= np.uint8)
    binary_ship_chip       = np.ones(chip_size, dtype= np.uint8)
    binary_ship_chip_mask  = np.ones(chip_size, dtype= np.uint8)
    rectangle_mask         = np.zeros(chip_size, dtype= np.uint8)
    dilated_rectangle_mask = np.ones(chip_size, dtype= np.uint8)

    burst_start_row       = burst_boundary[0][0]
    row_offset            = round(border)
    col_offset            = round(border)

    kernel = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])

    for j in range (0, ship_bounding_boxes.shape[0]):
        row     = ship_bounding_boxes[j][0][0] - burst_start_row -90 # WHY IS THIS OFFSET NEEDED? To do with overlap?
        column  = ship_bounding_boxes[j][0][1]                   -25
        h       = ship_bounding_boxes[j][1][0] - ship_bounding_boxes[j][0][0]
        w       = ship_bounding_boxes[j][3][1] - ship_bounding_boxes[j][0][1]
        roi     = create_roi(img,row, row+h, column, column+w)
        print(f' roi shape for j={j} is {roi.shape}'   )
        img_mask [row:row+h, column:column+w]                                  = np.copy(roi)
        ship_chip[row_offset:row_offset + h, col_offset:col_offset + w]        = np.copy(roi)
        binary_roi                                                             = create_ship_binary_mask(roi)  
        binary_ship_chip[row_offset:row_offset + h, col_offset:col_offset + w] = np.copy(binary_roi)
        binary_img_mask[row:row+h, column:column+w]                            = np.copy(binary_roi)
        if j==0:
            ship_chips_mask         = ship_chip             # Is this 'if' statement really needed?
            binary_ship_chips_mask  = binary_ship_chip
        else:
            ship_chips_mask         = np.concatenate((ship_chips_mask, ship_chip.T),0)
            binary_ship_chips_mask  = np.concatenate((binary_ship_chips_mask, binary_ship_chip.T),0)
        # Check that ship locations are within viewing field set by azFraction and rangeFraction

        # Need to highlight positions of ships in a mask : use dilation and subtraction to generate
        # outline of boundary and place in mask image
        rectangle_mask  [row_offset:row_offset + h, col_offset:col_offset + w] = 1
        dilated_rectangle_mask                                                 = dilation(rectangle_mask, kernel)
        number_of_dilations = 4
        for j in range(0,number_of_dilations):
            dilated_rectangle_mask = dilation(dilated_rectangle_mask, kernel)
        dilated_rectangle_mask = dilated_rectangle_mask - rectangle_mask          # Create outline
        dilated_rectangle_mask = dilated_rectangle_mask * 1e4                     # Give outline high value
        dilated_rectangle_mask = dilated_rectangle_mask + 1                       # Make inner pixels =1 

        print(f' Value of row {row} - row_offset {row_offset} is : {row - row_offset} h is {h}; column {column}-col_offset{col_offset} is: {column-col_offset} w is {w}; dilated_rectangle_mask.shape is : {dilated_rectangle_mask.shape}')
        ## Adding a kluge to get over some issue with assigning boundary box and forcing area to be 100x100
        if (row_offset*2 + h != dilated_rectangle_mask.shape[0]) or (col_offset*2 + w != dilated_rectangle_mask.shape[0]):
#            img_rectangle_mask[row - row_offset:row + row_offset + h, column-col_offset:column + col_offset + w] = dilated_rectangle_mask[row_offset:row_offset+h, col_offset:col_offset+w]
            img_rectangle_mask[row - row_offset:row + row_offset + 99, column-col_offset:column + col_offset + 99] = dilated_rectangle_mask
            print(f'row_offset*2 + h = {row_offset*2 + h}, dil_rec_mask.shape = {dilated_rectangle_mask.shape}, col_offset*2 + w ={col_offset*2 + w}')
        else:
            img_rectangle_mask[row - row_offset:row + row_offset + 99, column-col_offset:column + col_offset + 99] = dilated_rectangle_mask
            #img_rectangle_mask[row - row_offset:row + row_offset + h, column-col_offset:column + col_offset + w] = dilated_rectangle_mask
            print(f'row_offset*2 + h = {row_offset*2 + h}, dil_rec_mask.shape = {dilated_rectangle_mask.shape}, col_offset*2 + w ={col_offset*2 + w}')

    # Multiply image mask with roi around ships by mask with bounding boxes. Should end up with original image with bounding box
    # around each ship
    img_rectangle_mask = img_rectangle_mask + img
    #img_rectangle_mask = img
    if number_of_ships ==0:
        ship_chips_mask        = ship_chip_mask
        binary_ship_chips_mask = binary_ship_chip_mask
    return img_rectangle_mask, binary_img_mask, ship_chips_mask, binary_ship_chips_mask



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

## Read annotation file for burst paramaters and timing information
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
print(f'Number of bursts in {SLC_fnames[swath_of_interest_index]} is {nBursts}')
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


print(gridPoints[gridPoints['burst_number'] == '5'])
#print(df['azimuth_burst_end_time'])


pixel_value = '0'
burst_start_times = df[df['pixel'] == pixel_value]['azimuth_time']

"""
# Process each value: convert to datetime, add  convert back to string
burst_end_time = []
for time_str in burst_start_times[0:len(burst_start_times)-1]:
    # Convert string to datetime object
    burst_start_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')

    # Add burst interval time to start time
    t = burst_start_time + timedelta(seconds=(nBurstLines*azTimeInterval))

    # Convert back to string if necessary (optional)
    burst_end_time.append(t.strftime('%Y-%m-%dT%H:%M:%S.%f'))

# Output the result
print(f'End times for each burst are: {burst_end_time}')
"""



#Define boundaries of SLC file
file_corners = [(0,0),(nBurstLines*nBursts-1,0),(nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
#gps corner locations for ROI of IW swath - Note values are strings
gps_corners = find_coordinates_array(df, file_corners)
print(f'GPS corners of SLC file are {gps_corners}')


# Process each value: convert to datetime, add offset, convert back to string
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

# Output the result

print(f'Number of overlap lines for each burst are: {number_overlap_lines}')


#####
# Remove overlap between bursts i & i+1 by subtracting overlap[i]/2 from end line of each burst i
# and adding overlap[i]/2 to the start of i+1
####

overlap_offset_lines = [x//2 for x in number_overlap_lines]
print(f'Overlap offset lines is {overlap_offset_lines}')
print(f'**************************************')

merged_burst_start_line = []
merged_burst_end_line   = []
gridPoints['merged_burst_start_line']= 0
gridPoints['merged_burst_end_line']  = 0

for i in range(0,nBursts):
    burst_start_line  = gridPoints[gridPoints['burst_number'] == str(i+1)]['line'].to_numpy().astype(int) 
    burst_end_line    = gridPoints[gridPoints['burst_number'] == str(i+2)]['line'].to_numpy().astype(int) 
    if i==0:
        merged_start_line = burst_start_line
        merged_end_line   = burst_end_line   - round(overlap_offset_lines[i])
    else:
        merged_start_line = burst_start_line + round(overlap_offset_lines[i-1])
        merged_end_line   = burst_end_line   - round(overlap_offset_lines[i])
    
    gridPoints.loc[gridPoints['burst_number'] == str(i+1), 'merged_burst_start_line'] = merged_start_line
    gridPoints.loc[gridPoints['burst_number'] == str(i+1), 'merged_burst_end_line']   = merged_end_line



print(gridPoints.head(30))
print(gridPoints.tail(30))
IW_gps = getAllSwathBoundaries(SLC_fnames, Annotation_dir)
print(f'IW1 gps boundaries : {IW_gps[0]}')





print(gridPoints.loc[(gridPoints['burst_number'] == '1') & (gridPoints['pixel'] == '0')]['longitude'])


"""
# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[0]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
array = src.read(1)

# Test merging of bursts - plot 2 adjacent ones
burst_list     = [1,2]
az_fraction    = 0.5
range_fraction = 0.5
plotBursts(array, gridPoints, burst_list, az_fraction, range_fraction)
"""


df1         = pd.read_csv(label_file)
ship_long   = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lon']
ship_lat    = df1[(df1['swath_index'] == swath_of_interest) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']==SLC_safe_id)]['detect_lat']
#ship_long   = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lon']
#ship_lat    = df1[(df1['swath_index'] == 3) & (df1['is_vessel']==True) & (df1['SLC_product_identifier']=='S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E')]['detect_lat']
ship_coords = [ship_long, ship_lat]
ship_coords = np.transpose(ship_coords)
ship_coords = np.reshape(ship_coords, (len(ship_coords),2))


print(f'Output of isPointInROI: {isPointInROI(ship_coords[0,:],IW_gps[2])}')
print(f' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('f First 5 ship locations for file {SLC_product_identifier} are: {ship_coords[0:4,:]}')
print(f' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


IW_gps=getAllSwathBoundaries(SLC_fnames, Annotation_dir)
IW_gps_boundary,burst_boundaries,burst_gps_boundaries,burst_grid_boundaries,burst_grid_gps_boundaries = getSwathBurstGridGPSBoundaries(annotation_file)

ship_location= ship_coords[2,:]
print(f'Ship at {ship_location} is in IW{swath_of_interest} is {isPointInROI(ship_location,IW_gps[swath_of_interest_index])}')
print(f'******************************')
print(f'burst_gps_boundaries is:    {burst_gps_boundaries}')

print(f'******************************')


burst_index    = getGrid(ship_location,burst_gps_boundaries)
subgrid_index  = getGrid(ship_location,burst_grid_gps_boundaries[burst_index])
print(f'Ship at {ship_location} is in Burst # {burst_index}')

srr, scc = getShipInterpolatedPositionInBurst(ship_location,burst_gps_boundaries[burst_index] , burst_boundaries[burst_index])
print(f'Ship at {ship_location} is in Burst # {burst_index+1} and Sub-grid {subgrid_index+1} is located at row: {srr} and column: {scc} in global burst grid co-ordinates' )

srr1, scc1 = getShipInterpolatedPositionInGrid(ship_location, burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
print(f'Ship at {ship_location} is in Burst # {burst_index+1} and Sub-grid {subgrid_index+1} is located at row: {srr1} and column: {scc1} in local grid co-ordinates' )




#####################################################################################################
#
# Have Swath, Burst and Grid for 'a' ship.So can now determine interpolated position in merged bursts.
# Need to extend to all ships in burst. Will process on a burst by burst basis. So helpful to know
# how many ships in each burst so if will know need not process and can just generate generic
# binary mask. If ships in burst then locate them and plot bounding boxes to view and confirm
# accurate positioning. Once confirmed extract region around ship and apply Miguel's code.
#
#


# clearance around ship centroid for bounding box
clearance  = 50
ship_coordinates =ship_coords
good_ship_coords = np.array(getGoodShipCoords(ship_coordinates,burst_grid_gps_boundaries, burst_gps_boundaries))
ship_coords=[]
ship_coords = good_ship_coords
print(f'Number of good ships in swath is: {good_ship_coords.shape}')
print(f'Number of good ships in swath is: {ship_coords.shape}')

number_of_ships_in_swath = ship_coords.shape[0]
interp_ship_coords= []
burst_indices     = []
bounding_boxes    = []
for i in range(0, number_of_ships_in_swath):
    burst_index            = getGrid(ship_coords[i,:], burst_gps_boundaries)
    subgrid_index          = getGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index])
    if (burst_index == -1) or (subgrid_index == -1) :
        print(f'ERROR: Ship location {ship_coords[i]} :burst_index = {burst_index}, subgrid_index = {subgrid_index} not found correctly - not adding to ship locations')
    else:
        interp_row, interp_col = getShipInterpolatedPositionInGrid(ship_coords[i,:], burst_grid_gps_boundaries[burst_index][subgrid_index],burst_grid_boundaries[burst_index][subgrid_index])
        r_row                  = round(interp_row)
        r_col                  = round(interp_col) 
        interp_ship_coords.append([r_row, r_col])
        interp_coords          = np.array(interp_ship_coords)
        burst_indices.append(burst_index)
        print(f' Ship Location = {ship_coords[i]}  , Interpolated Ship Position in Burst #{burst_index+1} and Grid #{subgrid_index+1} is: {interp_coords[i]} for file {annotation_file}')
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
    
    # Have bounding box for each ship in swath in np.array
swath_bounding_boxes = np.array(bounding_boxes)
#bounding_img = img([box_start_row:box_end_row,box_start_col:box_end_col])




##########################################################################################################################
# Have ship location information in merged bursts
# Can now plot burst and overlay with ship bounding boxes
#

# Open SLC file for plotting merged bursts
if os.path.exists(src_file):
    print(f"{SLC_fnames[swath_of_interest_index]} found")
else:
    print(f"ERROR: {src_file} Not found. Exiting")
    #exit(0)

src   = rasterio.open(src_file)
img   = src.read(1)
#img = np.real(np.sqrt(img*np.conjugate(img)))
#img = np.square(np.real(img)) + np.square(np.imag(img))

# Test merging of bursts - plot 2 adjacent ones
#bursts_to_plot     =list(range(0,nBursts))
bursts_to_plot     =[1]
#az_fraction    = 1
#range_fraction = 1
#plotBursts(img, gridPoints, burst_list, az_fraction, range_fraction)


for burst in bursts_to_plot:
#for i in range (0, nBursts):
    burst_boxes = swath_bounding_boxes[np.array(burst_indices)== burst-1 ]
    print(f' Ship Bounding Boxes for burst index={i} has shape: {burst_boxes.shape}')
    print(f'swath_bounding_boxes  shape :{swath_bounding_boxes.shape} : {swath_bounding_boxes}')

 #   plotBurstsII(img, bursts_to_plot, burst_boundaries, swath_bounding_boxes, burst_indices)
    border = 25     # Put border of 20 pixels around ship chips => total size of chip is  (clearance+border)*2 = 140x140
    print(f' Before plotBurstsIII - following values ....')
    print(f' bursts_to_plot is : {bursts_to_plot}   File is {SLC_fnames[swath_of_interest_index]} ....')
    print(f' merged burst_boundares for burst #{bursts_to_plot} = {burst_boundaries}')


    plotBurstsIII(img, bursts_to_plot, burst_boundaries, swath_bounding_boxes, burst_indices, border) 

    #for j in range (0, burst_ship_coordinates.shape[0]):
     #   print(f' Ship Co-ordinates in Burst #{i} are : {burst_ship_coordinates[j]}, Cartesian [row,column] co-ordinates are: {interp_ship_coords[j]}')
        #plot_burst(burst)
        #overlay_ship_bounding_boxes(burst_ship_coordinates[j], bounding_boxes[j])
burst_row_start = burst_boundaries[bursts_to_plot[0]-1][0][0]
burst_row_end   = burst_boundaries[bursts_to_plot[0]-1][1][0]
burst_col_start = burst_boundaries[bursts_to_plot[0]-1][0][1]
burst_col_end   = burst_boundaries[bursts_to_plot[0]-1][3][1]

burst_img  = np.abs(img[burst_row_start:burst_row_end,burst_col_start:burst_col_end])
fig, ax = plt.subplots()
ax.imshow(burst_img,norm=colors.LogNorm(vmin=15,vmax = 700), aspect = '10')
plt.show()


burst_img  = np.square(np.real(img[burst_row_start:burst_row_end,burst_col_start:burst_col_end])) + np.square(np.imag(img[burst_row_start:burst_row_end,burst_col_start:burst_col_end])) 
fig, ax = plt.subplots()
ax.imshow(burst_img,norm=colors.LogNorm(vmin=200,vmax = 20000), aspect = '10')
plt.show()


###############################################################################################
# test get_ship_chips

#all_burst_chips,b,c,d,e = get_chips_and_masks(img, bursts_to_plot, burst_boundaries, swath_bounding_boxes, burst_indices, border)
# return chips_from_all_bursts, ship_loc_img, binary_image_ground_truth, ship_chips, binary_ship_chips
