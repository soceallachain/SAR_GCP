## cleanGenerateTrainingData.py
##
## Code to extract burst information for SLC file


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

import 
##############################################################################################
## Set path information

#path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
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


"""


swath_of_interest = 2
swath_of_interest_index = swath_of_interest - 1
path      ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/TrainingTestData/"

#SLCTrainingFileDirectories = ["S1A_IW_GRDH_1SDV_20200116T051412_20200116T051437_030821_038933_5CD5", 
#                              "S1A_IW_GRDH_1SDV_20200519T171753_20200519T171824_032637_03C7B2_1F74"]


SLCTrainingFileDirectories = os.listdir(path)
"""
for directory in SLCTrainingFileDirectories:
    print(f'{directory} in {SLCTrainingFileDirectories}')
    print(f'-------------------------------------------------')
    SLC_root_file_path = f'{path}/{directory}'
    src_file, annotation_file, a,b = getSLCFilesFromRoot(SLC_root_file_path, swath_of_interest)
    print(f'SLC src file : {src_file}')
    print(f'Annotation  file : {annotation_file}')
"""


##################################################
# Test with 1 file first
SLC_root_file_path   = f'{path}/{SLCTrainingFileDirectories[4]}'
print(f'*****************************************     SLC_root_file_path = {SLC_root_file_path}')
print(f'*****************************************     SLC_root_file_path = {path}/{SLCTrainingFileDirectories[0]}')
src_file, annotation_file, annotation_path, SLC_fnames, SAFE_dir = getSLCFilesFromRoot(SLC_root_file_path, swath_of_interest)
print(f' Annotation file is: {annotation_file}')
print(f' Annotation_dir  is: {annotation_path}')
print(f'SLC_fname are : {SLC_fnames}')
print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
Annotation_dir = annotation_path
SLC_safe_id, _ = os.path.splitext(SAFE_dir)
print(SLC_safe_id)
label_file     = "/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SLC_train.csv"
#################################################################################################
if os.path.exists(annotation_file) and os.path.exists(Annotation_dir) and os.path.exists(src_file):
    print(f"{annotation_file} and {Annotation_dir} and {src_file} found")
else:
    print(f"ERROR: {annotation_file} or {iAnnotation_dir} or {src_file} Not found. Exiting")
    exit(0)

#exit(0)


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
print(f'Number of bursts in {src_file} is {nBursts}')
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

print(f' Annotation_dir before IW_gps is: {Annotation_dir}')
print(f' annotation_file before IW_gps is: {annotation_file}')
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
#bursts_to_plot     =list(range(1,nBursts+1))
bursts_to_plot     =[9]
#az_fraction    = 1
#range_fraction = 1
#plotBursts(img, gridPoints, burst_list, az_fraction, range_fraction)

border = 25     # Put border of 20 pixels around ship chips => total size of chip is  (clearance+border)*2 = 140x140
"""
#for burst in bursts_to_plot:
#for i in range (0, nBursts):
    burst_boxes = swath_bounding_boxes[np.array(burst_indices)== burst-1 ]
    print(f' Ship Bounding Boxes for burst index={i} has shape: {burst_boxes.shape}')
    print(f'swath_bounding_boxes  shape :{swath_bounding_boxes.shape} : {swath_bounding_boxes}')

 #   plotBurstsII(img, bursts_to_plot, burst_boundaries, swath_bounding_boxes, burst_indices)
    border = 25     # Put border of 20 pixels around ship chips => total size of chip is  (clearance+border)*2 = 140x140
    print(f' Before plotBurstsIII - following values ....')
    print(f' bursts_to_plot is : {bursts_to_plot}   File is {SLC_fnames[swath_of_interest_index]} ....')
    print(f' merged burst_boundares for burst #{bursts_to_plot} = {burst_boundaries}')
"""

"""
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
"""

###############################################################################################
# test get_ship_chips

all_burst_chips,b,c,d,e = get_chips_and_masks(img, bursts_to_plot, burst_boundaries, swath_bounding_boxes, burst_indices, border)
# return chips_from_all_bursts, ship_loc_img, binary_image_ground_truth, ship_chips, binary_ship_chips
