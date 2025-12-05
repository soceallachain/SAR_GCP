import numpy as np
import requests
import json

# XML parse
import xml.etree.ElementTree as ET
import urllib.parse

from datetime import datetime, timedelta

# system modules
import os
import io
import re
import sys
import random
import shutil
# data manipulation
import pandas as pd
import numpy as np

import configparser

from pathlib import Path

from geopy.distance import great_circle, geodesic
from geopy.point import Point



#####################################################################
## Need ACCESS_TOKEN to retrieve data
## Need credentials.py file to contain USER and PASSWORD information
#
credentials_file = '/home/joe/Misc/credentials.INI'
## Function to generate access_token for Copernicis Hub
def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]
    response = requests.get(search_query).json()


def calculate_roi_vertices(gps_coords, roi_size_km=100):
    """
    Calculates the four corner vertices of a square ROI centered on the mean
    of a set of input GPS coordinates.

    The ROI size is defined by roi_size_km (e.g., 100 km x 100 km).
    
    :param gps_coords: A list of coordinates in [latitude, longitude] format.
    
    Note: gps_coords must be a list of lists of form :
          [[lat_1,long_1],[lat_2,long_2],..[lat_n,long_n]]
          so if only 1 gops location being used must be in form:
          [[lat_1,long_1]] NOT [lat_1, long_1]
          
          This is to enable multiple gops locations being used and 
          defining ROI around their mean
    :param roi_size_km: The side length of the square ROI in kilometers.
    :return: A list of four [latitude, longitude] vertices defining the ROI.
             [SW, NW, NE, SE]
    """
    if not gps_coords:
        return []

    # 1. Calculate the Geographic Center
    # Convert list of lists/tuples to a NumPy array for easier math
   # Check the number of coordinates
    if len(gps_coords) == 1:
        # If only one point, use it directly as the center
        center_lat = gps_coords[0][0] # Assuming format is [latitude, longitude]
        center_lon = gps_coords[0][1]
    else:
        # Standard procedure for two or more points: calculate the mean
        coords_np = np.array(gps_coords)
        
        # Calculate the mean latitude and mean longitude
        # NOTE: For very large areas, a true centroid calculation would be better,
        # but the mean is an excellent and common approximation for small areas.
        center_lat = np.mean(coords_np[:, 0])
        center_lon = np.mean(coords_np[:, 1])

    # Create the geopy Point object for subsequent calculations
    center_point = Point(center_lat, center_lon)

    # 2. Calculate Half-Distance and Half-Diagonal
    # Distance from center to edge (radius)
    half_side = roi_size_km / 2.0
    
    # Distance from center to corner (diagonal radius)
    # The half-diagonal is sqrt(half_side^2 + half_side^2)
    half_diagonal = np.sqrt(2) * half_side

    # 3. Calculate the Four Vertices using Geodetic Formulas
    
    # Geopy's VincentyDistance is highly accurate for defining points
    # based on a starting point, a bearing (azimuth), and a distance.
    
    # Bearings for corners (in degrees):
    # SW: 225° (Southwest)
    # NW: 315° (Northwest)
    # NE: 45° (Northeast)
    # SE: 135° (Southeast)

    # Note: We use the half_diagonal distance for the corner points.
    
    # --- SW Corner (Most Western-most and Southern-most) ---
    # Bearing: 225 degrees (Southwest)
    sw_point = geodesic(kilometers=half_diagonal).destination(center_point, 225)
    
    # --- NW Corner (Most Western-most and Northern-most) ---
    # Bearing: 315 degrees (Northwest)
    nw_point = geodesic(kilometers=half_diagonal).destination(center_point, 315)
    
    # --- NE Corner (Most Eastern-most and Northern-most) ---
    # Bearing: 45 degrees (Northeast)
    ne_point = geodesic(kilometers=half_diagonal).destination(center_point, 45)
    
    # --- SE Corner (Most Eastern-most and Southern-most) ---
    # Bearing: 135 degrees (Southeast)
    se_point = geodesic(kilometers=half_diagonal).destination(center_point, 135)

    # 4. Format the Output
    # Return the coordinates in the requested order: [SW, NW, NE, SE]
    vertices = [
        [sw_point.latitude, sw_point.longitude],
        [nw_point.latitude, nw_point.longitude],
        [ne_point.latitude, ne_point.longitude],
        [se_point.latitude, se_point.longitude]
    ]

    return vertices




def generate_roi_match_query(roi):
    # Code takes an ROI generated by calculate_roi_vertices() that has been 
    # converted to [long, lat] format and generates a OData query to find
    # datatakes that include all 4 vertices not just intersecting some fraction 
    # of the ROI
    #
    geo_filter=''
    geo_filter_start =f"(OData.CSC.Intersects(area=geography'SRID=4326;POINT"
    for vertex in roi:
        #print(f'{vertex[0]} {vertex[1]}')
        coords=f"({str(vertex[0])} {str(vertex[1])})')"
        geo_filter=f'{geo_filter}{geo_filter_start}{coords}) and '
    #Remove final 'and'. Ensure brackets around each sub-clause
    geo_filter=f'{geo_filter}-'
    geo_filter=geo_filter.replace(' and -','')
    #Ensure brackets aropund complete query
    return f'({geo_filter})'



def generate_cdse_odata_query(satellite, location, size_of_roi,
                              orb_node, mode,
                              product_types,
                              start_date, end_date,
                              start_time, end_time 
                              ):
    """
    Generates a Copernicus Data Space Ecosystem (CDSE) OData API query string 
    to find Sentinel  products that **completely contain** the defined ROI 
    within the specified date range.

    :param location gps co-ordinates of location of interest.
    :param size_of_roi  - size of region around location to be included
    :param start_date: Start date for the search (ISO format, YYYY-MM-DD).
    :param end_date: End date for the search (ISO format, YYYY-MM-DD).
    :param product_types: List of Sentinel-1 product types to search for (e.g., ["RAW", "SLC"]).
    :return: The complete, URL-encoded OData API query URL.
    """
    
    
    #1. Generate geometry filter for area match
    
    roi_vertices= calculate_roi_vertices(location, size_of_roi)
    
    if len(roi_vertices) != 4:
        raise ValueError("roi_vertices must contain exactly 4 points.")

    lon_lat_coords = [
        (v[1], v[0]) for v in roi_vertices
    ]
    area_match_filter = generate_roi_match_query(lon_lat_coords)
        
    
    # 2. Timing constraint filter
    
    date_time = f'{start_date}T{start_time}'
    search_date_time = datetime.strptime(date_time,'%Y-%m-%dT%H:%M:%S')
    start_date_time  = search_date_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    date_time = f'{end_date}T{end_time}'
    search_date_time = datetime.strptime(date_time,'%Y-%m-%dT%H:%M:%S')
    end_date_time  = search_date_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    date_filter = (
    f"(ContentDate/Start gt {start_date_time} and "
    f"ContentDate/End lt {end_date_time})"
    )


    #3. Mode filter for IW, SM or EW for S1 or others for Sentinel-2
    
    if mode not in ['IW', 'SM', 'EW']:
        print(f'Error: Unknown mode - must be IW, SM or EW ')
        print(f'Ignoring mode in search criteria')
        mode_filter=''
    else:
        mode_filter =(f"contains(Name, '{mode}')"
                     )




    # Satellite filter - filter for Collection and satellite based on satellite
    # argument
    satellite=satellite.upper()
    if satellite[0] =='S':
        collection_name=f'SENTINEL-{satellite[1]}'
        satellite_filter = (f"Collection/Name eq '{collection_name}'"
                            f" and (contains(Name,'{satellite}'))"
                            )
    else:
        print(f'Error:Unknown Satellite: {satellite} - Sentinel constellations Only')
        print('Exiting .....')
        sys.exit(0)
    
        
    #Orbital direction filter
    if     orb_node == 'ASCENDING' or orb_node =='DESCENDING':
        orbital_dir_filter = (
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
            f"'orbitDirection' and att/OData.CSC.StringAttribute/Value"
            f" eq '{orb_node}')"
                          )
    else:
        print(f"Orbital Node specification error - must be 'ASC' or 'DESC'")
        print('Ignoring orbital node in search criteria')
        orbital_dir_filter=''
          
        
    #product type filters
    product_filter=''
    if len(product_types)==0:
        print('No products specified: Ignored in search criteria')
    else:
        for product in product_types:
            #product_filter = f"{product_filter}(contains(Name, '{product}') or "
            product_filter = f"{product_filter}contains(Name, '{product}') or "
        product_filter=f'{product_filter}_'
        product_filter=product_filter.replace(' or _',')')
        product_filter="("+product_filter
    

    # 3. Combine all parts into the OData $filter query
    filter_value = (
       # f"{satellite_filter} and "
       # f"({product_filter}) and "
       # f"(PlatformName eq 'S1') and "
       # f"(OData_Crs:startswith('S1_L1C_SLC__1S')) and "
       # f"(contains(Name,'S1A'))"  
       # f"{date_filter} and "
       # f"{area_match_filter} and "
       # f"{mode_filter} and "
       # f"{orbital_dir_filter} "
       f"{satellite_filter} and "
       f"{product_filter} and "
       f"{date_filter} and "
       f"{area_match_filter} and "
       f"{mode_filter} and "
       f"{orbital_dir_filter}"
    )

    # 4. Construct the final query URL
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    
    # Use urllib.parse.quote_plus to safely encode the filter value
    encoded_filter = urllib.parse.quote_plus(filter_value)
    
    # The OData query structure uses $filter and $top
    num_results = 30
    final_query = f"{base_url}?$filter={encoded_filter}&$orderby=ContentDate/Start&$top={num_results}"
    
    return final_query



# Function to download SAFE file from Copernicus Hub
# Used UUID returned from generate_cdse_odata_query results
# Saves file to save_path
def get_copernicus_file_using_UUID(uuid, save_path, credentials_file):
    
    #Get access token
    config = configparser.ConfigParser()
    config.read(credentials_file)
    username = config.get('Data', 'username')
    password = config.get('Data', 'password')
    #print("USERNAME: %s\nPASSWORD: %s\n"%(username, password))
    access_token = get_access_token(username, password)
    
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({uuid})/$value"
    #print(url)
    headers = {"Authorization": f"Bearer {access_token}"}
 
    session   = requests.Session()
    session.headers.update(headers)
    response  = session.get(url, headers=headers, stream=True)
    data_file = os.path.join(save_path,'product.zip')
 
    with open(data_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
 
 
    if os.path.exists(data_file):
       print(f"{data_file} successfully written")
       #Unzip file
       return 1
       
    else:
       print('Error: {data_file} processing error')
       return 0
      

    ## Need to re-send access token after each product download a
    ## as lose access otherwise 
    ## access_token = get_access_token(username, password)

import zipfile
import os

def unzip_file(file_path, output_dir):
    """Handles .zip file extraction."""
    print(f"\n--- Processing: {file_path} ---")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ Successfully unzipped to '{output_dir}'")
    except FileNotFoundError:
        print(f"❌ Error: File not found.")
    except zipfile.BadZipFile:
        print(f"❌ Error: Not a valid ZIP file.")





# Function to: 1. Extract Copernicus zip file to specified directory
#              2. Move contents from <dest_dir>/product/ to <dest_dir>/
#              3. Remove <dest_dir>/product dir
#              4. Verify presence of SAFE dir and all files for SLC and RAW
def extract_and_verify_copernicus_files(zip_file):
    
    #Expect zip_file to be absolute path to zip file
    if os.path.isfile(zip_file):
        zip_file_path = os.path.dirname(zip_file)
        #zip_file_name = os.path.basename(zip_file)
        # Temporary archive created to unzip zip file into. Has same name as 
        # stem of zip file name  e.g. 'name.zip' -> archive_name = 'name'
        archive_path  = os.path.splitext(zip_file)[0]
        
        # Python unzip unzips SAFE dir directly to target dir UNLIKE Linux
        # Extract which unzips to a dir with the same name as the zip file
        # Consequently am making behaviour here mimic Linux and extracting to
        # /product dir
        unzip_file(zip_file, archive_path)
        
        if os.path.exists(archive_path):
            safe_dir_name = os.listdir(archive_path)[0]
            safe_dir  = os.path.join(archive_path, safe_dir_name)
            
            if os.path.exists(safe_dir):
                print(f'SAFE dir : {safe_dir} exists')
                #SAFE dir found get type - RAW, SLC or GRD or other
                data_acq_type = safe_dir_name.split('_')[2]
                
                # Set file_specifications for each 
                SLC_specifications=[[os.path.join(safe_dir,'measurement'),
                                     '*.tiff',6
                                     ],
                                    [os.path.join(safe_dir,'annotation'), 
                                     '*.xml',6
                                     ]
                                    ]
                RAW_specifications =[[f'{safe_dir}', '*.dat', 6]]
                GRD_specifications =[[os.path.join(safe_dir,'measurement'),
                                      '*.tiff',2
                                      ],
                                     [os.path.join(safe_dir,'annotation'), 
                                      '*.xml',2
                                      ]
                                     ]
                if data_acq_type == 'RAW':
                    #check for 6 off.dat files in safe_dir
                    r=verify_directory_contents(RAW_specifications)
                    
                elif data_acq_type == 'SLC':
                    #check for 6 xml and 6 .tiff files in measurement and annotaton
                    r=verify_directory_contents(SLC_specifications)

                elif data_acq_type =='GRD':
                    r=verify_directory_contents(GRD_specifications)
                else:
                    print(f'Unknown file type for verifying Copenicus file')
                    r=0
                if r:
                    print(f'SUCCESS: {safe_dir_name} successfully extracted')
            else:
                print(f'Error: SAFE dir {safe_dir} not found')
        else:
            print(f'Unzipped product directory: {archive_path}'
                  f'not found'
                  )
    # Move Safe file into temp dir, remove /product dir and move safe file 
    # back to 
    temp_dir = './dummy'
    os.makedirs(temp_dir, exist_ok=True)
    try:
        shutil.move(safe_dir, temp_dir)
    except Exception as e:
        print(f'Error moving SAFE dir: {safe_dir}')
    # Remove product directory - should be empty
    try:
        os.rmdir(archive_path)
        print(f' Product directory removed')
    except Exception as e:
        print(f'Error {archive_path} directory not empty or not found')
        
    try:
        shutil.move(f'{temp_dir}/{safe_dir_name}', zip_file_path)
        os.remove(zip_file)
        # Remove temporary directory
        os.rmdir(temp_dir)
        
    except Exception as e:
        print(f'Error moving SAFE dir: {safe_dir_name} to {zip_file_path}')
        
                
                                 
                
            
        


# Function to verify that given directories exist and have the expected numbers
# of files of particular file types.
# Specifically to test that Copernicus RAW and SLC files are correctly unzipped
# 
def verify_directory_contents(dir_specifications):
    for dir_info in dir_specifications:
        test_dir = Path(dir_info[0])
        file_type = dir_info[1]
        expected_number_files = dir_info[2]
        
        if os.path.exists(test_dir):
            files = list(test_dir.glob(f'{file_type}'))
            if len(files) == expected_number_files:
                print(f'{test_dir} found and contains '
                      f'{expected_number_files} expected '
                      f'{file_type} files')
                success = 1
            else:
                print(f'Error: {test_dir} found but contains only '
                      f'{len(files)} of the {expected_number_files} '
                      f'expected files'
                      )
                success = 0
        else:
            print(f' Error: {test_dir} not found!')
            success = 0   
                         
    return success

    
def acquire_extract_verify_copernicus_files(uuid, save_path, credentials_file):

    status = get_copernicus_file_using_UUID(uuid, save_path, credentials_file)
    if status:
        #extract zip file to /save_path
        #verify all files present
        #move contents of unzipped dir to /save_path
        #remove unzipped top directory
        zipped_file = os.path.join(save_path,'product.zip')
        if os.path.isfile(zipped_file):
            extract_and_verify_copernicus_files(zipped_file)
        else:
            print(f'Error : Zip file {zipped_file} not found!')
            
    else:
        print(f'Error: File ID {uuid}:- Acquisition Failed!')
        
        
    

    

# Function to return orbital node of S1 data take
# Reads SLC annotation file and returns orbital node specified in
# xml location   <pass>xxxx</pass>
def orbital_node_is(slc_annotation_file):
    
    import xml.etree.ElementTree as ET
    
    # Load the XML file
    tree = ET.parse(slc_annotation_file)
    root = tree.getroot()
    
    # Find the orbital_node element
    orbital_node_element = root.find('.//pass')
    
    # Extract the value if the element exists
    if orbital_node_element is not None:
        orbital_node_value = orbital_node_element.text
        return orbital_node_value
    else:
        return 0

    
# Function to return time difference between 2 date-time string variables with 
# format YYYYMMDDTHHMMSS   
def time_diff_seconds(dt1_str, dt2_str):
    # Define the format
    fmt = "%Y%m%dT%H%M%S" #NOTE: %Y ensures 4-digit year (%y would be 2-digit)
    # Parse the strings into datetime objects
    dt1 = datetime.strptime(dt1_str, fmt)
    dt2 = datetime.strptime(dt2_str, fmt)
    # Calculate the difference in seconds
    diff = abs((dt1 - dt2).total_seconds())
    return diff





def match_and_collect_ids(raw_df, slc_df, time_threshold=5):
    # Define the datetime format:- NOTE %Y means 4-digit year;%y means 2-digit
    fmt = "%Y%m%dT%H%M%S"
    
    # Prepare the result list
    matched_ids = []
    
    # Iterate over each row in the RAW dataframe
    for _, raw_row in raw_df.iterrows():
        raw_start = datetime.strptime(raw_row['Start Time'], fmt)
        raw_id = raw_row['Id']
        found_match = False
        
        # Iterate over each row in the SLC dataframe
        for _, slc_row in slc_df.iterrows():
            slc_start = datetime.strptime(slc_row['Start Time'], fmt)
            slc_id = slc_row['Id']
            
            # Calculate the absolute difference in seconds
            diff_seconds = abs((raw_start - slc_start).total_seconds())
            
            # Check if the difference is within the threshold
            if diff_seconds < time_threshold:
                # Append both IDs to the result list
                matched_ids.append((raw_id, slc_id))
                found_match = True
                break  # Stop looking for other matches for this RAW row
        
        # If no match was found, you could optionally append (raw_id, None) or skip
        if not found_match:
            print(f"Warning: No match found for {raw_row['Name']}")
    
    return matched_ids

# Example usage:
# raw_df = pd.DataFrame({'Id': ['id1', 'id2'], 'Type': ['RAW', 'RAW'], 'Start Time': ['20250913T181443', '20250913T181515']})
# slc_df = pd.DataFrame({'Id': ['id3', 'id4'], 'Type': ['SLC', 'SLC'], 'Start Time': ['20250913T181446', '20250913T181513']})
# result = match_and_collect_ids(raw_df, slc_df)
# print(result)

 
"""
# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Coordinates for a hypothetical study area near Dublin, Ireland
    # Format: [Latitude, Longitude]
    sample_coords = [
        [53.3498, -6.2603],   # Dublin City Center
        [53.4000, -6.1000],   # Northeast of city
        [53.2000, -6.3000],   # South of city
    ]

    roi_vertices = calculate_roi_vertices(sample_coords)
    
    if roi_vertices:
        print("--- 100 km x 100 km Region of Interest (ROI) ---")
        print(f"Centered on approximate mean: {np.mean(np.array(sample_coords), axis=0)}")
        print(f"ROI Size: 100 km x 100 km\n")
        
        print(f"1. SW Corner (West-South): {roi_vertices[0]}")
        print(f"2. NW Corner (West-North): {roi_vertices[1]}")
        print(f"3. NE Corner (East-North): {roi_vertices[2]}")
        print(f"4. SE Corner (East-South): {roi_vertices[3]}")

        # Verification (Optional): Check the distance between opposite corners
        # This should be approximately the diagonal distance (sqrt(100^2 + 100^2) = 141.42 km)
        sw_lat, sw_lon = roi_vertices[0]
        ne_lat, ne_lon = roi_vertices[2]
        diagonal_distance = great_circle((sw_lat, sw_lon), (ne_lat, ne_lon)).km
        
        print(f"\nVerification (SW to NE Diagonal): {diagonal_distance:.2f} km (Target: 141.42 km)")
"""
