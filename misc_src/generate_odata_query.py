#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 09:41:45 2025

@author: joe
Script to test acquistition of Sentinel1 data from Copernicus Hub
using query comannds
"""

from geo_module import *






# HTTP requests
import requests
# JSON parser





#####################################################################
## Parse credentials.INI file for username and password to pass to
## get_access_token
#
import configparser
config = configparser.ConfigParser()
#config.read('credentials.INI')
config.read('/home/joe/Misc/credentials.INI')
username = config.get('Data', 'username')
password = config.get('Data', 'password')
#print("USERNAME: %s\nPASSWORD: %s\n"%(username, password))
access_token = get_access_token(username, password)
print(access_token)

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
   
    location = [[52.4, -5.8]]
    satellite = 'S1A'
    product_types =['RAW', 'SLC']
    #orb_node = 'ASCENDING'
    orb_node = 'DESCENDING'
    mode = 'IW'
    roi_size = 50
    search_start_date = "2025-09-12"
    search_end_date   = "2025-09-25"
    start_time ='00:00:00'
    end_time  ='23:59:59'

    search_query = generate_cdse_odata_query(
         satellite, location, roi_size,
         orb_node, mode, product_types,
         search_start_date, search_end_date,
         start_time, end_time
    )
    
    
    print("\nGenerated Copernicus Data Space Ecosystem (CDSE) OData Query:")
    print(search_query)
    print('*************************************/n')
    print(f"""\n{search_query.replace(' ', "%20")}\n""")
    response = requests.get(search_query).json()
    result = pd.DataFrame.from_dict(response["value"])
    pd.set_option('display.max_colwidth', None)
    
    # print first 3 results
    print(result.head(8))
    
    # Get type and start time for each file found
    new_r = result[['Id','Name']].copy()
    #new_r = r2[['Id','Name']].copy()
    new_r['Type'] = result['Name'].apply(lambda x: x.split('_')[2])
    new_r['Start Time'] = result['Name'].apply(lambda x: x.split('_')[5])
    
    #Check that each RAW file has associated SLC file that starts <5s after it
    raw_files = new_r[new_r['Type'] == 'RAW']
    slc_files = new_r[new_r['Type'] == 'SLC']
    #Pair RAW and SLC files
    matching_raw_slc_ids = match_and_collect_ids(raw_files, 
                                                 slc_files, 
                                                 time_threshold=5
                                                 )
    if orb_node=='ASCENDING':
        orb_node_path =f'Ascending_Node'
    else:
        orb_node_path = f'Descending_Node'
        
    raw_path = (f'/home/joe/Work/Projects/RSAR/Test/SampleData/'
                f'test_results/test_velocity_hypothesis/Ships/WBYeats/'
                f'{orb_node_path}/RAW/'
                )
    
    slc_path = (f'/home/joe/Work/Projects/RSAR/Test/SampleData/'
                f'test_results/test_velocity_hypothesis/Ships/WBYeats/'
                f'{orb_node_path}/SLC/'
                )
    #Check exist, create if don't
    os.makedirs(slc_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)
    
    for row in range(len(matching_raw_slc_ids)):
        dest_path = raw_path
        for col in range(len(matching_raw_slc_ids[row])):
            uuid = matching_raw_slc_ids[row][col]
            #get_copernicus_file_using_UUID(uuid, dest_path, credentials_file)
            acquire_extract_verify_copernicus_files(uuid, dest_path, credentials_file)
            dest_path = slc_path

    
 