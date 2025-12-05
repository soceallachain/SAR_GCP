#!/usr/bin/python3


# HTTP requests
import requests
# JSON parser
import json

# XML parser
import xml.etree.ElementTree as ET

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

path ="/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/"
# If set overwrite all existing SAR data overwritten else only overwrite particular mode being acquired
OVER_WRITE_ALL = False
#####################################################################
## Need ACCESS_TOKEN to retrieve data
## Need credentials.py file to contain USER and PASSWORD information
#

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



#####################################################################
## Parse credentials.INI file for username and password to pass to
## get_access_token
#
import configparser
config = configparser.ConfigParser()
config.read('credentials.INI')
username = config.get('Data', 'username')
password = config.get('Data', 'password')
#print("USERNAME: %s\nPASSWORD: %s\n"%(username, password))
access_token = get_access_token(username, password)
print(access_token)

######################################################################



## Get search periods
## Parse filename of training set to find start and end times, add 10s either side for search query as GRD data is shorter than SLC which is shorter than RAW
##
from datetime import datetime, timedelta

#file_name_list=["S1A_IW_GRDH_1SDV_20200914T075756_20200914T075825_034352_03FE84_4B93", "S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"]
#file_name_list=["S1A_IW_GRDH_1SDV_20200914T075756_20200914T075825_034352_03FE84_4B93"]
#file_name_list=["S1B_S3_RAW__0SDV_20200615T162409_20200615T162435_022046_029D76_F3E6"]
#file_name_list=["S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"]
file_name_list=["S1A_S6_RAW__0SDV_20210226T214354_20210226T214424_036766_045287_FD68"] # Sao Paolo 

## Check if not using GRDH root : then must replace '__' with '_' so that fields line up
for fname in file_name_list:
   fname            = fname.replace('__','_')
   fields           = fname.split('_')
   satellite_name   = fields[0]
   GRD_crc_check    = fields[8] 
   start_date       = fields[4].replace('T','')
   search_date_time = datetime.strptime(start_date,'%Y%m%d%H%M%S')
   search_date_time = search_date_time + timedelta(seconds = -10)
   start_date_time  = search_date_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
   end_date         = fields[5].replace('T','')
   search_date_time = datetime.strptime(end_date,'%Y%m%d%H%M%S')
   search_date_time = search_date_time + timedelta(seconds = 10)
   end_date_time    = search_date_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
   
   search_period_start = start_date_time
   search_period_end   = end_date_time
   
   #############################################################################
   # Generate query
   # Base URL of the product catalogue
   catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
   collection_name     = "SENTINEL-1"
   product_type        = fields[0]
   
  # search for all RAW, SLC and GRD file associated with fname
   search_query = f"{catalogue_odata_url}/Products?$filter=Collection/Name eq '{collection_name}' and (contains(Name,'{satellite_name}')) and  (contains(Name, 'RAW') or contains(Name, 'SLC') or (contains(Name, 'GRD') and contains(Name, '{GRD_crc_check}'))) and (contains(Name, 'SAFE')) and ContentDate/Start gt {search_period_start} and ContentDate/End lt {search_period_end}"
   print(f"""\n{search_query.replace(' ', "%20")}\n""")
   response = requests.get(search_query).json()
   result = pd.DataFrame.from_dict(response["value"])
   pd.set_option('display.max_colwidth', None)
   
   # print first 3 results
   print(result.head(8))
   
   number_files_found=len(result)
   if number_files_found ==3:
     
      for i in range (0,3):
         name = result.loc[i].at['Name'].split('_')
         type = name[2]
         if type == 'RAW':
            RAW_PUUID = result.loc[i].at['Id']
         elif type == 'SLC':
            SLC_PUUID = result.loc[i].at['Id']
         elif type == 'GRDH':
            GRD_PUUID = result.loc[i].at['Id']
         else:
            print('WARNING: No RAW or SLC or GRD file found for %s',fname)
   elif number_files_found <3:
      print('WARNING: Less than 3 files found for %s', fname)
   else:
      print('WARNING: More than 3 files found for ', fname)
   
   ######################################################################
   ## Have PUUID for RAW, SLC & GRD files
   #####################################################################
   #
   ## Create directories to store Copernicus data 
   ## If already exists then delete directory and contents
   #
   #
   # mode_list={'SLC', 'RAW','GRD'}
   mode_list={'RAW'}

   for mode in mode_list:
      mode_path = path+fname+'/'+mode
      print(f"{mode} path is : {mode_path}")
      
      if os.path.exists(path+fname) and OVER_WRITE_ALL:
         shutil.rmtree(path+fname)
         print('Dir removed:', path+fname)
         os.mkdir(os.path.join(path,fname))
         print(' New directory created:', path+fname)
         os.mkdir(os.path.join(mode_path))
         print(' New directory created:', path+fname)
      elif os.path.exists(path+fname):
         #Check if mode directory exists and if so remove it and create new one
         if os.path.exists(mode_path):
            shutil.rmtree(mode_path)
            print('Dir removed:', mode_path)
         os.mkdir(os.path.join(mode_path))
         print(' New directory created:', mode_path)
      else:
         # Path does not exits to filename root so create it and the mode directory
         os.mkdir(os.path.join(path,fname))
         print(' New directory created:', path+fname)
         os.mkdir(os.path.join(mode_path))
         print(' New directory created:', mode_path)
             
            
   # Get UUID based on mode
   if mode == 'SLC':
      mode_UUID = SLC_PUUID
   elif mode == 'RAW':
      mode_UUID = RAW_PUUID
   else:
      mode_UUID = GRD_PUUID

   print(f"mode is {mode} and mode_UUID = {mode_UUID}")
   
   ## Get 'mode' data for fname using product id : mode_UUID
   url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({mode_UUID})/$value"
   print(url)
   headers = {"Authorization": f"Bearer {access_token}"}

   session = requests.Session()
   session.headers.update(headers)
   response = session.get(url, headers=headers, stream=True)
   mode_data_file=mode_path+'/product.zip'
   print(mode_data_file)

   with open(mode_data_file, "wb") as file:
       for chunk in response.iter_content(chunk_size=8192):
           if chunk:
               file.write(chunk)


   
   

if os.path.exists(mode_data_file):
   print(f"{mode_data_file} written")

print('*******************************Complete*********************************************\n')


   
