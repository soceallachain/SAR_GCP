#!/usr/bin/python3

## getSARData.py

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
#file_name_list=["S1A_S3_RAW__0SDV_20230518T213602_20230518T213627_048953_05D835_4B93"]
#file_name_list=["S1A_IW_GRDH_1SDV_20200421T051412_20200421T051437_032221_03BA16_FC53"]
#file_name_list=["S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"]
#file_name_list=["S1A_S6_RAW__0SDV_20210226T214354_20210226T214424_036766_045287_FD68"]
#file_name_list=["S1A_IW_GRDH_1SDV_20240107T063927_20240107T063952_051997_06488E_03CA"]    # Dublin



#file_name_list=["S1A_IW_SLC__1SDV_20201116T052156_20201116T052226_035269_041E82_FEF7","S1A_IW_SLC__1SDV_20200308T171749_20200308T171822_031587_03A3B9_1334", "S1A_IW_SLC__1SDV_20200213T171749_20200213T171822_031237_0397A3_1E6E",
#                "S1A_IW_SLC__1SDV_20200718T171755_20200718T171828_033512_03E220_27C9", "S1A_IW_SLC__1SDV_20200917T174439_20200917T174506_034402_040041_4D50"]


#file_name_list=["S1B_IW_GRDH_1SDV_20200722T062306_20200722T062331_022580_02ADAC_78E8","S1A_IW_GRDH_1SDV_20200116T051412_20200116T051437_030821_038933_5CD5","S1A_IW_GRDH_1SDV_20200519T171753_20200519T171824_032637_03C7B2_1F74"]

#file_name_list=["S1A_IW_SLC__1SDV_20200326T052956_20200326T053025_031842_03ACB7_F199",
#file_name_list=["S1A_IW_SLC__1SDV_20200328T051410_20200328T051438_031871_03ADC3_1ABD",
#                "S1B_IW_SLC__1SDV_20201119T062308_20201119T062335_024330_02E422_7A6F",
file_name_list=["S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B"]
#                "S1A_IW_SLC__1SDV_20200409T051410_20200409T051438_032046_03B3F3_A8A7",
#                "S1B_IW_SLC__1SDV_20200305T063110_20200305T063137_020553_026F28_2D42",
#                "S1A_IW_SLC__1SDV_20200304T051410_20200304T051437_031521_03A18B_9984",
#                "S1B_IW_SLC__1SDV_20200901T063119_20200901T063146_023178_02C029_E7F9",
#                "S1B_IW_SLC__1SDV_20201026T062308_20201026T062335_023980_02D93F_8994",
#                "S1A_IW_SLC__1SDV_20200309T175707_20200309T175734_031602_03A449_18A0",
#                "S1B_IW_SLC__1SDV_20200523T062301_20200523T062328_021705_029328_22E0",
#                "S1B_IW_SLC__1SDV_20200628T062303_20200628T062330_022230_02A309_E99E",
#                "S1A_IW_SLC__1SDV_20200302T052955_20200302T053025_031492_03A078_1FD4",
#                "S1B_IW_SLC__1SDV_20200826T182759_20200826T182825_023098_02BDB5_656B"]

#file_name_list=["S1A_IW_GRDH_1SDV_20200126T052956_20200126T053025_030967_038E50_B43E"]   
#file_name_list=["S1A_IW_GRDH_1SDV_20201018T051421_20201018T051446_034846_040FF3_76A1"]     #128 High Confidence Ships, 89 Medium, 127 Not Vessel
#file_name_list=["S1B_IW_GRDH_1SDV_20200715T063117_20200715T063142_022478_02AA98_34D9"]      # Cornwall scene, SARFish training 
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
   search_query = (f"{catalogue_odata_url}/Products?$filter=Collection/Name
                   eq '{collection_name}' 
                   and (contains(Name,'{satellite_name}')) 
                   and  (contains(Name, 'RAW') or contains(Name, 'SLC') 
                   or (contains(Name, 'GRD') and contains(Name, '{GRD_crc_check}'))) 
                   and (contains(Name, 'SAFE')) 
                   and ContentDate/Start gt {search_period_start} 
                   and ContentDate/End lt {search_period_end}"
                   )
   print(f"""\n{search_query.replace(' ', "%20")}\n""")
   response = requests.get(search_query).json()
   result = pd.DataFrame.from_dict(response["value"])
   pd.set_option('display.max_colwidth', None)
   
   # print first 3 results
   print(result.head(8))
   
   files_not_found = 'RAW SLC GRDH'
   number_files_found=len(result)
   for i in range (0, number_files_found):
       name = result.loc[i].at['Name'].split('_')
       type = name[2]
       if type == 'RAW':
           RAW_PUUID = result.loc[i].at['Id']
           files_not_found = files_not_found.replace ('RAW','')
       elif type == 'SLC':
           SLC_PUUID = result.loc[i].at['Id']
           files_not_found = files_not_found.replace ('SLC','')
       elif type == 'GRDH':
           GRD_PUUID = result.loc[i].at['Id']
           files_not_found = files_not_found.replace ('GRDH','')
       else:
           print('WARNING: No RAW or SLC or GRD file found for %s',fname)
        
   print('Files not found: %s',files_not_found)
   
   ######################################################################
   ## Have PUUID for RAW, SLC & GRD files
   #####################################################################
   #
   ## Create directories to store Copernicus data 
   ## If already exists then delete directory and contents
   #
   #
   #mode_list={'SLC', 'RAW','GRD'}
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
         print(' New directory created:', mode_path)
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

   ## Need to re-send access token after each product download as lose access otherwise
   access_token = get_access_token(username, password)
print('*******************************Complete*********************************************\n')


   
