# HTTP requests
import requests

# JSON parser
import json

# XML parser
import xml.etree.ElementTree as ET

# system modules
import os
import re
import sys
import random

# data manipulation
import pandas as pd
import numpy as np

# base URL of the product catalogue
catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"

# search parameters
#max_cloud_cover = 1
#aoi = "POLYGON((20.888443 52.169721,21.124649 52.169721,21.124649 52.271099,20.888443 52.271099,20.888443 52.169721))"
collection_name = "SENTINEL-1"
product_type = "S2MSI1C"
## Get search periods
## Parse filename of training set to find start and end times, add 10s either side and for search query as GRD data is shorter than SLC which is shorter than RAW
##

fname ="S1B_IW_GRDH_1SDV_20200709T054005_20200709T054030_022390_02A7EC_68FE"
fields = fname.split('_')
#start_times =  fields[4].split('T')
#end_times   =  fields[5].split('T')

start_date = fields[4].replace('T','')

from datetime import datetime
search_date_time  = datetime.strptime(start_date,'%Y%m%d%H%M%S')

d = search_date_time.strftime('%Y-%m-%d %H:%M:%S')
d=d.replace(' ','T')
d=d+'.000Z'

print(d)


search_period_start = "2023-06-01T00:00:00.000Z"
search_period_end = "2023-06-10T00:00:00.000Z"
#search_query = f"{catalogue_odata_url}/Products?$filter=Collection/Name eq '{collection_name}' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and ContentDate/Start gt {search_period_start} and ContentDate/Start lt {search_period_end}"

search_query = f"{catalogue_odata_url}/Products?$filter=Collection/Name eq '{collection_name}' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') and ContentDate/Start gt {search_period_start} and ContentDate/Start lt {search_period_end}"

print(f"""\n{search_query.replace(' ', "%20")}\n""")


response = requests.get(search_query).json()
result = pd.DataFrame.from_dict(response["value"])

#print(response)

# print first 3 results
print(result.head(8))


