#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 12:58:39 2025

@author: joe
"""



import pandas as pd
import numpy as np
import os


source = 'ext'
Dataset = 'Train'
#  Read CSV
csv_filename =f'{Dataset}_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m_Statistics.csv'
test_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results'

csv_filename1 =f'{Dataset}_RAW_IW2_ship_list_for_sea_based_bursts_with_only_HM_ships_gt100m.csv'
#csv_filename =f'Bursts_Ships_To_Test_Offset.csv'

csv_file = os.path.join(test_dir, csv_filename)
csv_file1 =os.path.join(test_dir, csv_filename1)
df = pd.read_csv(csv_file)
df1= pd.read_csv(csv_file1)

burst_id =[]
crc_burst = []
burst_names = df1['Burst_Name']
for i in range(0,len(burst_names)):
    b            = burst_names[i].split('_#')[0]
    burst_number = int(b.split('#')[1])
    burst_id.append(b.split('_RAW')[0])
    crc_burst.append(f"{burst_id[i].split('_')[9]}_burst#{burst_number}")



names_in_csv = df['Burst Name']
df1['Burst Name']=crc_burst

# Create mapping from 'Burst Name' to 'Burst_Name'
burst_map = df1.set_index('Burst Name')['Burst_Name'].to_dict()

# Use list comprehension to look up each name in names
name_ID = [burst_map.get(name, None) for name in names_in_csv]
df['Burst_Name']=name_ID


csv_outputfile = os.path.join(test_dir,csv_filename.replace('.csv','_with_SLC_ID.csv'))
df.to_csv(csv_outputfile)