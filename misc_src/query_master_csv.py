#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 17:11:01 2025

@author: joe
"""

import sys
import os
import re
import pandas as pd
from rawsarai_config import *
"""
dataset='Public'
sar_mode ='RAW'
iw_mode = 2

local =1
"""

if dataset == 'Public':
    label_file = 'SLC_Public_Labels.csv'
elif dataset == 'Train':
    label_file = 'SLC_train.csv'
elif dataset == 'Validation':
    label_file = 'SLC_validation.csv'
else:
    print(f'Issue with dataset - Check spelling')



if local ==1:
    raw_path  = local_raw_path
    slc_path  = local_slc_path
    save_path = local_save_path
    scene_mapping_file = f'{local_scene_mapping_path}/{label_file}'
else:
    raw_path  = nonlocal_raw_path
    slc_path  = nonlocal_slc_path
    save_path =  nonlocal_save_path
    scene_mapping_file = f'{nonlocal_scene_mapping_path}/{label_file}'


"""
if local ==1:
    raw_path  ='/home/joe/Work/Projects/RSAR/Test/SampleData/Public/raw_data'
    slc_path  ='/home/joe/Work/Projects/RSAR/Test/SampleData/Public/slc_data'
    #save_path ='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_get_bursts_for_HM_sea_only_ships' 
    save_path ='/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/t1' 
    scene_mapping_file = f"/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/{label_file}"
else:
    raw_path  = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Public/raw_data"
    slc_path  = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Public/slc_data"
    save_path = "/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/test_results/test_get_bursts_for_HM_sea_only_ships"
    scene_mapping_file = f'/mnt/smbshare/RAWSARAI_Data/Copernicus/Info/SARFishLabels/{label_file}'
"""

#save_path =''

#csv_path = '/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_get_bursts_for_HM_sea_only_ships'
csv_path = save_path
#Merge Ship and Non-Ship csv files

path=csv_path
ship_df_filename= f'{path}/SARFish_{sar_mode}_{dataset}_IW{iw_mode}_BurstInfoShipsPerBurstWithShipLocations.csv'
non_ship_df_filename =f'{path}/SARFish_{sar_mode}_{dataset}_IW{iw_mode}_BurstInfoShipsPerBurstWithNonShipLocations.csv'
all_ships_df_filename =f'{path}/{sar_mode}_{dataset}_IW{iw_mode}_AllBurstsShipAndNonShipInfo.csv'


if os.path.exists(ship_df_filename):
    ship_df = pd.read_csv(ship_df_filename)
else:
    print(f'Error: File {ship_df_filename} not found! Exiting .......')
    sys.exit(0)

if os.path.exists(non_ship_df_filename):
    non_ship_df = pd.read_csv(non_ship_df_filename)
else:
    print(f'Error: File {non_ship_df_filename} not found! Exiting .......')
    sys.exit(0)


merged_df = pd.concat([ship_df, non_ship_df], axis=0, ignore_index=True)
first_col = merged_df.columns[0]
sorted_df = merged_df.sort_values(by=first_col).reset_index(drop=True)
sorted_df.to_csv(f'{path}/{sar_mode}_{dataset}_IW{iw_mode}_AllBurstsShipAndNonShipInfo.csv')


#Open Master csv file for Public IW2
public_iw2_csv = f'{csv_path}/{sar_mode}_{dataset}_IW{iw_mode}_AllBurstsShipAndNonShipInfo.csv'
df_csv = pd.read_csv(public_iw2_csv)

#Determine number of bursts, ships amd non-ships in dataset
num_bursts    = len(df_csv[['Product_ID','Burst_Number']].drop_duplicates())
num_ships     = len(df_csv.loc[df_csv['Ship_Status']==True])
num_non_ships = len(df_csv.loc[df_csv['Ship_Status']==False])


# Number of SAR images 
sar_images= df_csv.drop_duplicates(
        subset=['Product_ID'],keep='first'
        )
num_sar_images = len(sar_images)

# Determine how many bursts are sea based
df_sea_based = df_csv.loc[df_csv['Sea_Status']==True]
num_sea_bursts = len(df_sea_based[['Product_ID',
                                   'Burst_Number']].drop_duplicates())

#Determine how many ships in sea based bursts
num_sea_ships     = len(df_sea_based.loc[df_sea_based['Ship_Status']==True])
num_sea_non_ships = len(df_sea_based.loc[df_sea_based['Ship_Status']==False])

#Determine how many sea based bursts have ships but no non-ships
cols_to_check = ['Product_ID', 'Burst_Number']


sea_based_ships     = df_sea_based.loc[df_sea_based['Ship_Status']==True]
sea_based_non_ships = df_sea_based.loc[df_sea_based['Ship_Status']==False]

# Bursts with labelled ships
sea_based_bursts_with_ships = sea_based_ships.drop_duplicates(
                              subset=cols_to_check, keep='first'
                            ).loc[lambda df: df['Number_Ships']>0]

# Bursts with no labelled ships
sea_based_bursts_without_ships = sea_based_ships.drop_duplicates(
                              subset=cols_to_check, keep='first'
                            ).loc[lambda df: df['Number_Ships']==0]


# Bursts with labelled non-ships/vessels
sea_based_bursts_with_non_ships = sea_based_non_ships.drop_duplicates(
                                           subset=cols_to_check, keep='first'
                            ).loc[lambda df: df['Number_Ships']>0]

# Bursts with no labelled non-ships/vessels
sea_based_bursts_without_non_ships = sea_based_non_ships.drop_duplicates(
                                           subset=cols_to_check, keep='first'
                            ).loc[lambda df: df['Number_Ships']==0]

merged_df = sea_based_bursts_with_ships.merge(sea_based_bursts_with_non_ships, 
                                               on=['Product_ID', 
                                                   'Burst_Number'], 
                                               how='left', indicator=True
                                              )
                                              
sea_based_bursts_with_ships_no_non_ships = merged_df[merged_df['_merge']
                                        =='left_only'].drop(columns=['_merge'])


#Get empty sea-based bursts;devoid of ships and non-ships
#Intersection of sea_based_bursts_without_ships & 
#                sea_based_bursts_without_non_ships

empty_sea_based_bursts = sea_based_bursts_without_non_ships.merge(
    sea_based_bursts_without_ships[['Product_ID', 'Burst_Number']],
    on=['Product_ID', 'Burst_Number'],
    how='inner'
)

empty_sea_based_bursts.to_csv(f'{csv_path}/{dataset}_{sar_mode}_IW{iw_mode}_'
                              f'empty_sea_based_bursts.csv')
num_empty_sea_based_bursts = len(empty_sea_based_bursts)


# All ships in sea based bursts with H or M confidence
# Note: Only ~20% of labelled ships have a labelled length
sea_based_HM_ships = df_sea_based.loc[
    (df_sea_based['Ship_Status'] == True) &
    (df_sea_based['Number_Ships'] > 0)    &
    (
       (df_sea_based['Confidence']=='HIGH') |
       (df_sea_based['Confidence']=='MEDIUM')
    )
]


# All ships in sea based bursts with H or M confidence and given length
# Note: Only ~20% of labelled ships have a labelled length
sea_based_HM_ships_gt0m = df_sea_based.loc[
    (df_sea_based['Ship_Status'] == True) &
    (df_sea_based['Number_Ships'] > 0)    &
    (df_sea_based['ship_length']>0)            &
    (
       (df_sea_based['Confidence']=='HIGH') |
       (df_sea_based['Confidence']=='MEDIUM')
    )
]


# All ships in sea based bursts with H or M confidence and length >100m
sea_based_HM_ships_gt100m = df_sea_based.loc[
    (df_sea_based['Ship_Status'] == True) &
    (df_sea_based['Number_Ships'] > 0)    &
    (df_sea_based['ship_length']>100)            &
    (
       (df_sea_based['Confidence']=='HIGH') |
       (df_sea_based['Confidence']=='MEDIUM')
    )
]

# All ships in sea based bursts with H or M confidence and length < 100m
sea_based_HM_ships_lt100m = df_sea_based.loc[
    (df_sea_based['Ship_Status'] == True) &
    (df_sea_based['Number_Ships'] > 0)    &
    (df_sea_based['ship_length']<100)            &
    (
       (df_sea_based['Confidence']=='HIGH') |
       (df_sea_based['Confidence']=='MEDIUM')
    )
]


# All ships in sea based bursts with L confidence
sea_based_L_ships = df_sea_based.loc[
    (df_sea_based['Ship_Status'] == True) &
    (df_sea_based['Number_Ships'] > 0)    &
    (
       df_sea_based['Confidence']=='LOW'
    )
]

############################################################################
#Find bursts with H+M ships only
#   a)Get bursts with H+M ships
#   b)Remove bursts with L ships
#   c)Remove bursts with non-ships

# sea_based_HM_ships
# sea_based_L_ships
# sea_based_non_ships

sea_based_bursts_with_HM_ships = sea_based_HM_ships.drop_duplicates(
                              subset=cols_to_check, keep='first')
sea_based_bursts_with_L_ships = sea_based_L_ships.drop_duplicates(
                              subset=cols_to_check, keep='first')

#Remove bursts that contain L confidence ships
m_df1 = sea_based_bursts_with_HM_ships.merge(sea_based_bursts_with_L_ships[
                                            ['Product_ID', 'Burst_Number']], 
                                              on=['Product_ID', 
                                                  'Burst_Number'], 
                                              how='left', indicator=True
                                             )
left_only_keys = m_df1[m_df1['_merge']=='left_only'][
                      ['Product_ID', 'Burst_Number']
                      ]

sea_based_bursts_with_HM_ships_no_L_ships = \
      sea_based_bursts_with_HM_ships.merge(
                          left_only_keys,
                          on=['Product_ID', 'Burst_Number'],
                          how='inner'
                          )
      
# Remove bursts that contain non-ships            
m_df2 = sea_based_bursts_with_HM_ships_no_L_ships.merge(
                         sea_based_bursts_with_non_ships[
                          ['Product_ID', 'Burst_Number']],
                         on=['Product_ID', 
                             'Burst_Number'], 
                         how='left', indicator=True
                         )

left_only_keys = m_df2[m_df2['_merge']=='left_only'][
                      ['Product_ID', 'Burst_Number']
                                                    ]

sea_based_bursts_with_only_HM_ships = \
      sea_based_bursts_with_HM_ships_no_L_ships.merge(
                            left_only_keys,
                            on=['Product_ID', 'Burst_Number'],
                            how='inner'
                            )

num_ships_in_sea_based_bursts_with_only_HM_ships = \
     sea_based_bursts_with_only_HM_ships['Number_Ships'].sum()

#Get list of ships of sea_based_bursts_with_only_HM_ships
# Find all rows in sea_based_ships that match with Product_ID and Burst_Number
ships_in_sea_based_bursts_with_only_HM_ships = sea_based_HM_ships.merge(
    sea_based_bursts_with_only_HM_ships[['Product_ID', 'Burst_Number']],  
    on=['Product_ID', 'Burst_Number'],
    how='inner'
)
#Save ship list to text file
ships_in_sea_based_bursts_with_only_HM_ships.to_csv(
                f'{csv_path}/{dataset}_{sar_mode}_IW{iw_mode}_'
                f'ship_list_for_sea_based_bursts_with_only_HM_ships.csv')

print(f'Number of bursts in '
      f'sea-based-bursts-with-only-HM-ships is: '
      f'{len(sea_based_bursts_with_only_HM_ships)}')
print(f'Number of ships  in '
      f'sea-based-bursts-with-only-HM-ships is: '
      f'{num_ships_in_sea_based_bursts_with_only_HM_ships}')


# Get bursts/ships for HM ships of differnt sizes
ship_sizes=[0,25,50,100]
file_list=[]
for index, ship_size in enumerate(ship_sizes):
    ##########################################################################
    # Get bursts which only have HM ships > {ship_size}m
    # Need to remove bursts that have any ships <{ship_size}m
    ship_length = ships_in_sea_based_bursts_with_only_HM_ships['ship_length']
    ships_gtXm_in_sea_based_bursts_with_only_HM_ships = \
          ships_in_sea_based_bursts_with_only_HM_ships.loc[
              (ship_length >ship_size) 
          ]
    
    ships_ltXm_in_sea_based_bursts_with_only_HM_ships = \
          ships_in_sea_based_bursts_with_only_HM_ships.loc[
              (ship_length <ship_size) | (ship_length.isna())
          ]
    
    sea_based_bursts_wth_HM_ships_gtXm = \
        ships_gtXm_in_sea_based_bursts_with_only_HM_ships.drop_duplicates(
            subset=['Product_ID', 'Burst_Number'],keep='first'
            )
    
    sea_based_bursts_wth_HM_ships_ltXm = \
        ships_ltXm_in_sea_based_bursts_with_only_HM_ships.drop_duplicates(
            subset=['Product_ID', 'Burst_Number'],keep='first'
            )
    
    
    m_df1 = sea_based_bursts_wth_HM_ships_gtXm.merge(
             sea_based_bursts_wth_HM_ships_ltXm[['Product_ID','Burst_Number']],
             on=['Product_ID', 'Burst_Number'],
             how='left', indicator=True
             )
    
    sea_based_bursts_with_only_HM_ships_gtXm = m_df1[m_df1['_merge'] 
                                =='left_only'].drop(columns=['_merge'])
    
    
    #Get list of ships of sea_based_bursts_with_only_HM_ships_gtXm
    # Find all rows in sea_based_ships that match with Product_ID and Burst_Number
    ships_in_sea_based_bursts_with_only_HM_ships_gtXm = \
        sea_based_HM_ships.merge(
           sea_based_bursts_with_only_HM_ships_gtXm[['Product_ID',
                                                     'Burst_Number'
                                                     ]],  # Only need the key columns from df2
            on=['Product_ID', 'Burst_Number'],
            how='inner'
        )
    #Save ship list to text file
    ships_in_sea_based_bursts_with_only_HM_ships_gtXm.to_csv(
                    f'{csv_path}/{dataset}_{sar_mode}_IW{iw_mode}_'
                    f'ship_list_for_sea_based_bursts_with_only_HM_ships'
                    f'_gt{ship_size}m.csv')
    file_list.append(f'{csv_path}/{dataset}_{sar_mode}_IW{iw_mode}_'
    f'ship_list_for_sea_based_bursts_with_only_HM_ships_gt{ship_size}m.csv')
        
    num_ships_in_sea_based_bursts_with_only_HM_ships_gtXm = \
                       len(ships_in_sea_based_bursts_with_only_HM_ships_gtXm)
    

    print(f'Number of bursts in '
          f'sea-based-bursts-with-only-HM-ships-gt{ship_size}m is: '
          f'{len(sea_based_bursts_with_only_HM_ships_gtXm)}')
    print(f'Number of ships  in '
          f'sea-based-bursts-with-only-HM-ships-gt{ship_size}m is: '
          f'{num_ships_in_sea_based_bursts_with_only_HM_ships_gtXm}')
    
save_path=csv_path
report_log=f'{save_path}/{dataset}_{sar_mode}_IW{iw_mode}_Report.dat'
with open(report_log, 'w') as f:
    f.write(f'---------------------------'
            f'Report file for {dataset}-IW{iw_mode} Dataset)'
            f'---------------------------\n')
    f.write(f'Number of SAR images :\t{num_sar_images}\n')
    f.write(f'Number of bursts:\t{num_bursts}\n')
    f.write(f'Number of ships :\t{num_ships}\n')
    f.write(f'Number of non-ships :\t{num_non_ships}\n')
    f.write(f'Number of sea-based-bursts :\t{num_sea_bursts}\n')
    f.write(f'Number of sea-based-bursts-with-ships :\t'
            f'{len(sea_based_bursts_with_ships)}\n')
    f.write(f'Number of sea-based-burst-ships :\t{num_sea_ships}\n')
    f.write(f'Number of sea-based-bursts-with-non-ships :\t'
            f'{len(sea_based_bursts_with_non_ships)}\n')
    f.write(f'Number of sea-based-burst-non-ships :\t{num_sea_non_ships}\n')
    f.write(f'Number of sea-based-bursts-without-ships :\t'
            f'{len(sea_based_bursts_without_ships)}\n')
    f.write(f'Number of sea-based-bursts-without-non-ships :\t'
            f'{len(sea_based_bursts_without_non_ships)}\n')
    f.write(f'Number of empty sea-based-bursts :\t'
            f'{num_empty_sea_based_bursts}\n')
    f.write(f'-----------------------------------------------------------\n')
    f.write(f'Number of sea-based-bursts-with-only-HM-ships :\t'
            f'{len(sea_based_bursts_with_only_HM_ships)}\n')
    f.write(f'Number of sea-based-burst-HM-ships :\t'
            f'{num_ships_in_sea_based_bursts_with_only_HM_ships}\n')
    f.write(f'Number of sea-based-burst-with-only-HM-ships_gt100m :\t'
            f'{len(sea_based_bursts_with_only_HM_ships_gtXm)}\n')
    f.write(f'Number of sea-based-burst-HM-ships-gt100m :\t'
            f'{num_ships_in_sea_based_bursts_with_only_HM_ships_gtXm}\n')
    f.write(f'-----------------------------------------------------------\n\n')
    f.write(f'Burst/Ship list for: HM-only-sea-based-bursts-and-ships: '
            f'\t{csv_path}/{dataset}_{sar_mode}_IW{iw_mode}_'
            f'ship_list_for_sea_based_bursts_with_only_HM_ships.csv\n\n')
    f.write(f'Burst/Ship list for: HM-only-sea-based-bursts-and-ships_gt100m: '
            f'{file_list[3]}\n\n')
    f.write(f'Burst list for empty-sea-based-bursts : \t'
            f'{csv_path}/{dataset}_{sar_mode}_IW{iw_mode}_'
            f'empty_sea_based_bursts.csv\n')
    f.write(f'-----------------------------------------------------------\n')


with open(report_log,'r') as file:
    for line in file:
        print(line, end='')  # end='' avoids adding extra newlines
    