#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:12:36 2025

Script to perform CFAR on chips used for testing performance of Resnet18 & 34
models.
Will process each chip and classify whether ship present or not
Need to specify train and guard parameters
Assume pFa of 1e-2

As parse through each chip record whether has ship(s) or not



@author: joe
"""
from skimage.measure import find_contours, label
from scipy.ndimage import label, find_objects, center_of_mass, distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.ndimage import median_filter
import time, os 

from SAR_code_module import *
#from SAR_focusing_module import *

from blob_module import *

    
# Set Train and Guard paramaters

train_band = 1
guard_band = 20
pfa        = 1e-2
kernel1 = np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]])

start = time.time()


dest_dir = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/Reports/CFAR_Results/'
slc_chip_dir  = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/slc_chip_data/'
csv_dir  = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/Train/Reports/'

chip_list = os.listdir(slc_chip_dir)
#ship_coordinates = get_ship_coordinates(csv_file)

TP=[]
TN=[]
FP=[]
FN=[]
Has_Ship=[]
Ship_Detected=[]
Number_Ships_Detected=[]
total_process_time = 0

#for file in chip_list[:5]:
for file in chip_list:
    #read chip file
    file_path = f'{slc_chip_dir}{file}'
    if not os.path.isfile(file_path):
        print(f'Error: File not found : {file_path}')
        continue
    try:
        chip_data = np.load(f'{slc_chip_dir}{file}')
    except:
        print(f'File Error with {file}')
        continue
    # check if ship or No-ship
    cfar_file_start_time = time.time()
    # Run CFAR
    cfar_gamma_map2=optimized_cfar_detector(chip_data, train_band, guard_band, pfa)
    cfar_gamma_map2_eroded=erosion(cfar_gamma_map2,kernel1)
    cfar_gamma_map2_eroded_dilated=dilation(cfar_gamma_map2_eroded,kernel1)              
    cfar_gamma_map2_eroded_dilatedx2=dilation(cfar_gamma_map2_eroded_dilated,kernel1)

    """
    plot_2d_array(cfar_gamma_map2,f"{file.replace('.npy','')}",'linear')
    plot_2d_array(cfar_gamma_map2_eroded,f"{file.replace('.npy','_eroded')}",'linear')
    plot_2d_array(cfar_gamma_map2_eroded_dilated,f"{file.replace('.npy','_eroded_dilated')}",'linear')
    plot_2d_array(cfar_gamma_map2_eroded_dilatedx2,f"{file.replace('.npy','_eroded_dilatedx2')}",'linear')
    """
    nearness_threshold = 5
    blob_data=detect_blobs(cfar_gamma_map2_eroded_dilatedx2, nearness_threshold)

    number_of_blobs = int(blob_data[0])
    Number_Ships_Detected.append(number_of_blobs)
    blob_centroids  = blob_data[1] 
    blobs           = blob_data[2]
    print(f'Detected {number_of_blobs} Ships in {file}')
    if not 'NoShip' in file:
        #Ship chip file
        Has_Ship.append('True')
        if number_of_blobs > 0:
            #Ship detected
            Ship_Detected.append(1)
            TP.append(1)
            TN.append(0)
            FP.append(0)
            FN.append(0)
        else:
            #Ship not detected
            Ship_Detected.append(0)
            TP.append(0)
            TN.append(0)
            FP.append(0)
            FN.append(1) 
    else:
        #Non-Ship chip file 
        Has_Ship.append('False')
        if number_of_blobs > 0:
            #Ship detected - where none should be
            Ship_Detected.append(1)
            TP.append(0)
            TN.append(0)
            FP.append(1)
            FN.append(0)
        else:
            #Ship not detected - as exepected
            Ship_Detected.append(0)
            TP.append(0)
            TN.append(1)
            FP.append(0)
            FN.append(0)
    cfar_file_end_time = time.time()
    process_time = cfar_file_end_time - cfar_file_start_time
    total_process_time += process_time    
               
end = time.time()
print(f'Time taken for {len(chip_list)} CFAR operations is :{end-start} s')
print(f'Time taken for {len(chip_list)} CFAR operations without file loading is :{total_process_time} s')


df = pd.DataFrame(
    list(zip(chip_list, Has_Ship, Ship_Detected, TP, TN, FP, FN, Number_Ships_Detected)),
    columns=['Chip_List', 'Has_Ship', 'Ship_Detected', 'TP', 'TN', 'FP', 'FN', 'Number_Ships_Detected']
)

df_ships = df[df['Has_Ship'] == 'True']
df_non_ships = df[df['Has_Ship'] == 'False']
                  
df.to_csv(f'{dest_dir}DetectionResults_GuardBand_{guard_band}_TrainBand_{train_band}.csv', index=False)

def accuracy(TP,TN,FP,FN):
    return (sum(TP) + sum(TN)) /(sum(TP) + sum(TN) + sum(FP) + sum(FN))

#accuracy  = (sum(TP) + sum(TN)) /(sum(TP) + sum(TN) + sum(FP) + sum(FN))
def precision(TP,FP):
    return sum(TP)/(sum(TP) + sum(FP))

def recall(TP, FN):
    return sum(TP)/(sum(TP) + sum(FN))

def F1(P,R):
    #2 * (precision * recall) / (precision + recall)
    return 2 * (P * R) / (P + R)




print(f'####################################################################')
print(f"Based on {len(df[df['Has_Ship'] == 'True'])} Ship Chips and {len(df[df['Has_Ship'] == 'False'])} Non-Ship Chips: ")
print(f' Accuracy  (A) [TP+TN/(TP+TN+FP+FN)] : {accuracy(TP,TN,FP,FN)}')
P=precision(TP,FP)
print(f' Precision (P) [TP/(TP+FP)]          : {P}')
R=recall(TP,FN)
print(f' Recall    (R) [TP/(TP+FN)]          : {R}')
print(f' F1-Score      [2*P*R/P+R)]          : {F1(P,R)}')
print(f'####################################################################')


ship_tp = df_ships['TP']
ship_tn = df_ships['TN']
ship_fn = df_ships['FN']
ship_fp = df_ships['FP']

print(f'####################################################################')
print(f"Based on {len(df_ships[df_ships['Has_Ship'] == 'True'])} Ship Chips : ")
print(f' Accuracy  (A) [TP+TN/(TP+TN+FP+FN)] : {accuracy(ship_tp, ship_tn,ship_fn, ship_fp)}')
P=precision(ship_tp,ship_fp)
print(f' Precision (P) [TP/(TP+FP)]          : {P}')
R=recall(ship_tp, ship_fn)
print(f' Recall    (R) [TP/(TP+FN)]          : {R}')
print(f' F1-Score      [2*P*R/P+R)]          : {F1(P,R)}')
print(f'####################################################################')


non_ship_tp = df_non_ships['TP']
non_ship_tn = df_non_ships['TN']
non_ship_fn = df_non_ships['FN']
non_ship_fp = df_non_ships['FP']

print(f'####################################################################')
print(f"Based on {len(df_non_ships[df_non_ships['Has_Ship'] == 'False'])} Non-Ship Chips : ")
print(f' Accuracy  (A) [TP+TN/(TP+TN+FP+FN)] : {accuracy(non_ship_tp, non_ship_tn, non_ship_fn, non_ship_fp)}')
#For negative case TP is TN
P=precision(non_ship_tn,non_ship_fp)
print(f' Precision (P) [TP/(TP+FP)]          : {P}')
R=recall(non_ship_tn, non_ship_fn)
print(f' Recall    (R) [TP/(TP+FN)]          : {R}')
print(f' F1-Score      [2*P*R/P+R)]          : {F1(P,R)}')
print(f'####################################################################')


"""
### Test Area
file = chip_list[5]
file = 'S1B_IW_SLC__1SDV_20200310T064009_20200310T064037_020626_02718C_FC56_RAW_IW2_data_burst#2_Ship#1_1186_1218_up_left_125_chip.npy'
chip_data = np.load(f'{slc_chip_dir}{file}')
# check if ship or No-ship

# Run CFAR
cfar_gamma_map2=optimized_cfar_detector(chip_data, train_band, guard_band, pfa)
cfar_gamma_map2_eroded=erosion(cfar_gamma_map2,kernel1)
cfar_gamma_map2_eroded_dilated=dilation(cfar_gamma_map2_eroded,kernel1)              
cfar_gamma_map2_eroded_dilatedx2=dilation(cfar_gamma_map2_eroded_dilated,kernel1)


plot_2d_array(cfar_gamma_map2,f"{file.replace('.npy','')}")
plot_2d_array(cfar_gamma_map2_eroded,f"{file.replace('.npy','_eroded')}",'linear')
plot_2d_array(cfar_gamma_map2_eroded_dilated,f"{file.replace('.npy','_eroded_dilated')}",'linear')
plot_2d_array(cfar_gamma_map2_eroded_dilatedx2,f"{file.replace('.npy','_eroded_dilatedx2')}",'linear')


"""