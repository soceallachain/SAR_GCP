#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3rd Dec 2025
Trying to read data from GCP and store locally and on GCP

@author: joe
"""


import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from modules.SAR_focusing_module import *
from modules.blob_module import *

from modules.SAR_code_module import *
from google.cloud import storage
from io import BytesIO


# --- Configuration ---
# Replace with your actual project ID and bucket name
project_id = "ds-ml-vertex-ai"
bucket_name = "ubot-sar-data"

# --- Execution ---
# Executes on a VM so knows project id already
client = storage.Client()

def gcp_list_dir(bucket, prefix):
    blobs=client.list_blobs(bucket, prefix=prefix, delimiter='/')
    files=[]
    all_blobs = list(blobs)
    print(f'Number of blobs found = {len(all_blobs)}')
    for blob in all_blobs:
        # Only want the part of the name *after* the prefix
        # If the prefix is "dir/sub/", and the blob name is "dir/sub/file.txt",
        # want "file.txt".
        relative_name = blob.name.replace(prefix, "", 1)
        #print(f'Relative name .... {relative_name}')

        # Exclude the "directory" itself if it's explicitly listed as a 
        # zero-byte object (e.g., 'dir/sub/')
        if relative_name and not relative_name.endswith('/'):
            files.append(relative_name)
     # --- Output ---
    print(f"Found {len(files)} files in {bucket}/{prefix}")
    return files

def download_and_load_npy(bucket_name, object_name):
    """Downloads a GCS object's bytes and loads it directly into a NumPy array."""
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    # 1. Download the file content as a single Bytes object
    data_bytes = blob.download_as_bytes()
    
    # 2. Use io.BytesIO to create an in-memory file-like object 
    #    that numpy.load() can read from.
    data_stream = BytesIO(data_bytes)
    
    # 3. Load the NumPy array from the in-memory stream
    np_array = np.load(data_stream)
    
    return np_array



def download_GCP_file(bucket_name, object_name, local_filename):
    """Downloads a GCS object to a local file and handles common exceptions."""
    try:
        # Initialize the client (assumes you are authenticated via gcloud auth or service account)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        print(f"Attempting to download gs://{bucket_name}/{object_name}...")
        
        # 1. Execute the download command
        blob.download_to_filename(local_filename)
        
        print(f"✅ Success: File downloaded to {local_filename}")
        return True

    except NotFound:
        # 2. Catch 404 Error (Object Not Found)
        print(f"❌ Error: The object '{object_name}' was NOT FOUND in the bucket.")
        print("Please verify the object path and bucket name.")
        return False
        
    except Forbidden:
        # 3. Catch 403 Error (Authorization/Permissions)
        print(f"❌ Error: Permission DENIED (403 Forbidden).")
        print("Check if the credentials used have 'storage.objects.get' permission for this bucket.")
        return False
        
    except GoogleAPICallError as e:
        # 4. Catch general API/Network Errors (Timeouts, Transient Errors)
        print(f"⚠️ Warning: A general API error occurred (e.g., network failure or timeout).")
        print(f"Error details: {e}")
        # Depending on the error, you might implement retry logic here.
        return False
    
    except Exception as e:
        # 5. Catch any unexpected errors (e.g., local disk I/O failure)
        print(f"❌ Critical Error: An unexpected error occurred: {e}")
        return False


def get_statistics(roi):    
    mean = roi.mean()
    std  = roi.std()
    SNR  = mean/std
    return mean, std, SNR


def intensity(arr):
    return np.square(np.abs(arr))



def process_data_on_GCP():
    
    
    path = "data/Copernicus/Train/Reports/Train/raw_burst_data/"
    file_name = "xxxx"
    bucket = client.bucket(bucket_name)
    
    """
    # Get the bucket and the blob (file)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Download the file contents into memory
    print(f"Downloading {file_name} in memory...")
    file_bytes = blob.download_as_bytes()
    data_stream = BytesIO(file_bytes)
    
    # Read the data directly into a Pandas DataFrame
    df = pd.read_csv(data_stream)
    print("\n--- Data Head (Method 2) ---")
    print(df.head())
    
    """
    
    
    plt.style.use('dark_background')
    source = 'GCP'
    Dataset = 'Public'
    
    if source == 'ext':
        raw_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_data'
        slc_path =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_data'
        slc_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/slc_burst_data'
        raw_burst_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/raw_burst_data'
    
        #Get ship locations in SLC file
        path = f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}'
        #img_path=f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m'
        #img_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results/range_compression_tests_16Oct2025'
        img_path =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_focusing_results/range_compression_tests_Public_29Oct2025'
        dest_dir =f'/media/joe/WD_12TB_HD1/jc_work/Copernicus/{Dataset}/rc_raw_burst_data'
    
    
    elif source == 'int':    
        raw_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/raw_data'
        slc_path =f'/home/joe/Work/RSAR/Test/sampleData/{Dataset}/slc_data'
        slc_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/slc_burst_data'
        raw_burst_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}/raw_burst_data'
        
        #Get ship locations in SLC file
        path = f'/home/joe/Work/Projects/RSAR/Test/SampleData/{Dataset}'
        img_path = f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m'
    
        
    elif source == 'Spain':
        raw_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/raw_data'
        slc_path =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/slc_data'
        slc_burst_dir =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/slc_burst_data'
        raw_burst_dir =f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}/raw_burst_data'
    
        #Get ship locations in SLC file
        path = f'/mnt/smbshare/RAWSARAI_Data/Copernicus/{Dataset}'
        img_path = f'/mnt/smbshare/RAWSARAI_Data/Copernicus/Sandbox/test_slc_raw_interferogram_results/{Dataset}_gt100m'
    
    elif source == 'GCP':
        raw_path =f'data/Copernicus/{Dataset}/raw_data/'
        slc_path =f'data/Copernicus/{Dataset}/slc_data/'
        slc_burst_dir =f'data/Copernicus/{Dataset}/slc_burst_data/'
        raw_burst_dir =f'data/Copernicus/{Dataset}/raw_burst_data/'
        dest_dir =f'data/Copernicus/{Dataset}/rc_raw_burst_data/'
    else:
        print(f'Source error Exiting.....')
        sys.exit()
    
    
    
    print(f'Bucket_Name = {bucket_name}   ...... rc_file_dir = {dest_dir} ')
    
    files_already_done = gcp_list_dir(bucket_name, dest_dir)
    print(f'Files already done .....')
    print(f'{files_already_done[0:10]}')
    #burst_list = os.listdir(raw_burst_dir)
    burst_list = gcp_list_dir(bucket_name, raw_burst_dir)
    save_rc_data = 1
    
    print(f'{burst_list[0:5]}')
    
    #exit(0)
    print(f' ***** Past exit *****')
    #read chip file
    for file in burst_list[:19]:
        if '_real_' not in file:
            print(f' File {file} does not contain _real_')
            continue
        if file.replace('_real_','_rc_') in files_already_done:
            print(f'{file} already processed. Continuing to next file.')
            continue   
        if source == 'GCP': 
            real_file_path = f'{raw_burst_dir}{file}'
            imag_file_path = real_file_path.replace('_real_','_imag_')
            
            raw_real_burst = download_and_load_npy(bucket_name, real_file_path)
            raw_imag_burst = download_and_load_npy(bucket_name, imag_file_path)
    
        else:
            real_file_path = f'{raw_burst_dir}/{file}'
            imag_file_path = real_file_path.replace('_real_','_imag_')
        
            if not os.path.isfile(real_file_path) or not os.path.isfile(imag_file_path):
                print(f'Error: File not found : Either  {real_file_path} or {imag_file_path}')
                continue
            try:
                raw_real_burst = np.load(f'{real_file_path}')
                raw_imag_burst = np.load(f'{imag_file_path}')
                    
            except:
                print(f'File Error with {file}')
                continue  # exit the loop if any of the files are missing
            
            
        #Create raw burst as real + i*imag
        print(f'Found suitable file for range compression {real_file_path}')
        raw_burst = raw_real_burst + 1j * raw_imag_burst
        #Pad raw burst in azimuth and  range
        range_padding = 1000
        az_padding    = 200
        padded_raw_burst = np.pad(raw_burst, ((az_padding,az_padding),
                                  (range_padding,range_padding)),
                                  mode='constant', constant_values=0
                                  )
    
        #Range compress raw and padded_raw
        #rc_raw_burst = range_compress_burst(raw_burst)
        rc_padded_burst = range_compress_burst(padded_raw_burst)
        resized_rc_padded_burst=rc_padded_burst[
                                     az_padding:az_padding+raw_burst.shape[0],
                                     range_padding:range_padding+raw_burst.shape[1]
                                     ]
        del rc_padded_burst
        rc_padded_burst = resized_rc_padded_burst
        
        if save_rc_data:
            rc_raw_burst = file.replace('_real_','_rc_')
            #np.save(f'{dest_dir}/{rc_raw_burst}', rc_padded_burst)
            os.makedirs('/home/joe/tmp/', exist_ok=True)
            np.save(f'/home/joe/tmp/{rc_raw_burst}', rc_padded_burst)
            
        
        
    #plot Raw, SLC, rc_Raw, and rc_padded_Raw
    
    #Label ships in SLC and rc_RAW and rc_padded_Raw
    
    #?? Do blob detection on chips of 1024x1024 centered on expected ship position
    # and then label with detected position


process_data_on_GCP()


