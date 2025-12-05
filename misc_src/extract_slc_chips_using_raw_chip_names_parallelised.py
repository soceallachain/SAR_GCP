import os
import sys
import glob
import numpy as np
import multiprocessing
from SAR_code_module import *  # Assuming this module defines:
                                # extract_product_burst_and_chip_information and
                                # get_slc_chip_and_bursts

from datetime import datetime
#Start timer
start_time =datetime.now()



# Set parameters
swath_of_interest = 2
polarization = 'vh'
swath_of_interest_index = swath_of_interest - 1
IW_mode = swath_of_interest_index

# What mode are the data be used for : SLC or RAW
sar_mode = 'RAW'

# Set parameters for chip number extraction
chip_size = [512, 3072]

# Frequency multipliers for RAW vs SLC
freq_multipliers = [1, 54.595959/64.345238, 46.915402/64.345238]
if sar_mode == 'RAW':
    sampling_freq_mul = freq_multipliers[IW_mode]
else:
    sampling_freq_mul = 1

freq_scale_factor = sampling_freq_mul
print(f'**** In extract_slc_chips_using_raw ...: freq_scale_factor = {freq_scale_factor}')

# Directory paths
slc_dir = '/media/joe/Elements/jc_work/Copernicus/SLC_Public'
chip_source_dir ='/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/raw'
destination_dir = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips/slc'

#destination_dir ='/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_m_and_hc_sea_ships_no_nonships_gt100m_chips/slc'
#slc_dir = '/mnt/smbshare/RAWSARAI_Data/Copernicus/SLC_Public'
#chip_source_dir = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_gt100m_chips/raw'
#destination_dir = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_gt100m_chips/slc'




# Get the list of files to process (exclude files with binary_mask, imag, or _N in the filename)
files__ = [f for f in glob.glob(os.path.join(chip_source_dir, "*.npy"))
         if '_real_' in os.path.basename(f)
         ]

files = files__[:5]


#         if '_binary_mask_' not in os.path.basename(f)
#         and '_imag_' not in os.path.basename(f)
#         and '_N' not in os.path.basename(f)
def process_file(f):
    """
    Process a single file:
      1. Extract product, burst, and chip information.
      2. Get the SLC chip data.
      3. Construct a new filename in the destination directory.
      4. Save the SLC chip data.
    """
    try:
        print(f'Processing file: {f}')
        file_information = extract_product_burst_and_chip_information(f)
        slc_id = file_information[0]
        burst_number = int(file_information[1])
        chip_number  = int(file_information[2])
        
        # Get chip data; assume chip_data[0] is what we need to save
        chip_data = get_slc_chip_and_bursts(slc_id, burst_number, chip_number, freq_scale_factor)
        
        # Create new filename; include necessary information in the file name
        slc_chip_name = f'{destination_dir}/{slc_id}_{sar_mode}_IW{swath_of_interest}_slc_burst#{burst_number}_chip#{chip_number}.npy'
        print(f'Renaming to: {slc_chip_name}')
        
        # Save the SLC chip data to the new file
        np.save(slc_chip_name, chip_data[0])
        return f"Processed: {f}"
    
    except Exception as e:
        return f"Error processing {f}: {e}"

def main():
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    print(f'Files are: {files}')    
    # Use multiprocessing Pool to speed up file processing
    #with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_file, files)
    
    for res in results:
        print(res)
    
    print("All files processed.")

if __name__ == "__main__":
    main()





#Stop timer
end_time = datetime.now()
delta = end_time - start_time

# Extract days
days = delta.days

# Extract hours, minutes, seconds from the remainder
hours, remainder = divmod(delta.seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f'******************* Completed at  {end_time} ************')
print(f'Execution time was {days}days:{hours}hours:{minutes}:{seconds}seconds')


