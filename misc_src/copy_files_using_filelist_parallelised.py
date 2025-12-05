import os
import shutil
from multiprocessing import Pool, cpu_count

from datetime import datetime

#Start timer
start_time =datetime.now()


# Settings
src_dir   = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/raw_public_burst_chip_data'
dest_dir  = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/public_high_confidence/public_hc_sea_ships_no_nonships_gt100m_chips'
file_list = 'RAW_high_conf_ship_no_nonships_only_sea_gt100.txt'
#src_dir = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/raw_public_burst_chip_data'
#dest_dir = '/mnt/smbshare/RAWSARAI_Data/Copernicus/public_burst_data/public_high_confidence/public_sea_only_empty_chips/raw'
#file_list = 'Public_IW2_empty_sea_chips_list_reduced.txt'
file_list_path =f'/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/{file_list}'

# Read file list
with open(file_list_path, 'r') as file:
    files = [line.strip() for line in file]

# Ensure destination directory exists
os.makedirs(dest_dir, exist_ok=True)

def copy_file(filename):
    src_path = os.path.join(src_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    
    # Ensure destination subdirectories exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
        return f"Copied: {filename}"
    except Exception as e:
        return f"Failed to copy {filename}: {e}"

if __name__ == '__main__':
    # Use number of CPUs or set your preferred number of workers
    num_workers = min(8, cpu_count())  # for example, max 8 workers

    with Pool(num_workers) as pool:
        results = pool.map(copy_file, files)

    for res in results:
        print(res)

    print(f'Raw real files copied from {src_dir} to {dest_dir}')



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


