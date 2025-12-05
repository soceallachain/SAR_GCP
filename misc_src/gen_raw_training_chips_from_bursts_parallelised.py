import os
import numpy as np
import re
from multiprocessing import Pool, cpu_count


###############################################################################
#
# Note: Currently expects destination directory to be empty. If files in it 
#       these are assumed to be wanted and only processes bursts that are not
#       aleady present. If starting afresh delete contents of destination 
#       directory
###############################################################################

# Helper function to extract base name and burst number using regex
def extract_base_and_burst(filename):
    pattern = re.compile(r'^(.*)_(real|imag|binary_mask)_burst#(\d+)\.npy$')
    match = pattern.match(filename)
    if match:
        base_name = match.group(1)  # Base part of the file name
        file_type = match.group(2)  # real, imag, or binary_mask
        burst_number = match.group(3)  # Burst number
        return base_name, file_type, burst_number
    return None, None, None

# Helper function to extract chips from an image
def extract_chipsI(image, chip_height=512, chip_width=3072):
    img_height, img_width = image.shape
    chips = []
    for y in range(0, img_height, chip_height):
        for x in range(0, img_width, chip_width):
            if y + chip_height <= img_height and x + chip_width <= img_width:
                chip = image[y:y+chip_height, x:x+chip_width]
                chips.append(chip)
    return chips




# Extract chips ensuring complete coverage of burst
def extract_chips(image, chip_height=512, chip_width=3072):
    img_height, img_width = image.shape
    chips = []

    for y in range(0, img_height, chip_height):
        for x in range(0, img_width, chip_width):
            # Ensure the chip stays within the image bounds
            y_start = min(y, img_height - chip_height)
            x_start = min(x, img_width - chip_width)

            # Extract a fixed-size chip
            chip = image[y_start:y_start + chip_height, x_start:x_start + chip_width]
            chips.append(chip)

    return chips



# Helper function to process a single group using multiple arguments
def process_single_group(args):
    base_name, burst_number, files, data_dir, output_dir = args
    try:
        print(f'Processing files for base: {base_name} burst: {burst_number}')
        
        # Load numpy arrays
        real_img = np.load(os.path.join(data_dir, files['real']))
        imag_img = np.load(os.path.join(data_dir, files['imag']))
        binary_mask = np.load(os.path.join(data_dir, files['binary_mask']))

        # Extract chips
        real_chips = extract_chips(real_img)
        imag_chips = extract_chips(imag_img)
        binary_mask_chips = extract_chips(binary_mask)

        # Save chips
        num_chips = min(len(real_chips), len(imag_chips), len(binary_mask_chips))
        for i in range(num_chips):
            chip_suffix = f'_chip#{i+1}.npy'
            real_chip_filename = f"{base_name}_real_burst#{burst_number}{chip_suffix}"
            imag_chip_filename = f"{base_name}_imag_burst#{burst_number}{chip_suffix}"
            mask_chip_filename = f"{base_name}_binary_mask_burst#{burst_number}{chip_suffix}"

            np.save(os.path.join(output_dir, real_chip_filename), real_chips[i])
            np.save(os.path.join(output_dir, imag_chip_filename), imag_chips[i])
            np.save(os.path.join(output_dir, mask_chip_filename), binary_mask_chips[i])

        return True
    except Exception as e:
        print(f"Error processing {base_name} burst {burst_number}: {str(e)}")
        return False

# Modified main processing function
def process_npy_files(npy_files, data_dir, output_dir, num_processes=None):
    # Grouping logic remains the same
    grouped_files = {}
    for npy_file in npy_files:
        base_name, file_type, burst_number = extract_base_and_burst(npy_file)
        if base_name and file_type and burst_number:
            key = (base_name, burst_number)
            if key not in grouped_files:
                grouped_files[key] = {}
            grouped_files[key][file_type] = npy_file

    # Prepare arguments for multiprocessing
    task_args = []
    for (base_name, burst_number), files in grouped_files.items():
        if all(k in files for k in ('real', 'imag', 'binary_mask')):
            task_args.append((
                base_name, 
                burst_number,
                files,
                data_dir,
                output_dir
            ))

    # Create worker pool (use 3/4 of available CPUs by default)
    num_workers = num_processes or max(1, int(cpu_count() * 0.75))
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_group, task_args)
    
    print(f"Processed {len(task_args)} groups with {sum(results)} successes")

# Usage example
if __name__ == '__main__':
    data_dir   = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/raw_public_burst_data'   
    output_dir = '/media/joe/Elements/jc_work/Copernicus/public_burst_data/raw_public_burst_chip_data'   
    #output_dir = os.path.join(data_dir, 'Chip_Data_To_Be_Copied_to_HD')
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Check if some files already processes and if so skip them
    filenames = os.listdir(output_dir)
    # Filter only files (skip directories)
    files = [f for f in filenames if os.path.isfile(os.path.join(output_dir, f))]

    # Replace '_chip#xx.npy' with '.npy'
    pattern = re.compile(r'_chip#\d+\.npy$')
    completed_files = [pattern.sub('.npy', f) for f in files]

    #with open('/home/joe/Misc/raw_bursts_for_chips_completed.txt', 'r') as file:
    #    # Read all lines, strip newline characters, and store in a list
    #    completed_files = [line.strip() for line in file]

    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and f not in completed_files]
    print(f'Completed files are:')
    print(*completed_files, sep='\n') 
    print(f'*********************************************************************************')
    print(f' Files to yet process are:')
    print(*npy_files, sep='\n')
    # Start processing with multiprocessing
    #process_npy_files(npy_files, data_dir, output_dir)

