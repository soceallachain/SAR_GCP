#####################################################################
# Script to take training data directory, parse it, create a directory
# for the chips and then generates chips of size 512x3072 from each
# burst. The chips do not overlap.

import os
import re
import numpy as np

# Directory containing the .npy files
# Remember to set!
data_dir = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Bursts'
output_dir = '/home/joe/Work/Projects/RSAR/Test/RAW_Training_Data/Chips/'
#output_dir = os.path.join(data_dir, 'Chip_Data')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

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
def extract_chips(image, chip_height=512, chip_width=3072):
    img_height, img_width = image.shape
    chips = []
    for y in range(0, img_height, chip_height):
        for x in range(0, img_width, chip_width):
            if y + chip_height <= img_height and x + chip_width <= img_width:
                chip = image[y:y+chip_height, x:x+chip_width]
                chips.append(chip)
    return chips

# Function to process the npy files
def process_npy_files(npy_files):
    # Group the files by base name and burst number
    grouped_files = {}
    
    for npy_file in npy_files:
        base_name, file_type, burst_number = extract_base_and_burst(npy_file)
        
        if base_name and file_type and burst_number:
            if (base_name, burst_number) not in grouped_files:
                grouped_files[(base_name, burst_number)] = {}
            grouped_files[(base_name, burst_number)][file_type] = npy_file

    # Debugging: Show how the files are grouped
    print(f'Grouped Files: {grouped_files}')

    # Process each group
    for (base_name, burst_number), files in grouped_files.items():
        print(f'Processing files for base: {base_name} burst: {burst_number}')
        
        # Check if the group has all three files: real, imag, binary_mask
        if 'real' in files and 'imag' in files and 'binary_mask' in files:
            print(f'Processing group: {files}')
            real_img = np.load(os.path.join(data_dir, files['real']))
            imag_img = np.load(os.path.join(data_dir, files['imag']))
            binary_mask = np.load(os.path.join(data_dir, files['binary_mask']))
            
            # Debugging: Ensure files are loaded properly
            print(f'Loaded real, imag, and binary mask files. Shapes: real={real_img.shape}, imag={imag_img.shape}, binary_mask={binary_mask.shape}')
            
            # Extract chips
            real_chips = extract_chips(real_img)
            imag_chips = extract_chips(imag_img)
            binary_mask_chips = extract_chips(binary_mask)
            
            # Ensure all three have the same number of chips
            num_chips = min(len(real_chips), len(imag_chips), len(binary_mask_chips))
            
            for i in range(num_chips):
                # Save chips to the output directory
                chip_suffix = f'_n{i+1}.npy'
                
                real_chip_filename = f"{base_name}_real_burst#{burst_number}{chip_suffix}"
                imag_chip_filename = f"{base_name}_imag_burst#{burst_number}{chip_suffix}"
                mask_chip_filename = f"{base_name}_binary_mask_burst#{burst_number}{chip_suffix}"
                
                np.save(os.path.join(output_dir, real_chip_filename), real_chips[i])
                np.save(os.path.join(output_dir, imag_chip_filename), imag_chips[i])
                np.save(os.path.join(output_dir, mask_chip_filename), binary_mask_chips[i])
                
                print(f'Saved chip {i+1} for burst: {burst_number}')
        else:
            print(f'Skipping incomplete group: {files}')
    print(f'Grouped Files: {grouped_files}')
    print(f'********** COMLETED *************************************')

# Get a list of all npy files in the directory
npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
process_npy_files(npy_files)

