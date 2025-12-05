import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Import LogNorm for log scaling


# Directory containing the chip files
data_dir = '/home/joe/Work/Projects/RSAR/Test/SampleData'
output_dir = os.path.join(data_dir, 'Chip_Data')
# Function to gather chips for a particular burst
def gather_chips(root_name, burst_number):
    real_chips = []
    imag_chips = []
    binary_mask_chips = []
    
    
    def extract_chip_number(filename):
        match = re.search(r'_n(\d+)\.npy$', filename)  # Look for '_n<number>.npy'
        if match:
            return int(match.group(1))  # Return the number as an integer
        return -1  # Default value in case no match is found
    
    # Sorting files numerically based on chip number
    sorted_files = sorted(os.listdir(output_dir), key=extract_chip_number)
    
    # Loop through the files in the output directory
    for file in sorted_files: #sorting to ensure correct order
        if f'{root_name}_real_burst#{burst_number}' in file:
            real_chips.append(np.load(os.path.join(output_dir, file)))
        elif f'{root_name}_imag_burst#{burst_number}' in file:
            imag_chips.append(np.load(os.path.join(output_dir, file)))
        elif f'{root_name}_binary_mask_burst#{burst_number}' in file:
            binary_mask_chips.append(np.load(os.path.join(output_dir, file)))
    
    return real_chips, imag_chips, binary_mask_chips

# Function to merge chips back into the original burst image
def merge_chips(chips, chip_height=512, chip_width=3072):
    if not chips:
        raise ValueError("No chips to merge!")
    
    # Number of rows and columns based on the chip size
    #num_rows = len(chips) // (chips[0].shape[1] // chip_width)
    #print(f'chips shape is: {chips[0].shape}  chips length is : {len(chips)}')
    #print(f'chips shape is: {np.array(chips).shape}  chips length is : {len(chips)}')
    num_chips = np.array(chips).shape[0]
    orig_image_height = 1500  # Temporary - need to get from annotation file
    orig_image_width  = 25000 # Ditto    
    num_rows_in_merged_image = orig_image_height // chip_height
    num_cols_in_merged_image = orig_image_width // chip_width
    #img_height = num_rows * chip_height
    #img_width = chips[0].shape[1]
    # Fix it for now with approxomite values of height and width
    # In future need to extract these from annotation file
    img_height = num_rows_in_merged_image * chip_height
    img_width  = num_cols_in_merged_image * chip_width
    
    # Create an empty array to hold the merged image
    merged_image = np.zeros((img_height, img_width))
    #print(f'Merged Image size is: {merged_image.shape}')
    #print(f'Chips[0] size is : {chips[0].shape}')
    
    row = 0
    col = 0
    chip_index = 0
    while chip_index < num_chips:
        while col + chip_width <= img_width:
            #print(f'Chip Index = {chip_index}, row+chip_height ={row+chip_height}')
            merged_image[row:(row+chip_height),col:col+chip_width] = chips[chip_index]
            col = col + chip_width
            #print(f'col is: {col}')
            chip_index = chip_index + 1
        col = 0
        row = row + chip_height
    return merged_image
"""        
    chip_index = 0
    for row in range(0, img_height, chip_height):
        for col in range(0, img_width, chip_width):
            merged_image[row:row+chip_height, col:col+chip_width] = chips[chip_index]
            chip_index += 1
"""    
    

# Function to calculate amplitude from real and imaginary arrays
def calculate_amplitude(real, imag):
    amplitude = np.sqrt(np.square(real) + np.square(imag))
    return amplitude

# Function to plot the amplitude and binary_mask
def plot_amplitude_and_mask(amplitude, binary_mask, vmax=800, vmin=20):
    #fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    #screen_width = plt.get_current_fig_manager().window.winfo_screenwidth()  # Get screen width
    #fig, axs = plt.subplots(2, 1, figsize=(screen_width / 100, 12))  # Adjust width dynamically
    fig, axs = plt.subplots(2, 1, figsize=(15,15))  # Increase the width to 16 and the height to 8 for a wider plot


    
    # Plot amplitude in the first subplot
    im1 = axs[0].imshow(amplitude, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), aspect=5)
    #im1 = axs[0].imshow(amplitude, cmap='viridis', norm=plt.LogNorm(vmin=vmin, vmax=vmax))
    axs[0].set_title('Amplitude (Log Normal scale)')
    #fig.colorbar(im1, ax=axs[0])
    
    # Plot binary_mask in the second subplot
    im2 = axs[1].imshow(binary_mask, cmap='gray', aspect =5)
    axs[1].set_title('Binary Mask (Grayscale)')
    #fig.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    plt.show()

# Main function to test and visualize
def test_and_visualize(root_name, burst_number):
    # Step 1: Gather chips for the burst
    real_chips, imag_chips, binary_mask_chips = gather_chips(root_name, burst_number)
    
    # Step 2: Merge the chips to recreate the original burst
    real_merged = merge_chips(real_chips)
    imag_merged = merge_chips(imag_chips)
    binary_mask_merged = merge_chips(binary_mask_chips)
    
    # Step 3: Calculate amplitude
    amplitude = calculate_amplitude(real_merged, imag_merged)
    
    # Step 4: Plot the amplitude and binary_mask
    plot_amplitude_and_mask(amplitude, binary_mask_merged)
###############################################################################
# Test the function with a specific root name and burst number
#
# Need to add code to interrogate associated annotation file to determine 
# original file dimensions. Search for <linesPerBurst> and <samplesPerBurst> in 
# .xml file
#


#root_name = 'S1B_IW_SLC__1SDV_20200710T062304_20200710T062331_022405_02A85C_F12B'

root_name = 'S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E'
burst_number = '4'
test_and_visualize(root_name, burst_number)

