###################################################
## Test merging of burst-chips
## may be an issue with how chips were aligned
##

import os
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
    
    # Loop through the files in the output directory
    for file in sorted(os.listdir(output_dir)):  # Sorting to ensure correct order
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
    num_rows = len(chips) // (chips[0].shape[1] // chip_width)
    img_height = num_rows * chip_height
    img_width = chips[0].shape[1]
    
    # Create an empty array to hold the merged image
    merged_image = np.zeros((img_height, img_width))
    
    chip_index = 0
    for row in range(0, img_height, chip_height):
        for col in range(0, img_width, chip_width):
            merged_image[row:row+chip_height, col:col+chip_width] = chips[chip_index]
            chip_index += 1
    
    return merged_image

# Function to calculate amplitude from real and imaginary arrays
def calculate_amplitude(real, imag):
    amplitude = np.sqrt(np.square(real) + np.square(imag))
    return amplitude

# Function to plot the amplitude and binary_mask
def plot_amplitude_and_mask(amplitude, binary_mask, vmax=800, vmin=20):
    #fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    #screen_width = plt.get_current_fig_manager().window.winfo_screenwidth()  # Get screen width
    #fig, axs = plt.subplots(2, 1, figsize=(screen_width / 100, 12))  # Adjust width dynamically
    fig, axs = plt.subplots(2, 1, figsize=(32, 32))  # Increase the width to 16 and the height to 8 for a wider plot


    
    # Plot amplitude in the first subplot
    im1 = axs[0].imshow(amplitude, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), aspect=1)
    #im1 = axs[0].imshow(amplitude, cmap='viridis', norm=plt.LogNorm(vmin=vmin, vmax=vmax))
    axs[0].set_title('Amplitude (Log Normal scale)')
    #fig.colorbar(im1, ax=axs[0])
    
    # Plot binary_mask in the second subplot
    im2 = axs[1].imshow(binary_mask, cmap='gray', aspect =1)
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

# Test the function with a specific root name and burst number
#root_name = 'S1B_IW_SLC__1SDV_20200710T062304_20200710T062331_022405_02A85C_F12B'
root_name = 'S1B_IW_SLC__1SDV_20200715T063116_20200715T063143_022478_02AA98_0F2E'
burst_number = '4'
test_and_visualize(root_name, burst_number)
