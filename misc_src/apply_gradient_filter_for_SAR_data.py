#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 15:01:56 2025

@author: joe
"""
from skimage.exposure import equalize_adapthist
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def enhance_and_display_sar_gradient(raw_i_data: np.ndarray):
    """
    Calculates the spatial gradient of raw SAR I-channel data and displays
    it side-by-side with the original data for comparison. 🛰️

    Args:
        raw_i_data (np.ndarray): A 2D NumPy array representing the raw,
                                 in-phase SAR data.
    """
    print("Calculating spatial gradient...")

    # --- 1. Calculate the Spatial Gradient ---
    # The Sobel operator calculates the gradient in the x and y directions.
    # This is a standard and effective method for edge/fringe detection.
    gradient_x = sobel(raw_i_data, axis=0)
    gradient_y = sobel(raw_i_data, axis=1)

    # The total gradient magnitude is calculated using the hypotenuse.
    # This gives a measure of the total intensity of the fringe at each point.
    gradient_magnitude = np.hypot(gradient_x, gradient_y)

    # --- 2. Display the Results ---
    print("Displaying results...")
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('SAR Data Enhancement using Spatial Gradient', fontsize=8)

    # Display the original raw data on the left
    ax1 = axes[0]
    im1 = ax1.imshow(raw_i_data, cmap='gray', aspect='auto')
    ax1.set_title('Original Raw I-Channel Data')
    ax1.set_xlabel('Range Samples')
    ax1.set_ylabel('Azimuth Samples')
    #fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

    # Display the gradient magnitude on the right
    ax2 = axes[1]
    # 'plasma' or 'viridis' are good colormaps for highlighting gradients
    im2 = ax2.imshow(gradient_magnitude, cmap='viridis', aspect='auto')
    ax2.set_title('Gradient Magnitude (Enhanced Fringes)')
    ax2.set_xlabel('Range Samples')
    ax2.set_ylabel('Azimuth Samples')
    #fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
 

def apply_clahe_and_display(raw_i_data: np.ndarray):
    """
    Applies the CLAHE algorithm to raw SAR I-channel data and displays
    it side-by-side with the original data for comparison. 🛰️

    Args:
        raw_i_data (np.ndarray): A 2D NumPy array representing the raw,
                                 in-phase SAR data.
    """
    print("Applying CLAHE algorithm...")

    # --- 1. Apply CLAHE ---
    # `equalize_adapthist` is scikit-image's implementation of CLAHE.
    # `kernel_size`: Defines the size of the regional neighborhood for equalization.
    #                Choosing a size around 1/8th of the image dimension is a good start.
    # `clip_limit`: Sets the contrast enhancement limit. Lower values (e.g., 0.01)
    #               prevent over-amplification of noise.
    clahe_data = equalize_adapthist(
        raw_i_data,
        kernel_size=np.array(raw_i_data.shape) // 16,
        clip_limit=0.01
    )

    # --- 2. Display the Results ---
    print("Displaying results...")
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('SAR Data Enhancement using CLAHE', fontsize=16)

    # Display the original raw data on the left
    ax1 = axes[0]
    im1 = ax1.imshow(raw_i_data, cmap='gray', aspect = 'auto')
    ax1.set_title('Original Raw I-Channel Data')
    ax1.set_xlabel('Azimuth Samples')
    ax1.set_ylabel('Range Samples')
    #fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

    # Display the CLAHE-enhanced data on the right
    ax2 = axes[1]
    im2 = ax2.imshow(clahe_data, cmap='gray', aspect = 'auto')
    ax2.set_title('CLAHE Enhanced Data')
    ax2.set_xlabel('Range Samples')
    ax2.set_ylabel('Azimuth Samples')
    #fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_roi_as_offset_lines(arr_2d: np.ndarray, row_start: int, row_end: int, col_start: int, col_end: int):
    """
    Extracts, normalizes, and plots a region of a 2D array as a series of
    vertically offset line graphs.

    Args:
        arr_2d (np.ndarray): The source 2D NumPy array.
        row_start (int): The starting row index of the ROI.
        row_end (int): The ending row index of the ROI (exclusive).
        col_start (int): The starting column index of the ROI.
        col_end (int): The ending column index of the ROI (exclusive).
    """
    # --- 1. Extract the Region of Interest (ROI) ---
    roi_data = arr_2d[row_start:row_end, col_start:col_end]

    if roi_data.size == 0:
        print("Error: The specified region is empty. Please check your coordinates.")
        return

    # --- 2. Normalize the ROI data to the [0, 1] range ---
    min_val = roi_data.min()
    max_val = roi_data.max()
    
    # Avoid division by zero if the ROI is flat
    if max_val - min_val > 0:
        normalized_roi = (roi_data - min_val) / (max_val - min_val)
    else:
        normalized_roi = np.zeros_like(roi_data)

    # --- 3. Plot the data ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # The number of columns determines the x-axis values
    x_axis = np.arange(normalized_roi.shape[1])

    # Plot each row with a vertical offset
    for i, row in enumerate(normalized_roi):
        # The offset 'i' separates each line vertically by 1 unit
        ax.plot(x_axis, row + i, lw=1.5)

    # --- 4. Format the plot ---
    ax.set_title(f'Normalized Rows {row_start}-{row_end-1} from Columns {col_start}-{col_end-1}')
    ax.set_xlabel(f'Column Index (Offset by {col_start})')
    ax.set_ylabel('Vertically Offset Rows')

    # Set the y-axis ticks to correspond to the original row index
    # and adjust limits for better visibility
    row_indices = np.arange(row_start, row_end)
    ax.set_yticks(np.arange(len(row_indices)))
    ax.set_yticklabels(row_indices)
    ax.set_ylim(-0.5, len(row_indices) - 0.5)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def parse_row_selection_string(selection_str: str) -> list[int]:
    """
    Parses a string of rows and ranges (e.g., "1,10,25, 51:54, 70")
    into a sorted list of unique integer indices.

    Args:
        selection_str: The string defining the rows to select.

    Returns:
        A sorted list of unique row indices.
    """
    indices = set()
    # Remove whitespace and split by comma
    parts = selection_str.replace(" ", "").split(',')
    
    for part in parts:
        if not part:
            continue
        if ":" in part:
            # Handle range like "51:54"
            start, end = map(int, part.split(':'))
            # Add all numbers from start to end (inclusive)
            indices.update(range(start, end + 1))
        else:
            # Handle single number
            indices.add(int(part))
            
    return sorted(list(indices))

def plot_selected_lines(arr_2d: np.ndarray, row_selection: str, col_start: int, col_end: int):
    """
    Extracts, normalizes, and plots specified rows from a 2D array as
    vertically offset line graphs.

    Args:
        arr_2d (np.ndarray): The source 2D NumPy array.
        row_selection (str): A string specifying rows (e.g., "1, 10:15, 25").
        col_start (int): The starting column index of the ROI.
        col_end (int): The ending column index of the ROI (exclusive).
    """
    # --- 1. Parse the selection string to get row indices ---
    row_indices = parse_row_selection_string(row_selection)
    
    if not row_indices:
        print("Error: No valid rows were selected.")
        return

    # --- 2. Extract the Region of Interest (ROI) using advanced indexing ---
    try:
        # Select specific rows and the column slice
        roi_data = arr_2d[row_indices, col_start:col_end]
    except IndexError as e:
        print(f"Error: Row index out of bounds. {e}")
        return
        
    if roi_data.size == 0:
        print("Error: The specified region is empty. Please check your coordinates.")
        return

    # --- 3. Normalize the ROI data to the [0, 1] range ---
    min_val, max_val = roi_data.min(), roi_data.max()
    normalized_roi = (roi_data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(roi_data)

    # --- 4. Plot the data ---
    fig, ax = plt.subplots(figsize=(12, 8))
    x_axis = np.arange(normalized_roi.shape[1])

    # Plot each selected row with a vertical offset
    for i, row_data in enumerate(normalized_roi):
        ax.plot(x_axis, row_data + i, lw=1.5)

    # --- 5. Format the plot ---
    ax.set_title(f'Selected Rows from Columns {col_start}-{col_end-1}')
    ax.set_xlabel(f'Column Index (Offset by {col_start})')
    ax.set_ylabel('Original Row Index')

    # Set the y-axis ticks to correspond to the original row indices
    ax.set_yticks(np.arange(len(row_indices)))
    ax.set_yticklabels(row_indices)
    ax.set_ylim(-0.5, len(row_indices) - 0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def create_synthetic_sar_data(dims=(512, 512), num_scatterers=6):
    """
    Generates a synthetic 2D NumPy array to simulate raw SAR I-channel data
    with several isolated point scatterers showing hyperbolic fringes.
    """
    # Create a base of low-level random noise
    data = np.random.randn(dims[0], dims[1]) * 0.1

    # Add point scatterers
    for _ in range(num_scatterers):
        # Randomize position and amplitude for variety
        px = np.random.randint(int(dims[0]*0.2), int(dims[0]*0.8))
        py = np.random.randint(int(dims[1]*0.2), int(dims[1]*0.8))
        amp = np.random.uniform(2.0, 5.0)
        
        # Create the grid of coordinates relative to the scatterer
        y = np.arange(dims[1])
        x = np.arange(dims[0])
        X, Y = np.meshgrid(y - py, x - px)
        
        # Chirp rate determines the curvature of the hyperbolic fringes
        chirp_rate = np.random.uniform(0.003, 0.008)
        
        # Calculate the phase of the I-channel signal (cosine component)
        phase = chirp_rate * X**2
        scatterer_signal = amp * np.cos(phase)
        
        # Add a Gaussian envelope to localize the signal
        envelope = np.exp(-((X**2 / (2 * 200**2)) + (Y**2 / (2 * 20**2))))
        data += scatterer_signal * envelope

    # Normalize the final data to a typical floating-point range
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

if __name__ == '__main__':
    # --- Generate or Load Your Data ---
    # In a real application, you would load your data here, for example:
    # my_sar_data = np.load('path/to/your/data.npy')
    print("Generating synthetic raw SAR data for demonstration...")
    synthetic_data = create_synthetic_sar_data(dims=(600, 600), num_scatterers=8)

    # --- Run the Enhancement and Display Routine ---
    enhance_and_display_sar_gradient(synthetic_data)
    apply_clahe_and_display(synthetic_data)
