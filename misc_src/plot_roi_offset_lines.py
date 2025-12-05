#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:29:39 2025

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt

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


def create_synthetic_data(rows=50, cols=2000):
    """Generates synthetic 2D data with some interesting features."""
    # Create a base of random noise
    data = np.random.randn(rows, cols) * 0.1
    
    # Add a sine wave that changes frequency
    x = np.linspace(0, 100, cols)
    for i in range(rows):
        freq = 1 + (i / rows) * 4  # Frequency increases with row number
        amp = 1 + np.sin(i / 10)
        data[i, :] += np.sin(x * freq) * amp

    # Add a prominent transient feature
    data[10:20, 1000:1050] += 5
    return data


if __name__ == '__main__':
    # --- 1. Generate or load your data ---
    # This creates sample data. In your use case, you would load your own array.
    # my_array = np.load('path/to/your/data.npy')
    print("Generating synthetic 2D data...")
    my_array = create_synthetic_data(rows=50, cols=2000)

    # --- 2. Define your Region of Interest ---
    # Here, we plot 10 lines (rows 10 to 19)
    # from a specific column range (1000 to 1500).
    r_start = 10
    r_end = 20
    c_start = 1000
    c_end = 1500

    # --- 3. Call the plotting function ---
    plot_roi_as_offset_lines(my_array, r_start, r_end, c_start, c_end)