#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:23:37 2026

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

def compute_image_offset(img1: np.ndarray, img2: np.ndarray, plot_surface: bool = True):
    """
    Computes the sub-pixel row and column offset between two 2D image arrays using 
    Zero-Normalized Frequency-Domain Cross-Correlation and plots the gamma(u,v) surface.

    Parameters:
        img1 (np.ndarray): 2D array representing the reference image.
        img2 (np.ndarray): 2D array representing the shifted image.
        plot_surface (bool): If True, renders and displays the 2D correlation surface.

    Returns:
        row_offset (float): Vertical offset in pixels (img2 relative to img1).
        col_offset (float): Horizontal offset in pixels (img2 relative to img1).
    """
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D arrays.")

    # 1. Dynamic Range Floating-Point Normalization
    f = img1.astype(np.float64)
    g = img2.astype(np.float64)
    
    f = (f - np.min(f)) / (np.max(f) - np.min(f) + 1e-12)
    g = (g - np.min(g)) / (np.max(g) - np.min(g) + 1e-12)

    # 2. Local Zero-Mean Centering (DC Offset Removal)
    f_centered = f - np.mean(f)
    g_centered = g - np.mean(g)

    # 3. Spatial Windowing (2D Hann Window to prevent FFT edge leakage)
    win_f = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    win_g = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))

    f_win = f_centered * win_f
    g_win = g_centered * win_g

    # 4. Zero-Padding Dimensions (N1 + N2 - 1, M1 + M2 - 1)
    pad_shape = (f.shape[0] + g.shape[0] - 1, f.shape[1] + g.shape[1] - 1)

    # 5. 2D Fast Fourier Transform
    F = fft2(f_win, s=pad_shape)
    G = fft2(g_win, s=pad_shape)

    # 6. Complex Conjugate Multiplication (Cross-Spectrum)
    cross_spectrum = np.conj(F) * G

    # 7. Inverse FFT to Spatial Domain and Zero-Lag Centering
    raw_corr = np.real(ifft2(cross_spectrum))
    gamma = fftshift(raw_corr)

    # 8. Energy Normalization to clamp gamma inside [-1, 1]
    norm_factor = np.sqrt(np.sum(f_win**2) * np.sum(g_win**2)) + 1e-12
    gamma /= norm_factor

    # 9. Coarse Peak Detection
    r_peak, c_peak = np.unravel_index(np.argmax(gamma), gamma.shape)
    center_r, center_c = pad_shape[0] // 2, pad_shape[1] // 2

    # 10. Sub-Pixel Refinement via 2D Local Quadratic Hessian Surface Fitting
    dr, dc = 0.0, 0.0
    if 0 < r_peak < gamma.shape[0] - 1 and 0 < c_peak < gamma.shape[1] - 1:
        patch = gamma[r_peak - 1 : r_peak + 2, c_peak - 1 : c_peak + 2]

        # First Partial Derivatives
        d_r = (patch[2, 1] - patch[0, 1]) / 2.0
        d_c = (patch[1, 2] - patch[1, 0]) / 2.0

        # Second Partial Derivatives (Hessian Matrix Elements)
        d_rr = patch[2, 1] - 2 * patch[1, 1] + patch[0, 1]
        d_cc = patch[1, 2] - 2 * patch[1, 1] + patch[1, 0]
        d_rc = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4.0

        H = np.array([[d_rr, d_rc], [d_rc, d_cc]])
        grad = np.array([d_r, d_c])

        if np.abs(np.linalg.det(H)) > 1e-10:
            delta = -np.linalg.solve(H, grad)
            dr, dc = delta[0], delta[1]

    row_offset = (r_peak - center_r) + dr
    col_offset = (c_peak - center_c) + dc
    
    # Not finding correct peak. Why not?
    print(f'r_peak - center_r + dr = {r_peak} - {center_r} + {dr}')
    print(f'c_peak - center_c + dc = {c_peak} - {center_c} + {dc}')

    # 11. Surface Visualization
    if plot_surface:
        fig, ax = plt.subplots(figsize=(9, 7))
        lags_r = np.arange(-center_r, pad_shape[0] - center_r)
        lags_c = np.arange(-center_c, pad_shape[1] - center_c)

        im = ax.imshow(
            gamma,
            extent=[lags_c[0], lags_c[-1], lags_r[-1], lags_r[0]],
            cmap="viridis",
            aspect="auto"
        )
        ax.scatter(
            col_offset,
            row_offset,
            color="red",
            marker="x",
            s=120,
            linewidths=2,
            label=f"Detected Offset: Row={row_offset:.2f}px, Col={col_offset:.2f}px"
        )

        ax.set_title(r"Zero-Normalized Cross-Correlation Surface $\gamma(u,v)$", fontsize=12)
        ax.set_xlabel("Column Lag $v$ (pixels)", fontsize=10)
        ax.set_ylabel("Row Lag $u$ (pixels)", fontsize=10)
        fig.colorbar(im, ax=ax, label=r"Correlation Coefficient $\gamma$")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    return row_offset, col_offset



################################################################################# Gemini function to label plots with Ship Information

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def track_and_annotate_vessels(
    ax: plt.Axes, df: pd.DataFrame, interval_sec: float = 0.5
):
    """Monitors mouse movement on a Matplotlib plot every 0.5 seconds and draws a

    connection line and label to the nearest vessel coordinate if the mouse
    moves by more than half the distance to the next nearest vessel.

    Parameters:
    -----------
    ax : plt.Axes
        The Matplotlib axes handle containing the plot.
    df : pd.DataFrame
        DataFrame with columns 'MMSI', 'Length', 'SoG', 'Scene Row', 'Scene Col'.
    interval_sec : float
        Monitoring interval in seconds (default: 0.5s).
    """
    fig = ax.figure

    # Extract scene coordinates into NumPy array for vector distance operations
    # Note: 'Scene Col' = X-coordinate, 'Scene Row' = Y-coordinate
    scene_cols = df["Scene Col"].to_numpy()
    scene_rows = df["Scene Row"].to_numpy()
    coords = np.column_stack((scene_cols, scene_rows))

    # Stateful variables to track mouse and annotation status
    last_mouse_pos = [None, None]  # [col, row]
    active_line = None
    active_text = None
    timer = None

    def update_annotation(mouse_col, mouse_row):
        nonlocal active_line, active_text

        # 3. Calculate Euclidean distance from mouse to all scene coordinates
        distances = np.hypot(coords[:, 0] - mouse_col, coords[:, 1] - mouse_row)

        if len(distances) == 0:
            return

        # Find the index of the closest coordinate
        closest_idx = np.argmin(distances)
        closest_coord = coords[closest_idx]

        # Extract vessel metadata
        vessel_row = df.iloc[closest_idx]
        mmsi = vessel_row["MMSI"]
        length = vessel_row["Length"]
        sog = vessel_row["SOG"]

        # Clean up previous drawing elements if they exist
        if active_line is not None:
            active_line.remove()
        if active_text is not None:
            active_text.remove()

        # 4. Draw line from scene coordinates to mouse coordinates
        (active_line,) = ax.plot(
            [closest_coord[0], mouse_col],
            [closest_coord[1], mouse_row],
            color="red",
            linestyle="--",
            linewidth=1.5,
        )

        # Place text label at mouse position
        label = f"MMSI:{mmsi}, Length:{length}, SoG:{sog}"
        active_text = ax.text(
            mouse_col,
            mouse_row,
            label,
            fontsize=9,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="yellow",
                edgecolor="black",
                alpha=0.8,
            ),
        )

        fig.canvas.draw_idle()

    def on_mouse_move(event):
        nonlocal last_mouse_pos
        # 2. Capture mouse position in plot data coordinates [col (x), row (y)]
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            last_mouse_pos = [event.xdata, event.ydata]

    def check_mouse_movement():
        # 5. Monitor mouse position on each 0.5s timer tick
        if last_mouse_pos[0] is None or last_mouse_pos[1] is None:
            return

        curr_col, curr_row = last_mouse_pos[0], last_mouse_pos[1]

        # Calculate distances from current mouse position to all scene points
        distances = np.hypot(coords[:, 0] - curr_col, coords[:, 1] - curr_row)

        if len(distances) < 2:
            # If only 1 vessel exists, update directly
            update_annotation(curr_col, curr_row)
            return

        # Sort distances to identify nearest (d1) and second nearest (d2) points
        sorted_indices = np.argsort(distances)
        d1 = distances[sorted_indices[0]]  # Distance to nearest coordinate
        d2 = distances[sorted_indices[1]]  # Distance to second nearest coordinate

        # Step 5 Rule: Update if movement/distance > half the distance to the next nearest scene point
        if d1 > (0.5 * d2):
            update_annotation(curr_col, curr_row)

    def on_close(event):
        # Clean up timer when the plot window is closed
        nonlocal timer
        if timer is not None:
            timer.stop()

    # Connect mouse movement event handler
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # 6. Set up recurring 0.5s timer that runs continuously until figure closes
    timer = fig.canvas.new_timer(interval=int(interval_sec * 1000))
    timer.add_callback(check_mouse_movement)

    # Disconnect timer on plot close event
    fig.canvas.mpl_connect("close_event", on_close)

    # Start monitoring timer
    timer.start()


###############################################################################################################


def interactive_vessel_annotator(ax: plt.Axes, df: pd.DataFrame):
    """Attaches a right-click listener to a Matplotlib Axes that finds the closest

    vessel coordinate in `df`, draws a line to it, and adds an MMSI/Length label.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes handle of the target plot.
    df : pd.DataFrame
        DataFrame containing 'MMSI', 'Length', 'Scene Row', and 'Scene Col'.
    """
    fig = ax.figure

    # Track drawn elements (line and text label) across clicks
    active_line = None
    active_text = None

    def on_click(event):
        nonlocal active_line, active_text

        # 1. Filter for right-click inside the target axes
        # (event.button == 3 represents right-click in Matplotlib; MouseButton.RIGHT in newer versions)
        if event.button != 3 or event.inaxes != ax:
            return

        # 2. Get mouse position in data coordinates [Col (X), Row (Y)]
        mouse_col = event.xdata
        mouse_row = event.ydata

        if mouse_col is None or mouse_row is None:
            return

        # 3. Calculate Euclidean distance between mouse coordinates and scene coordinates
        scene_cols = df["Scene Col"].values
        scene_rows = df["Scene Row"].values

        distances = np.sqrt(
            (scene_cols - mouse_col) ** 2 + (scene_rows - mouse_row) ** 2
        )
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # Requirement 5: If distance is <= 0.5 units, do nothing
        if min_dist <= 0.5:
            return

        # 4. Extract closest vessel details
        closest_row = df.iloc[min_idx]
        mmsi = closest_row["MMSI"]
        length = closest_row["Length"]
        vessel_col = closest_row["Scene Col"]
        vessel_row = closest_row["Scene Row"]

        # 5. Remove previous line and text label if they exist
        if active_line is not None:
            active_line.remove()
        if active_text is not None:
            active_text.remove()

        # 6. Draw line from closest vessel coordinates to mouse coordinates
        (active_line,) = ax.plot(
            [vessel_col, mouse_col],
            [vessel_row, mouse_row],
            color="red",
            linestyle="--",
            linewidth=1.5,
            marker="o",
        )

        # 7. Place text label at the mouse position
        label_str = f"MMSI:{mmsi}, Length:{length}"
        active_text = ax.text(
            mouse_col,
            mouse_row,
            label_str,
            color="black",
            fontsize=9,
            backgroundcolor="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="red",
                facecolor="white",
                alpha=0.8,
            ),
        )

        # Redraw the canvas to render updates
        fig.canvas.draw_idle()

    # Connect click event callback
    cid = fig.canvas.mpl_connect("button_press_event", on_click)

    # Disconnect automatically if the plot window is closed
    def on_close(event):
        fig.canvas.mpl_disconnect(cid)

    fig.canvas.mpl_connect("close_event", on_close)





