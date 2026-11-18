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


import numpy as np

# Create synthetic reference image (e.g., Gaussian noise + feature pattern)
np.random.seed(42)
base_img = np.random.randn(256, 256)

# Create shifted image with +12.35 row offset and -8.65 column offset using SciPy
from scipy.ndimage import shift
shifted_img = shift(base_img, shift=[12.35, -8.65], mode='reflect')

# Add synthetic contrast & brightness scale variations (simulating different sensor settings)
shifted_img = (shifted_img * 3.5) + 50.0

# Execute alignment function
r_off, c_off = compute_image_offset(base_img, shifted_img, plot_surface=True)
print(f"Calculated Row Offset: {r_off:.2f} pixels")
print(f"Calculated Col Offset: {c_off:.2f} pixels")


