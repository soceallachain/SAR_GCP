#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:38:35 2025

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, kstest, probplot

def fit_and_evaluate_weibull(intensity_data, plot=True, bins=100):
    """
    Fit a Weibull distribution to SAR intensity data and evaluate the fit.

    Parameters:
        intensity_data (array-like): 1D array of positive SAR intensity values.
        plot (bool): Whether to show diagnostic plots.
        bins (int): Number of histogram bins for PDF plotting.

    Returns:
        dict: Fitted parameters and K-S test results.
    """
    # Clean data (remove zeros/negatives)
    intensity = np.asarray(intensity_data)
    intensity = intensity[intensity > 0]

    # Fit Weibull (fix location to 0)
    shape, loc, scale = weibull_min.fit(intensity, floc=0)

    # K-S test
    D, p = kstest(intensity, 'weibull_min', args=(shape, loc, scale))

    results = {
        'shape': shape,
        'scale': scale,
        'ks_stat': D,
        'ks_p_value': p,
        'n_samples': len(intensity)
    }

    if plot:
        x = np.linspace(0, np.percentile(intensity, 99), 500)
        pdf_fitted = weibull_min.pdf(x, shape, loc, scale)
        cdf_empirical = np.sort(intensity)
        cdf_theoretical = weibull_min.cdf(cdf_empirical, shape, loc, scale)

        # Plot 1: Histogram + PDF
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(intensity, bins=bins, density=True, alpha=0.6, label='Empirical')
        plt.plot(x, pdf_fitted, 'r-', label='Weibull PDF')
        plt.title("Histogram + Weibull PDF")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.legend()

        # Plot 2: Empirical vs Theoretical CDF
        plt.subplot(1, 3, 2)
        plt.plot(cdf_empirical, np.linspace(0, 1, len(cdf_empirical)), label='Empirical CDF')
        plt.plot(cdf_empirical, cdf_theoretical, 'r--', label='Weibull CDF')
        plt.title("CDF Comparison")
        plt.xlabel("Intensity")
        plt.ylabel("Cumulative Probability")
        plt.legend()

        # Plot 3: Q–Q plot
        plt.subplot(1, 3, 3)
        probplot(intensity, dist=weibull_min, sparams=(shape, loc, scale), plot=plt)
        plt.title("Q–Q Plot (Weibull)")

        plt.tight_layout()
        plt.show()

    return results
