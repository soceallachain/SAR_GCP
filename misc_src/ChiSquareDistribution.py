#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:41:39 2025

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Step 1: Generate standard normal samples
num_samples = 100000  # Increase for smoother histogram
normal_samples = np.random.randn(num_samples)  # From N(0,1)

# Step 2: Square the samples
squared_samples = normal_samples ** 2  # Now follows χ²_1 distribution

# Step 3: Plot the histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot original normal distribution
axes[0].hist(normal_samples, bins=100, density=True, alpha=0.6, color='blue', label='Samples')
x = np.linspace(-4, 4, 1000)
axes[0].plot(x, stats.norm.pdf(x, 0, 1), 'r', lw=2, label="N(0,1) PDF")
axes[0].set_title("Standard Normal Distribution (N(0,1))")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Density")
axes[0].legend()

# Plot chi-square (squared) distribution
axes[1].hist(squared_samples, bins=100, density=True, alpha=0.6, color='green', label='Samples')
x = np.linspace(0, 10, 1000)
axes[1].plot(x, stats.chi2.pdf(x, df=1), 'r', lw=2, label=r"$\chi^2_1$ PDF")
axes[1].set_title("Chi-Square Distribution (df=1)")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.show()
