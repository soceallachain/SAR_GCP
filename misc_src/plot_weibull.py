#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:53:04 2025

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data (replace this with your actual dataset 's')
np.random.seed(42)
s = np.random.weibull(2, 1000) * 5

# Fit Weibull distribution to the data
shape, loc, scale = stats.weibull_min.fit(s, floc=0)

# Create histogram
plt.figure(figsize=(10, 6))
hist, bin_edges, _ = plt.hist(s, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# Generate points for the fitted Weibull PDF
x = np.linspace(0, max(s), 100)
pdf = stats.weibull_min.pdf(x, shape, loc, scale)

# Plot the fitted Weibull distribution
plt.plot(x, pdf, 'r-', lw=2, label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')

plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Histogram with Fitted Weibull Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
