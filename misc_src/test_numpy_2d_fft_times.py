#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:00:33 2025

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Generate 100 equally spaced sizes for m and n
m_values = np.linspace(100, 1600, num=100, dtype=int)
n_values = np.linspace(2000, 28000, num=100, dtype=int)

fft_times = []
sizes = []

for m, n in zip(m_values, n_values):
    arr = np.random.rand(m, n)
    start = time.perf_counter()
    np.fft.fft2(arr)
    elapsed = time.perf_counter() - start
    fft_times.append(elapsed)
    sizes.append(m * n)
    print(f"Size: {m}x{n}, Time: {elapsed:.6f} seconds")

efficiency = np.array(sizes) / np.array(fft_times)

plt.figure(figsize=(10, 6))
plt.plot(sizes, efficiency, marker='o', linestyle='-')
plt.xlabel('Array Size (m × n)')
plt.ylabel('Elements Processed per Second')
plt.title('Efficiency of 2D FFT in NumPy')
plt.xscale('log')  # Apply logarithmic scaling to the x-axis
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()
