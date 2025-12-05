#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:19:58 2025

@author: joe
"""
import numpy as np
import time
from numba import jit

@jit(nopython=True)
def mandelbrot_numba(height, width, max_iter):
    y = np.linspace(-1.4, 1.4, height)
    x = np.linspace(-2, 0.8, width)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    z = np.zeros_like(c)
    divtime = np.zeros(z.shape, dtype=np.int32)

    for i in range(max_iter):
        z = z**2 + c
        for row in range(z.shape[0]):
            for col in range(z.shape[1]):
                if divtime[row, col] == 0 and (z[row, col].real**2 + z[row, col].imag**2) > 4:
                    divtime[row, col] = i
                    z[row, col] = 2

    return divtime

# Measure execution time
start = time.perf_counter()
result = mandelbrot_numba(1000, 1500, 100)
end = time.perf_counter()
print(f"Numba execution time: {end - start:.2f} seconds")
