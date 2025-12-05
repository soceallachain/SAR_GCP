#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:35:20 2025

@author: joe
"""
import numpy as np
M=1548
N=24072

arr_2d = np.random.rand(M,N)

fft_arr_2d = np.fft.ifftshift(np.fft.ifft(fft_arr_2d, axis=1), axes=1)
fft_arr_2d = np.fft.ifft(arr_2d, axis=0)