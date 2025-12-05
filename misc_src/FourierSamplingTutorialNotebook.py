#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:00:17 2025

@author: joe
"""# Fourier Sampling and DSP Tutorial (Jupyter Notebook)

# This tutorial demonstrates the role of sampling in Fourier analysis,
# with emphasis on Nyquist theorem, aliasing, window functions,
# zero-padding, and data length optimization.
# Extended to include real-world signals, frequency axis computation,
# inverse FFT, and noise mitigation techniques.

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import windows
from scipy.io import wavfile
from ipywidgets import interact, FloatSlider
import warnings
warnings.filterwarnings("ignore")

# -- 1. Define helper functions --
def plot_signal_and_spectrum(signal, Fs, title_prefix=""):
    N = len(signal)
    T = 1.0 / Fs
    t = np.linspace(0.0, N*T, N, endpoint=False)
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]

    plt.figure(figsize=(12, 4))

    # Time-domain signal
    plt.subplot(1, 2, 1)
    plt.plot(t, signal)
    plt.title(f"{title_prefix} Signal (Time Domain)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # Frequency-domain spectrum
    plt.subplot(1, 2, 2)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title(f"{title_prefix} Spectrum (Frequency Domain)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()

# -- 2. Demonstrate Nyquist Theorem --
print("## Nyquist Sampling Theorem\n")

Fs = 1000  # Sampling frequency
f_signal = 200  # Signal frequency (Hz)
T = 1.0 / Fs
N = 1000
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(2.0 * np.pi * f_signal * x)

plot_signal_and_spectrum(y, Fs, title_prefix=f"{f_signal} Hz")

# -- 3. Show aliasing due to undersampling --
print("## Aliasing Example (Undersampling)\n")

# Use a signal with 600 Hz frequency, which exceeds Nyquist (500 Hz)
# and simulate undersampling with Fs = 1000 Hz
f_alias = 600
Fs = 1000  # Below twice the signal frequency (Nyquist condition violated)
T = 1.0 / Fs
x = np.linspace(0.0, N*T, N, endpoint=False)
y_alias = np.sin(2.0 * np.pi * f_alias * x)

plot_signal_and_spectrum(y_alias, Fs, title_prefix=f"{f_alias} Hz (Aliased)")

# Explain what the aliased frequency is observed as:
f_observed = np.abs(Fs - f_alias)
print(f"Due to undersampling, the {f_alias} Hz signal appears at {f_observed} Hz in the FFT output.")

# -- 4. Use of Window Functions to Reduce Spectral Leakage --
print("## Window Functions\n")

window = windows.hann(N)
y_windowed = y * window

plot_signal_and_spectrum(y_windowed, Fs, title_prefix="Windowed")

# -- 5. Zero-padding --
print("## Zero Padding\n")

N_pad = 2048
y_padded = np.zeros(N_pad)
y_padded[:N] = y

plot_signal_and_spectrum(y_padded, Fs, title_prefix="Zero-Padded")

# -- 6. Importance of Power-of-Two Sample Lengths --
print("## Power-of-Two Sample Lengths\n")

lengths = [1000, 1024, 2048]
for L in lengths:
    y_test = np.sin(2.0 * np.pi * f_signal * np.linspace(0.0, L/Fs, L, endpoint=False))
    %timeit fft(y_test)

print("FFT is generally fastest when input length is a power of two (e.g., 1024, 2048).")

# -- 7. Frequency Axis Calculation --
print("## Frequency Axis Computation\n")

N = 1024
Fs = 2000
T = 1.0 / Fs
t = np.linspace(0.0, N*T, N, endpoint=False)
signal = np.sin(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*500*t)

# FFT and axis setup
yf = fft(signal)
yf_shifted = np.fft.fftshift(yf)
xf = fftfreq(N, T)
xf_shifted = np.fft.fftshift(xf)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(xf, np.abs(yf))
plt.title("FFT Without fftshift")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(xf_shifted, np.abs(yf_shifted))
plt.title("FFT With fftshift")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# -- 8. Real-world Signal Example (e.g., synthetic noisy signal) --
print("## Real-World Signal Example (Synthetic)\n")

np.random.seed(0)
Fs = 1000
T = 1.0 / Fs
N = 1024
x = np.linspace(0.0, N*T, N, endpoint=False)
# Add multiple sinusoids + noise
y_real = (np.sin(2*np.pi*50*x) + 0.5*np.sin(2*np.pi*120*x) + 0.3*np.random.randn(N))

plot_signal_and_spectrum(y_real, Fs, title_prefix="Noisy Real-World")

# -- 9. Inverse FFT and Denoising --
print("## Inverse FFT and Noise Mitigation\n")

# Interactive filter control
def interactive_denoise(cutoff):
    yf = fft(y_real)
    xf = fftfreq(N, T)
    yf_filtered = yf.copy()
    yf_filtered[np.abs(xf) > cutoff] = 0
    y_denoised = ifft(yf_filtered).real

    plt.figure(figsize=(10, 4))
    plt.plot(x, y_real, label="Original (Noisy)", alpha=0.5)
    plt.plot(x, y_denoised, label=f"Denoised (cutoff={cutoff:.0f} Hz)", linewidth=2)
    plt.title("Interactive Denoising with Low-Pass Filter")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

interact(interactive_denoise, cutoff=FloatSlider(min=50, max=500, step=10, value=200))

# Summary tip:
print("\nTip: Use frequency-domain filtering + windowing to reduce spectral leakage and isolate real signal components.")
