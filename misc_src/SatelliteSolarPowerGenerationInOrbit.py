#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:15:11 2025

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
R_earth = 6371e3           # Earth radius in meters
h = 700e3                  # Orbit altitude in meters
mu = 3.986e14              # Earth gravitational parameter (m^3/s^2)
solar_irradiance = 1360    # W/m^2
panel_area = 1.0           # m^2
panel_efficiency = 0.25    # 25%

# Calculate orbit period (circular orbit)
r = R_earth + h
T = 2 * np.pi * np.sqrt(r**3 / mu)  # seconds

# Calculate eclipse duration fraction (approximate)
# Earth shadow radius at satellite altitude
Re = R_earth
Rs = 696340e3  # Sun radius (m)
d_es = 1.496e11  # Earth-Sun distance (m)

# Angular radius of Earth and Sun as seen from satellite
theta_e = np.arcsin(Re / r)
theta_s = np.arcsin(Rs / d_es)

# Approximate eclipse half-angle (radians)
# Using geometry from https://space.stackexchange.com/questions/17628/eclipse-duration-for-satellite
eclipse_half_angle = theta_e + theta_s

# Fraction of orbit in eclipse
eclipse_fraction = 2 * eclipse_half_angle / (2 * np.pi)

# Time arrays
num_points = 1000
times = np.linspace(0, T, num_points)  # seconds

# Power output array initialization
power = np.zeros_like(times)

# Define sunlight and eclipse intervals
eclipse_time = eclipse_fraction * T
sunlight_time = T - eclipse_time

# Define eclipse start and end times (centered in orbit)
eclipse_start = (T - eclipse_time) / 2
eclipse_end = eclipse_start + eclipse_time

# Calculate power during sunlight assuming panel normal points to sun
# For simplicity, power = efficiency * irradiance * area * cos(angle)
# Angle varies linearly from 0 to pi during sunlight (simulate incidence angle change)
for i, t in enumerate(times):
    if eclipse_start <= t <= eclipse_end:
        power[i] = 0  # In eclipse
    else:
        # Map time outside eclipse to [0, pi] for incidence angle
        if t < eclipse_start:
            angle = np.pi * (t / eclipse_start)
        else:
            angle = np.pi * ((t - eclipse_end) / (T - eclipse_end))
        power[i] = panel_efficiency * solar_irradiance * panel_area * np.cos(angle)
        power[i] = max(power[i], 0)  # No negative power

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(times / 60, power, label='Power Output (W)')
plt.axvspan(eclipse_start / 60, eclipse_end / 60, color='gray', alpha=0.3, label='Eclipse')
plt.title('Solar Panel Power Output Over One Orbit\n(700 km Sun-Synchronous Orbit)')
plt.xlabel('Time (minutes)')
plt.ylabel('Power (Watts)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
