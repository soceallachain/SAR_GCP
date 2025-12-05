#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 09:37:14 2025

@author: joe
"""

import pandas as pd
import numpy as np
import rasterio
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



src_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results'
csv_file = f'Train_ships_gt100m_with_visibility_and_orientation_annotation_1_120.csv'
#open csv file with statistics
csv_file_name = f'{src_dir}/{csv_file}'

# convert to dataframe
df =pd.read_csv(csv_file_name)
plt.style.use('dark_background')
#plot ship length v ship area
fig,ax = plt.subplots(figsize=(10,5))
#ax.set_facecolor('black')  # Set axes background
ax.plot(df['Ship Length'], df['Ship Area'],'s')
ax.set_xlabel('Ship Length (m)')
ax.set_ylabel('Ship Area (pixels)')
ax.set_title(f'Ship Area v Ship Length')

# Plot ship length v ship rcs

# plot ship length v ship area v ship visibility
colour_map = {0: 'red', 1: 'white', 2: 'blue', 3: 'green'}
colours = df['Signal Strength'].map(colour_map)
fig1,ax1 = plt.subplots(figsize=(10,5))
ax1.scatter( df['Ship Length'], df['Ship Area'], c=colours, marker='s', edgecolor='yellow')
ax1.set_ylabel('Ship Area (pixels)')
ax1.set_xlabel('Ship Length (m)')
ax1.set_title('Scatter Plot of Area vs Length with Visibility Colour Coded')
#Create custom legend
legend_elements = [
    Patch(facecolor='red',   edgecolor='black', label='C = 0'),
    Patch(facecolor='white', edgecolor='black', label='C = 1'),
    Patch(facecolor='blue',  edgecolor='black', label='C = 2'),
    Patch(facecolor='green', edgecolor='black', label='C = 3'),
]
ax1.legend(handles=legend_elements, title='Visibility / Colour', loc='best')

# plot ship length v ship rcs v ship visibility

#colour_map = {0: 'red', 1: 'white', 2: 'blue', 3: 'green'}
#colours = df['Signal Strength'].map(colour_map)
fig2,ax2 = plt.subplots(figsize=(10,5))
ax2.scatter( df['Ship Length'], df['Ship RCS'], c=colours, marker='s', 
               edgecolor='yellow')
ax2.set_ylabel('Ship RCS (A.U.)')
ax2.set_xlabel('Ship Length (m)')
ax2.set_title('Scatter Plot of RCS vs Length with Visibility Colour Coded')
#Create custom legend
legend_elements = [
    Patch(facecolor='red',   edgecolor='black', label='C = 0'),
    Patch(facecolor='white', edgecolor='black', label='C = 1'),
    Patch(facecolor='blue',  edgecolor='black', label='C = 2'),
    Patch(facecolor='green', edgecolor='black', label='C = 3'),
]
ax2.legend(handles=legend_elements, title='Visibility / Colour', loc='best')



# Calculate signal to clutter ratio
# what is signal ? mean value of ship pixels
# what is clutter? mean or std dev of non-ship pixels?
# Assign SNR to : (Signal - average Noise Level )/Std Dev Noise
# where Noise is background rcs pixels not in ship crop

SNR = (df['Ship Mean RCS'] - df['CWS Mean'])/df['CWS StdDev']
mean_noise = df['CWS Mean']
fig3,ax3 = plt.subplots(figsize=(10,5))
ax3.scatter( df['Signal Strength'], SNR)
ax3.set_ylabel('SNR (A.U.)')
ax3.set_xlabel('Visibility')
ax3.set_title('Scatter Plot of SNR vs Visibility ')


# Get average values of SNR for each visibility bin
# Plot histogram of SNR with 4 bins

bin_edges = [-0.5, 0.5, 1.5, 2.5, 3.5]
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 
               for i in range(len(bin_edges)-1)]
plt.figure()
plt.hist(df['Signal Strength'], bins=bin_edges, density=False, rwidth = 0.5, 
             alpha=0.6, color='g', label='Visibility'
         )
plt.xticks(bin_centers)
plt.legend()
plt.xlabel('Visibility')
plt.ylabel('Count')
plt.title(f"Histogram of Visibility Counts from {df['Signal Strength'].count()} Ships")


df['SNR'] = SNR
plt.figure()

# Group by column B and calculate mean of column A for each group
mean_values = df.groupby('Signal Strength')['SNR'].mean()

# Plot as a bar chart with B values on x-axis and means of A on y-axis
mean_values.plot(kind='bar', color='skyblue', edgecolor='black')

plt.xlabel('Visibility')
plt.ylabel('Mean SNR')
plt.title('Mean of SNR grouped by Visibility (0, 1, 2, 3)')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal

plt.show()


# Plot noise v visibnilty
plt.figure()

# Group by column B and calculate mean of column A for each group
mean_values = df.groupby('Signal Strength')['CWS Mean'].mean()

# Plot as a bar chart with B values on x-axis and means of A on y-axis
mean_values.plot(kind='bar', color='skyblue', edgecolor='black')

plt.xlabel('Visibility')
plt.ylabel('Mean Noise')
plt.title('Mean of Noise grouped by Visibility (0, 1, 2, 3)')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal

plt.show()



####
plt.figure()

# Group the data by column B and calculate statistics on A
grouped = df.groupby('Signal Strength')['SNR'].agg(['mean', 'min', 'max'])

# Prepare bar heights and error bars
means = grouped['mean']
lower_errors = means - grouped['min']
upper_errors = grouped['max'] - means
asymmetric_error = [lower_errors.values, upper_errors.values]

# Plot bar chart
fig, ax = plt.subplots()

bars = ax.bar(
    means.index, means.values,
    color='#00CED1',  # bright turquoise
    edgecolor='white',
    yerr=asymmetric_error,
    capsize=5,
    error_kw={'elinewidth':1.5, 'ecolor':'red'}
)
# Labels and title
ax.set_xlabel('Visibility', color='white')
ax.set_ylabel('Mean SNR',color='white')
ax.set_title('Mean of SNR grouped by Visibility with Min/Max Error Bars',color='white')

# Update tick color to white
ax.tick_params(colors='white')

# Optionally add grid with light lines for readability
ax.grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
plt.show()


# Calculate mean length and area of ships for different visibilities
# as histogram, add error bar as max and min 