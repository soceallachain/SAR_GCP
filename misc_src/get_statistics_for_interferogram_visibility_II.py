#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:12:38 2025

@author: joe
"""

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


from SAR_code_module import *


Dataset =f'Train'
ships_gt_size = 100
src_dir =f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results'
#csv_file = f'{Dataset}_RAW_IW2_ships_gt{ships_gt_size}m_Statistics_Blaise_annotation_only_26Aug2025.csv'
csv_file = f'{Dataset}_RAW_IW2_ships_gt{ships_gt_size}m_Statistics_Blaise_annotation_27Aug2025.csv'
#csv_file = f'{Dataset}_ships_gt{ships_gt_size}m_with_visibility_and_orientation_annotation_1_120.csv'
#open csv file with statistics
csv_filename = f'{src_dir}/{csv_file}'

img_path=f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m/Statistics'



# convert to dataframe
df =pd.read_csv(csv_filename)
# Generate SNR and save to dataframe
SNR = (df['Ship Mean RCS'] - df['CWS Mean'])/df['CWS StdDev']
mean_noise = df['CWS Mean']
df['SNR'] = SNR

plt.style.use('dark_background')

bin_edges = [-0.5, 0.5, 1.5, 2.5, 3.5]
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 
               for i in range(len(bin_edges)-1)]
outliers=[]
outlier_label =''
remove_outliers = 0
for i in range(0,2):
    if remove_outliers:
        # Remove rows with index labels 15 and 16
        df = df.drop(outliers)
        outlier_label ='_outliers_removed'
    
        
    
            
    fig, axs  = plt.subplots(2, 3, figsize=(16,9),constrained_layout=True)
    plot_fig_name = f'{Dataset}_ships_gt{ships_gt_size}_Statistics_BlaiseAnnotation_only_{get_date()}{outlier_label}'
    
    # Plot Visibility Counts
    img_name  = f"Visibility_Statistics for {df['Signal Strength'].count()} Ships"
    #plot_hist_with_fit_on_ax(axs[0, 0], df['Signal Strength'].to_numpy(), bin_edges, 
    #                         0,'',img_name
    #                         )
    axs[0,0].hist(df['Signal Strength'].to_numpy(), bins=bin_edges, alpha=0.6, 
                     rwidth=0.6, color='g', edgecolor='black', label=img_name
                    )
    axs[0,0].set_xticks(bin_centers)
    axs[0,0].set_ylabel('Counts')
    axs[0,0].set_xlabel('Visibility')
    axs[0,0].set_title(img_name)
    
    
    img_name =f'Ship Area v Ship Length with Visibility Colour Coded'
    colour_map = {0: 'red', 1: 'white', 2: 'blue', 3: 'green'}
    colours = df['Signal Strength'].map(colour_map)
    
    # Plot Ship Area v Length coded for Visibility
    axs[0,1].scatter( df['Ship Length'], df['Ship Area'], c=colours, marker='s', edgecolor='slategray')
    axs[0,1].set_ylabel('Ship Area (pixels)')
    axs[0,1].set_xlabel('Ship Length (m)')
    axs[0,1].set_title('Scatter Plot of Area vs Length with Visibility Colour Coded')
    #Create custom legend
    legend_elements = [
        Patch(facecolor='red',   edgecolor='black', label='C = 0'),
        Patch(facecolor='white', edgecolor='black', label='C = 1'),
        Patch(facecolor='blue',  edgecolor='black', label='C = 2'),
        Patch(facecolor='green', edgecolor='black', label='C = 3'),
    ]
    axs[0,1].grid(True, color='gray', alpha=0.3, linestyle='--', axis='both')
    axs[0,1].legend(handles=legend_elements, title='Visibility / Colour', loc='best')
    
    
    # Plot Ship Radar Cross Section (Luminosity = Mean RCS * Area) v Length coded for Visibility
    axs[0,2].scatter( df['Ship Length'], df['Ship RCS'], c=colours, marker='s', 
                   edgecolor='slategray')
    # Remove annoying exponent for y-axis on top of graph
    axs[0,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # Get the scalar formatter used by y axis
    formatter = axs[0,2].yaxis.get_major_formatter()
    # Get the order of magnitude (exponent), e.g. 6 for 1e6
    exponent = int(np.log10(df['Ship RCS'].max()))
    
    # Construct new y-label with the scale factor
    if exponent != 0:
        new_ylabel = f'Ship RCS x $10^{exponent}$ (A.U.)'
    else:
        new_ylabel = 'Ship RCS'
    
    # Disable offset text (the 10^6 label in corner)
    axs[0,2].get_yaxis().get_offset_text().set_visible(False)
    axs[0,2].set_ylabel(new_ylabel)
    axs[0,2].set_xlabel('Ship Length (m)')
    axs[0,2].set_title('Scatter Plot of RCS vs Length with Visibility Colour Coded')
    axs[0,2].grid(True, color='gray', alpha=0.3, linestyle='--', axis='both')
    
    axs[0,2].legend(handles=legend_elements, title='Visibility / Colour', loc='best')
    
    """
    # Plot SNR
    # Calculate signal to clutter ratio
    # what is signal ? mean value of ship pixels
    # what is clutter? mean or std dev of non-ship pixels?
    # Assign SNR to : (Signal - average [Noise Level] )/Std Dev [Noise Level]
    # where Noise is background rcs pixels not in ship crop
    
    mask = ~np.isnan(SNR)
    axs[1,0].scatter( df['Signal Strength'][mask], SNR[mask])
    axs[1,0].set_ylabel('SNR (A.U.)')
    axs[1,0].set_xlabel('Visibility')
    axs[1,0].set_title('Scatter Plot of SNR vs Visibility ')
    plt.show()
    
    """
    
    # Group the data by visibility (signal strength) and calculate statistics
    # on SNR
    grouped = df.groupby('Signal Strength')['SNR'].agg(['mean', 'min', 
                                                        'max','std']
                                                       )
    
    # Prepare bar heights and error bars
    means = grouped['mean']
    lower_errors = means - grouped['min']
    upper_errors = grouped['max'] - means
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[1,0].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,0].set_xlabel('Visibility', color='white')
    axs[1,0].set_ylabel('Mean SNR',color='white')
    axs[1,0].set_title('Mean of SNR grouped by Visibility with Min/Max Error Bars',color='white')
    
    # Update tick color to white
    axs[1,0].tick_params(colors='white')
    axs[1,0].set_xticks(bin_centers)
    # Optionally add grid with light lines for readability
    axs[1,0].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    # Plot SNR with std dev
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[1,1].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,1].set_xlabel('Visibility', color='white')
    axs[1,1].set_ylabel('Mean SNR',color='white')
    axs[1,1].set_title('Mean of SNR grouped by Visibility with StdDev Error Bars',color='white')
    axs[1,1].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[1,1].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[1,1].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    
    
    
    #Plot Noise levels
    
    # Group by column B and calculate mean of column A for each group
    mean_Noise = df.groupby('Signal Strength')['CWS Mean'].mean()
    
    # Plot as a bar chart with B values on x-axis and means of A on y-axis
    #mean_Noise.plot(kind='bar', color='skyblue', edgecolor='black')
    
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    bars = axs[1,2].bar(
        mean_Noise.index, mean_Noise.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,2].set_xlabel('Visibility', color='white')
    axs[1,2].set_ylabel('Mean Noise',color='white')
    axs[1,2].set_title('Mean of Noise grouped by Visibility with StdDev Error Bars',color='white')
    axs[1,2].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[1,2].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[1,2].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    #Save image
    fig.savefig(f'{img_path}/{plot_fig_name}.jpg', format='jpg', 
             bbox_inches='tight', dpi=600
             )
    outliers =[12,15,16,40]
    remove_outliers = 1
    new_csv_filename=csv_file.replace('.csv',f'{outlier_label}.csv')
    df.to_csv(f"{img_path}/{new_csv_filename}")

# Save amended  dataframe as csv file
"""

#######################################################
# Get plot of visibility v ship length
   
fig, axs  = plt.subplots(1, 2, figsize=(16,9),constrained_layout=True)
plot_fig_name = f'{Dataset}_ships_gt{ships_gt_size}_Statistics_BlaiseAnnotation_only_{get_date}{outlier_label}'

# Plot Visibility Counts
img_name  = f"Visibility_Statistics for {df['Signal Strength'].count()} Ships"
#plot_hist_with_fit_on_ax(axs[0, 0], df['Signal Strength'].to_numpy(), bin_edges, 
#                         0,'',img_name
#                         )
axs[0].hist(df['Signal Strength'].to_numpy(), bins=bin_edges, alpha=0.6, 
                 rwidth=0.6, color='g', edgecolor='black', label=img_name
                )
axs[0].set_xticks(bin_centers)
axs[0].set_ylabel('Counts')
axs[0].set_xlabel('Visibility')
axs[0].set_title(img_name)


img_name =f'Ship Area v Ship Length with Visibility Colour Coded'
colour_map = {0: 'red', 1: 'orange', 2: 'blue', 3: 'green'}
colours = df['Signal Strength'].map(colour_map)

# Plot Ship Area v Length coded for Visibility
axs[1].scatter( df['Ship Length'], df['Signal Strength'],  c=colours, marker='s', edgecolor='white')
axs[1].set_xlabel('Ship Length (m)')
axs[1].set_ylabel('Visibility')
axs[1].set_title('Scatter Plot of Visibility v Area vs Length with Visibility Colour Coded')


#################
#Find mean value of ship length and area for each visibility class
signal_values = sorted(df['Signal Strength'].unique())
means = []
stds = []
mins = []
maxs = []

for val in signal_values:
    ship_lengths = df.loc[df['Signal Strength'] == val, 'Ship Length']
    means.append(ship_lengths.mean())
    stds.append(ship_lengths.std())
    mins.append(ship_lengths.min())
    maxs.append(ship_lengths.max())
cap_width = 0.25  # How wide the horizontal cap lines should be
plt.figure(figsize=(8,6))
width = 0.5  # box width
for i, val in enumerate(signal_values):
    mean = means[i]
    std = stds[i]
    min_val = mins[i]
    max_val = maxs[i]
    # Main vertical error bar
    plt.plot([val, val], [min_val, max_val], color='blue', linewidth=2)
    # Top and bottom caps
    plt.plot([val - cap_width/2, val + cap_width/2], [min_val, min_val], color='blue', linewidth=2)
    plt.plot([val - cap_width/2, val + cap_width/2], [max_val, max_val], color='blue', linewidth=2)
    # The mean marker and box stay the same as before
    plt.gca().add_patch(
        plt.Rectangle((val - width/2, mean - std), width, 2*std, fill=True, color='green', alpha=0.7)
    )
    plt.plot(val, mean, 'o', color='red', markersize=8)


plt.xticks(signal_values)
plt.xlabel('Visibility')              # Set x-axis label
plt.ylabel('Ship Length (m)')         # Set y-axis label
plt.title(r'Ship Length (Mean $\pm\,1\sigma$) v Visibility with Min-Max Error bars')

"""


#################
signal_values = sorted(df['Signal Strength'].unique())
means = []
stds = []
mins = []
maxs = []
fig, axs  = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
plot_fig_name = f'{Dataset}_ships_gt{ships_gt_size}_Visibility_v_ShipLength_BlaiseAnnotation_only_26Aug2025{outlier_label}'

# Plot Visibility Counts on axs[0]
img_name  = f"Visibility_Statistics for {df['Signal Strength'].count()} Ships"
axs[0].hist(df['Signal Strength'].to_numpy(), bins=bin_edges, alpha=0.6, 
            rwidth=0.6, color='g', edgecolor='black', label=img_name)
axs[0].set_xticks(bin_centers)
axs[0].set_ylabel('Counts')
axs[0].set_xlabel('Visibility')
axs[0].set_title(img_name)

# Plot Ship Area v Length coded for Visibility on axs[1]
colour_map = {0: 'red', 1: 'white', 2: 'blue', 3: 'green'}
colours = df['Signal Strength'].map(colour_map)
axs[1].scatter(df['Ship Length'], df['Signal Strength'], c=colours, marker='s', edgecolor='white')
axs[1].set_xlabel('Ship Length (m)')
axs[1].set_ylabel('Visibility')
axs[1].set_title('Visibility v Ship Length')
axs[1].set_yticks(signal_values)

# Calculate mean, std, min, max for Ship Length per Signal Strength for the box plot
signal_values = sorted(df['Signal Strength'].unique())
means, stds, mins, maxs = [], [], [], []
for val in signal_values:
    ship_lengths = df.loc[df['Signal Strength'] == val, 'Ship Length']
    means.append(ship_lengths.mean())
    stds.append(ship_lengths.std())
    mins.append(ship_lengths.min())
    maxs.append(ship_lengths.max())

# Plot the custom box plot with error bars on axs[2]
cap_width = 0.25  # Horizontal cap width
width = 0.5       # Box width

for i, val in enumerate(signal_values):
    mean = means[i]
    std = stds[i]
    min_val = mins[i]
    max_val = maxs[i]
    # Vertical error bar line
    axs[2].plot([val, val], [min_val, max_val], color='blue', linewidth=2)
    # Caps on error bars
    axs[2].plot([val - cap_width/2, val + cap_width/2], [min_val, min_val], color='blue', linewidth=2)
    axs[2].plot([val - cap_width/2, val + cap_width/2], [max_val, max_val], color='blue', linewidth=2)
    # Box centered on mean with 2*std height
    from matplotlib.patches import Rectangle
    rect = Rectangle((val - width/2, mean - std), width, 2*std, fill=True, color='green', alpha=0.7)
    axs[2].add_patch(rect)
    # Mean marker
    axs[2].plot(val, mean, 'o', color='red', markersize=8)

axs[2].set_xticks(signal_values)
axs[2].set_xlabel('Visibility')
axs[2].set_ylabel('Ship Length (m)')
axs[2].set_title(r'Ship Length (Mean $\pm\,1\sigma$) v Visibility with Min-Max Error bars')

plt.show()
fig.savefig(f'{img_path}/{plot_fig_name}_{len(means)}_Ships_{get_date()}.jpg', format='jpg', 
         bbox_inches='tight', dpi=600
         )





##### Plot table

# Prepare table data as list of lists (rows), including Visibility as first column
table_data = []
for i in range(len(means)):
    row = [f"{int(signal_values[i])}",
           f"{int(means[i])}", 
           f"{int(stds[i])}",
           f"{int(mins[i])}", 
           f"{int(maxs[i])}"]

    table_data.append(row)
    
   

# Column labels including Visibility
columns = ['Visibility', 'Mean (m)', 'Std Dev (m)', 'Min (m)', 'Max (m)']

fig, ax = plt.subplots()
ax.axis('off')  # Hide axes

# Create table with the updated data
table = ax.table(cellText=table_data,
                 colLabels=columns,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)  # scale width, height


# Set font color to green for all cells (including header)
for key, cell in table.get_celld().items():
    cell.get_text().set_color('black')


plt.show()
