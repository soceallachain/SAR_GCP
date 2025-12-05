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
csv_file = f'{Dataset}_ships_gt{ships_gt_size}m_with_visibility_and_orientation_annotation_1_120.csv'
#open csv file with statistics
csv_filename = f'{src_dir}/{csv_file}'

img_path=f'/home/joe/Work/Projects/RSAR/Test/SampleData/test_results/test_slc_raw_interferogram_results/{Dataset}_gt100m/Statistics'



# convert to dataframe
df =pd.read_csv(csv_filename)
# Generate SNR and save to dataframe
SNR = (df['Ship Mean RCS'] - df['CWS Mean'])/df['CWS StdDev']
mean_noise = df['CWS Mean']
df['SNR'] = SNR

#Rename Orientation column
df.rename(columns={'Orientation_wrt_range': 'Orientation'}, inplace=True)
#Drop rows where orientation not specified
df = df.dropna(subset=['Orientation'])


plt.style.use('dark_background')

bin_edges = [-7.5, 7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5]
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
    plot_fig_name = f'{Dataset}_ships_gt{ships_gt_size}_Orientation_Statistics{outlier_label}'
    
    # Plot Visibility Counts
    img_name  = f"Orientation_Statistics for {df['Signal Strength'].count()} Ships"
    #plot_hist_with_fit_on_ax(axs[0, 0], df['Signal Strength'].to_numpy(), bin_edges, 
    #                         0,'',img_name
    #                         )
    axs[0,0].hist(df['Orientation'].to_numpy(), bins=bin_edges, alpha=0.6, 
                     rwidth=0.9, color='g', edgecolor='black', label=img_name
                    )
    axs[0,0].set_xticks(bin_centers)
    axs[0,0].set_ylabel('Counts')
    axs[0,0].set_xlabel('Visibility')
    axs[0,0].set_title(img_name)
    
    
    img_name =f'Ship Area v Ship Length with Orientation Colour Coded'
    colour_map = {0: 'red', 15: 'white', 30: 'blue',45: 'green', 
                  60: 'yellow', 75: 'cyan',90: 'brown' }
    colours = df['Orientation'].map(colour_map)
    
    # Plot Ship Area v Length coded for Orientation
    axs[0,1].scatter( df['Ship Length'], df['Ship Area'], c=colours, marker='s', edgecolor='yellow')
    axs[0,1].set_ylabel('Ship Area (pixels)')
    axs[0,1].set_xlabel('Ship Length (m)')
    axs[0,1].set_title('Scatter Plot of Area vs Length with Orientation Colour Coded')
    #Create custom legend
    legend_elements = [
        Patch(facecolor='red',    edgecolor='black', label=r'$\theta$ = 0'),
        Patch(facecolor='white',  edgecolor='black', label=r'$\theta$ = 15'),
        Patch(facecolor='blue',   edgecolor='black', label=r'$\theta$ = 30'),
        Patch(facecolor='green',  edgecolor='black', label=r'$\theta$ = 45'),
        Patch(facecolor='yellow', edgecolor='black', label=r'$\theta$ = 60'),
        Patch(facecolor='cyan',   edgecolor='black', label=r'$\theta$ = 75'),
        Patch(facecolor='brown',  edgecolor='black', label=r'$\theta$ = 90'),
    ]
    axs[0,1].grid(True, color='gray', alpha=0.3, linestyle='--', axis='both')
    axs[0,1].legend(handles=legend_elements, title='Orientation / Colour', loc='best')
    
    
    # Plot Ship Radar Cross Section (Luminosity = Mean RCS * Area) v Length coded for Orientation
    axs[0,2].scatter( df['Ship Length'], df['Ship RCS'], c=colours, marker='s', 
                   edgecolor='yellow')
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
    
    axs[0,2].legend(handles=legend_elements, title='Orientation / Colour', loc='best')
    
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
    
    # Group the data by Orientation (signal strength) and calculate statistics
    # on SNR
    grouped = df.groupby('Orientation')['SNR'].agg(['mean', 'min', 
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
        width = 10,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,0].set_xlabel('Orientation', color='white')
    axs[1,0].set_ylabel('Mean SNR',color='white')
    axs[1,0].set_title('Mean of SNR grouped by Orientation with Min/Max Error Bars',color='white')
    
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
        width = 10,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,1].set_xlabel('Orientation', color='white')
    axs[1,1].set_ylabel('Mean SNR',color='white')
    axs[1,1].set_title('Mean of SNR grouped by Orientation with StdDev Error Bars',color='white')
    axs[1,1].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[1,1].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[1,1].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    
    
    
    #Plot Noise levels
    
    # Group by column B and calculate mean of column A for each group
    mean_Noise = df.groupby('Orientation')['CWS Mean'].mean()
    
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
        width = 10,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,2].set_xlabel('Orientation', color='white')
    axs[1,2].set_ylabel('Mean Noise',color='white')
    axs[1,2].set_title('Mean of Noise grouped by Orientation with StdDev Error Bars',color='white')
    axs[1,2].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[1,2].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[1,2].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    #Save image
    fig.savefig(f'{img_path}/{plot_fig_name}.jpg', format='jpg', 
             bbox_inches='tight', dpi=600
             )
    outliers =[15,16,40]
    remove_outliers = 1
    new_csv_filename=csv_file.replace('.csv',f'{outlier_label}.csv')
    df.to_csv(f"{img_path}/{new_csv_filename}")

# Save amended  dataframe as csv file

#Close all figures
fig.clf()
# Example: mass_arr and vol_arr are your arrays of N elements each
# M = number of orientation bins, V = number of visibility bins
M = 7  # for example
V = 4   # for example

num_ships=df['Orientation'].count()
plt.hist2d(df['Orientation'].to_numpy(), 
           df['Signal Strength'].to_numpy(), bins=[M, V], cmap='viridis')
plt.xlabel(fr'Orientation ($\theta$)')

plt.ylabel('Visibility')
plt.colorbar(label='Counts')
plt.title(f'{plot_fig_name}_{num_ships}_ships_heatmap.jpg')
plt.show()

plt.savefig(f'{img_path}/{plot_fig_name}_{num_ships}_ships_heatmap.jpg', format='jpg', dpi=300)

