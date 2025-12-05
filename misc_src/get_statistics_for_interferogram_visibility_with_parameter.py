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


sar_data_type =f'SLC'
df['CV Burst I'] = df[f'Burst {sar_data_type} I StDev'] /df[f'Burst {sar_data_type} I Mean']
df['CV Burst Q'] = df[f'Burst {sar_data_type} Q StDev'] /df[f'Burst {sar_data_type} Q Mean']
df['CV Crop I']  = df[f'Crop {sar_data_type} I StDev']  /df[f'Crop {sar_data_type} I Mean']
df['CV Crop Q']  = df[f'Crop {sar_data_type} Q StDev']  /df[f'Crop {sar_data_type} Q Mean']




plt.style.use('dark_background')

bin_edges = [-0.5, 0.5, 1.5, 2.5, 3.5]
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 
               for i in range(len(bin_edges)-1)]


outliers=[]
outlier_label =''
remove_outliers = 0
for i in range(0,1):
    if remove_outliers:
        # Remove rows with index labels 15 and 16
        df = df.drop(outliers)
        outlier_label ='_outliers_removed'
    
    fig, axs  = plt.subplots(3, 4, figsize=(16,9),constrained_layout=True)
    plot_fig_name = f'{Dataset}_ships_gt{ships_gt_size}_Statistics_{sar_data_type}_I_Q{outlier_label}'
    fig.suptitle(plot_fig_name)
    
    # Plot Visibility Counts
    img_name  = f"Visibility_Statistics for {df['Signal Strength'].count()} Ships"
    #plot_hist_with_fit_on_ax(axs[0, 0], df['Signal Strength'].to_numpy(), bin_edges, 
    #                         0,'',img_name
    #                         )

    
    # Group the data by visibility (signal strength) and calculate statistics
    # on SNR
    grouped = df.groupby('Signal Strength')[f'Burst {sar_data_type} I Mean'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )    
    # Prepare bar heights and error bars
    means = grouped['mean']
      
    # Update tick color to white
    axs[0,0].tick_params(colors='white')
    axs[0,0].set_xticks(bin_centers)
    # Optionally add grid with light lines for readability
    axs[0,0].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    # Plot  with std dev
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[0,0].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[0,0].set_xlabel('Visibility', color='white')
    axs[0,0].set_ylabel('Mean Burst I',color='white')
    axs[0,0].set_title('Mean of Burst I StdDev Error Bars',
                       color='white'
                        )
    axs[0,0].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[0,0].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[0,0].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    
    grouped = df.groupby('Signal Strength')[f'Burst {sar_data_type} Q Mean'].agg(
                                         ['mean', 'min', 'max','std']
                                                                    )
    # Prepare bar heights and error bars
    means = grouped['mean']
    # Plot  with std dev
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[0,1].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[0,1].set_xlabel('Visibility', color='white')
    axs[0,1].set_ylabel('Burst Q Mean ',color='white')
    axs[0,1].set_title('Mean of Burst Q StdDev Error Bars',
                       color='white'
                       )
  
    axs[0,1].set_xticks(bin_centers)    
    axs[0,1].tick_params(colors='white')
    axs[0,1].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    ## Process Burst Raw I StDev  
    grouped = df.groupby('Signal Strength')[f'Burst {sar_data_type} I StDev'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )
    # Prepare bar heights and std deverror bars
    means = grouped['mean']
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[0,2].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[0,2].set_xlabel('Visibility', color='white')
    axs[0,2].set_ylabel('Mean Burst I StdDev ',color='white')
    axs[0,2].set_title('Mean of Burst I StdDev StDevError Bars',
                       color='white'
                       )
    axs[0,2].tick_params(colors='white')
    axs[0,2].set_xticks(bin_centers)
    axs[0,2].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    # Plot  with std dev
    ## Process Burst Raw I StDev  
    grouped = df.groupby('Signal Strength')[f'Burst {sar_data_type} Q StDev'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )
    # Prepare bar heights and std deverror bars
    means = grouped['mean']
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[0,3].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[0,3].set_xlabel('Visibility', color='white')
    axs[0,3].set_ylabel('Mean Burst Q StDev',color='white')
    axs[0,3].set_title('Mean of Burst Q StdDev  StDev Error Bars',
                       color='white'
                        )
    axs[0,3].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[0,3].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[0,3].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    
    #Plot Crop Data
    # Group the data by visibility (signal strength) and calculate statistics
    # on SNR
    grouped = df.groupby('Signal Strength')[f'Crop {sar_data_type} I Mean'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )    
    # Prepare bar heights and error bars
    means = grouped['mean']
      
    # Update tick color to white
    axs[1,0].tick_params(colors='white')
    axs[1,0].set_xticks(bin_centers)
    # Optionally add grid with light lines for readability
    axs[1,0].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    # Plot  with std dev
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
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
    axs[1,0].set_ylabel('Mean Crop I',color='white')
    axs[1,0].set_title('Mean of Crop I StdDev Error Bars',
                       color='white'
                        )
    axs[1,0].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[1,0].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[1,0].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    
    grouped = df.groupby('Signal Strength')[f'Crop {sar_data_type} Q Mean'].agg(
                                         ['mean', 'min', 'max','std']
                                                                    )
    # Prepare bar heights and error bars
    means = grouped['mean']
    # Plot  with std dev
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
    axs[1,1].set_ylabel('Crop Q Mean ',color='white')
    axs[1,1].set_title('Mean of Crop Q StdDev Error Bars',
                       color='white'
                       )
  
    axs[1,1].set_xticks(bin_centers)    
    axs[1,1].tick_params(colors='white')
    axs[1,1].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    ## Process Burst Raw I StDev  
    grouped = df.groupby('Signal Strength')[f'Crop {sar_data_type} I StDev'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )
    # Prepare bar heights and std deverror bars
    means = grouped['mean']
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[1,2].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,2].set_xlabel('Visibility', color='white')
    axs[1,2].set_ylabel('Mean Crop I StdDev ',color='white')
    axs[1,2].set_title('Mean of Crop I StdDev StDev Error Bars',
                       color='white'
                       )
    axs[1,2].tick_params(colors='white')
    axs[1,2].set_xticks(bin_centers)
    axs[1,2].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    # Plot  with std dev
    ## Process Burst Raw I StDev  
    grouped = df.groupby('Signal Strength')[f'Crop {sar_data_type} Q StDev'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )
    # Prepare bar heights and std deverror bars
    means = grouped['mean']
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[1,3].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[1,3].set_xlabel('Visibility', color='white')
    axs[1,3].set_ylabel('Mean Crop Q StDev',color='white')
    axs[1,3].set_title('Mean of Crop Q StdDev  StDev Error Bars',
                       color='white'
                        )
    axs[1,3].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[1,3].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[1,3].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    

    #Plot CV Data
    # Group the data by visibility (signal strength) and calculate statistics
    # on SNR
    grouped = df.groupby('Signal Strength')['CV Burst I'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )    
    # Prepare bar heights and error bars
    means = grouped['mean']
      
    # Update tick color to white
    axs[2,0].tick_params(colors='white')
    axs[2,0].set_xticks(bin_centers)
    # Optionally add grid with light lines for readability
    axs[2,0].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    # Plot  with std dev
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[2,0].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[2,0].set_xlabel('Visibility', color='white')
    axs[2,0].set_ylabel('CV Burst I',color='white')
    axs[2,0].set_title('CV Burst I  StdDev Error Bars',
                       color='white'
                        )
    axs[2,0].set_xticks(bin_centers)
    # Update tick color to white
    axs[2,0].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[2,0].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    
    grouped = df.groupby('Signal Strength')['CV Burst Q'].agg(
                                         ['mean', 'min', 'max','std']
                                                                    )
    # Prepare bar heights and error bars
    means = grouped['mean']
    # Plot  with std dev
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[2,1].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[2,1].set_xlabel('Visibility', color='white')
    axs[2,1].set_ylabel('CV Burst Q ',color='white')
    axs[2,1].set_title('CV Burst Q StdDev Error Bars',
                       color='white'
                       )
  
    axs[2,1].set_xticks(bin_centers)    
    axs[2,1].tick_params(colors='white')
    axs[2,1].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    ## Process Burst Raw I StDev  
    grouped = df.groupby('Signal Strength')['CV Crop I'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )
    # Prepare bar heights and std deverror bars
    means = grouped['mean']
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[2,2].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[2,2].set_xlabel('Visibility', color='white')
    axs[2,2].set_ylabel('CV Crop I ',color='white')
    axs[2,2].set_title('CV Crop I StDev Error Bars',
                       color='white'
                       )
    axs[2,2].tick_params(colors='white')
    axs[2,2].set_xticks(bin_centers)
    axs[2,2].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')
    
    # Plot  with std dev
    ## Process Burst Raw I StDev  
    grouped = df.groupby('Signal Strength')['CV Crop Q'].agg(
                                           ['mean', 'min', 'max','std']
                                                                     )
    # Prepare bar heights and std deverror bars
    means = grouped['mean']
    lower_errors = grouped['std']
    upper_errors = grouped['std'] 
    asymmetric_error = [lower_errors.values, upper_errors.values]
    
    # Plot bar chart
    bars = axs[2,3].bar(
        means.index, means.values,
        color='#00CED1',  # bright turquoise
        edgecolor='white',
        yerr=asymmetric_error,
        capsize=5,
        error_kw={'elinewidth':1.5, 'ecolor':'red'}
    )
    # Labels and title
    axs[2,3].set_xlabel('Visibility', color='white')
    axs[2,3].set_ylabel('CV Crop Q',color='white')
    axs[2,3].set_title('CV Crop Q  StDev Error Bars',
                       color='white'
                        )
    axs[2,3].set_xticks(bin_centers)
    
    # Update tick color to white
    axs[2,3].tick_params(colors='white')
    
    # Optionally add grid with light lines for readability
    axs[2,3].grid(True, color='gray', alpha=0.3, linestyle='--', axis='y')




    #Save image
    fig.savefig(f'{img_path}/{plot_fig_name}.jpg', format='jpg', 
             bbox_inches='tight', dpi=600
             )
    outliers =[15,16,40]
    remove_outliers = 1
    new_csv_filename=csv_file.replace('.csv',f'{outlier_label}.csv')
    df.to_csv(f"{img_path}/{new_csv_filename}")

# Save amended  dataframe as csv file


