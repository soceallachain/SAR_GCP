#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:49:13 2025

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt

file= '/home/joe/tmp/strong_signal_ships_in_bursts.txt'

#read file and and store both columns
col1, col2 = np.loadtxt(file, unpack=True)

print("Column 1:", col1)
print("Column 2:", col2)
#plot histogram of column 1
counts, bin_edges = np.histogram(col1, bins=20)


fig1, ax1 = plt.subplots()
ax1.hist(col1, bins=bin_edges, edgecolor='black')
ax1.set_xticks(bin_edges)
ax1.set_xticklabels([str(round(edge, 2)) for edge in bin_edges], rotation=45)
ax1.set_xlabel('Azimuth Bins')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Azimuth Locations')
plt.tight_layout()
plt.show()

#plot histogram of column 2

counts, bin_edges = np.histogram(col2, bins=20)



# Set x-axis ticks to show bin edges (range of each bin)
fig2, ax2 = plt.subplots()
ax2.hist(col2, bins=bin_edges, edgecolor='black')
ax2.set_xticks(bin_edges)
ax2.set_xticklabels([str(round(edge, 2)) for edge in bin_edges], rotation=45)
ax2.set_xlabel('Range Bins')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of Range Locations')
plt.tight_layout()
plt.show()




import numpy as np
from scipy import stats

# Example data (replace with your actual dataset)
#data = np.random.uniform(low=0.0, high=1.0, size=100)
data = col1
# Parameters of the uniform distribution
# If you know the expected range, specify loc and scale accordingly
loc = np.min(data)
scale = np.max(data) - np.min(data)

# Perform the KS test against uniform distribution on [loc, loc+scale]
ks_statistic, p_value = stats.kstest(data, 'uniform', args=(loc, scale))

print(f"KS statistic: {ks_statistic}")
print(f"P-value: {p_value}")

alpha = 0.05  # significance level

if p_value > alpha:
    print("Fail to reject the null hypothesis: data is consistent with a uniform distribution.")
else:
    print("Reject the null hypothesis: data is NOT consistent with a uniform distribution.")



import numpy as np
from scipy.stats import uniform, norm
# Use Bayesian inference to determine liklihood of being uniform
# Example data: replace with your dataset
#data = np.random.uniform(0, 1, 100)

# Hypothesis 1: Data is uniform on [a, b]
a, b = np.min(data), np.max(data)
likelihood_uniform = np.prod(uniform.pdf(data, loc=a, scale=b-a))

# Hypothesis 2: Data is normal (alternative)
mu, sigma = np.mean(data), np.std(data, ddof=1)
likelihood_normal = np.prod(norm.pdf(data, loc=mu, scale=sigma))

# Prior probabilities (uninformative)
prior_uniform = 0.5
prior_normal = 0.5

# Compute unnormalized posteriors
posterior_uniform_unnorm = likelihood_uniform * prior_uniform
posterior_normal_unnorm = likelihood_normal * prior_normal

# Normalize to get posterior probabilities
evidence = posterior_uniform_unnorm + posterior_normal_unnorm
posterior_uniform = posterior_uniform_unnorm / evidence
posterior_normal = posterior_normal_unnorm / evidence

print(f"Posterior probability data is uniform: {posterior_uniform:.4f}")
print(f"Posterior probability data is normal: {posterior_normal:.4f}")

if posterior_uniform > posterior_normal:
    print("Bayesian test favors the uniform hypothesis.")
else:
    print("Bayesian test favors the non-uniform (normal) hypothesis.")
    
    

import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt

# Example: replace this with your actual data array


# Number of bins for the histogram
num_bins = 10

# Compute observed frequencies by binning data
observed_freq, bin_edges = np.histogram(data, bins=num_bins)

# Expected frequencies assuming uniform distribution
expected_freq = np.full(num_bins, len(data) / num_bins)

# Perform Chi-square goodness-of-fit test
chi_stat, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)

print(f"Chi-square statistic: {chi_stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05  # significance level

if p_value > alpha:
    print("Fail to reject null hypothesis: data is consistent with a uniform distribution.")
else:
    print("Reject null hypothesis: data is NOT consistent with a uniform distribution.")
    
    
    

file= '/home/joe/tmp/all_ships_in_public.txt'

#read file and and store both columns
col1, col2 = np.loadtxt(file, unpack=True)

print("Column 1:", col1)
print("Column 2:", col2)
#plot histogram of column 1
counts, bin_edges = np.histogram(col1, bins=20)


fig1, ax1 = plt.subplots()
ax1.hist(col1, bins=bin_edges, edgecolor='black')
ax1.set_xticks(bin_edges)
ax1.set_xticklabels([str(round(edge, 2)) for edge in bin_edges], rotation=45)
ax1.set_xlabel('Azimuth Bins')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of All Ships in Public - Azimuth Locations')
plt.tight_layout()
plt.show()

#plot histogram of column 2

counts, bin_edges = np.histogram(col2, bins=20)



# Set x-axis ticks to show bin edges (range of each bin)
fig2, ax2 = plt.subplots()
ax2.hist(col2, bins=bin_edges, edgecolor='black')
ax2.set_xticks(bin_edges)
ax2.set_xticklabels([str(round(edge, 2)) for edge in bin_edges], rotation=45)
ax2.set_xlabel('Range Bins')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of All Ships in Public - Range Locations')
plt.tight_layout()
plt.show()
