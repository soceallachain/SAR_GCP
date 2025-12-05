import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data (replace this with your actual data)
#N = 1000
#data = np.random.randn(N)  # Example: standard normal distribution
data=pixel_values[(pixel_values>0) & (pixel_values<pixel_values.max()/2 )]
# Fit distributions
rayleigh_params = stats.rayleigh.fit(data)
chi2_params = stats.chi2.fit(data)
weibull_params = stats.weibull_min.fit(data)

# Create a range of x values for plotting
x = np.linspace(min(data), max(data), 100)

# Plot histogram and fitted distributions
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', label='Data')
plt.plot(x, stats.rayleigh.pdf(x, *rayleigh_params), 'r-', lw=2, label='Rayleigh')
plt.plot(x, stats.chi2.pdf(x, *chi2_params), 'g-', lw=2, label='Chi-squared')
plt.plot(x, stats.weibull_min.pdf(x, *weibull_params), 'b-', lw=2, label='Weibull')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with Fitted Distributions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Generate sample data (replace with your actual data)

# Method of Moments
sample_mean = np.mean(data)
sample_variance = np.var(data, ddof=1)
alpha_mom = sample_mean**2 / sample_variance
beta_mom = sample_variance / sample_mean

# Maximum Likelihood Estimation (MLE)
alpha_mle, loc_mle, beta_mle = gamma.fit(data, floc=0)  # Fix location to 0

# Create x values for plotting
x = np.linspace(min(data), 1000, 100)

# Plot histogram and fitted Gamma distributions
plt.figure(figsize=(10, 6))
plt.hist(data, bins=1000, density=True, alpha=0.6, color='skyblue', label='Data')
plt.plot(x, gamma.pdf(x, alpha_mom, loc=0, scale=beta_mom), 'r-', lw=2, label='Gamma (Moments)')
plt.plot(x, gamma.pdf(x, alpha_mle, loc_mle, beta_mle), 'g--', lw=2, label='Gamma (MLE)')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with Fitted Gamma Distributions')
plt.legend()
plt.grid(True)
plt.show()
