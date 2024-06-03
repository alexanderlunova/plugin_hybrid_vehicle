import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import norm

file_path = r"C:\Users\alexl\Downloads\emobpy_timeseries_original\emobpy_timeseries_original.csv"

dtype = {
    'ID': str,
    'Distance_km': float,
    'Consumption_kWh': float,
    'PowerRating_kW': float,
    'Load_kW': float,
    'SoC': float,
    'Load_kW.1': float,
    'SoC.1': float,
    'Load_kW.2': float,
    'SoC.2': float,
    'Load_kW.3': float,
    'SoC.3': float
}

# Define a date parser function matching your date format
date_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# Read the CSV file with specified dtypes and date parser
df = pd.read_csv(file_path, dtype=dtype, low_memory=False, skiprows=1)

# Create a dictionary to hold DataFrames for each year
driving_profiles = {}

# Group by year and create a DataFrame for each year
for id, group in df.groupby('ID'):
    driving_profiles[id] = group.reset_index(drop=True)

data = driving_profiles["1.0"]
ts_load = df['Load_kW']
ts_load = ts_load[ts_load != 0]
mask = ~np.isnan(ts_load)
ts_load = ts_load[mask]

# Fit a normal distribution to the data
mu, sigma = norm.fit(ts_load)

# Plot the histogram of the data
plt.hist(ts_load, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')

# Plot the PDF of the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)

# Add title and labels
plt.title('Histogram with Fitted Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')

plt.show()

print("")