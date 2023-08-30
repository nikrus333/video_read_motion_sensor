import pandas as pd
import time
import scipy.stats as stats
import numpy as np

df = pd.read_csv('value_axises.csv')
print(df.describe())
df.x_axes_data.plot(kind='hist', density=1, bins=20, stacked=False, alpha=.5, color='grey')
df.plot()
z = np.abs(stats.zscore(df))

#only keep rows in dataframe with all z-scores less than absolute value of 3 
data_clean = df[(z<3).all(axis=1)]

#find how many rows are left in the dataframe 

df.to_csv('value_axises_filter.csv')