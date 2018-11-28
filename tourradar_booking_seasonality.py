import pandas as pd
from numpy import log
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# read csv
raw = pd.read_csv('tourradar_bookings.csv', index_col='date')
raw.index = pd.to_datetime(raw.index)
raw.plot()

# resample data to be monthly
monthly = raw.resample(rule='M').sum()

# augmented Dickey-Fuller test. This dataset fails, even with log transformation, indicating that it is NON stationary.
result = adfuller(log(monthly['bookings']))
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#ETS decomposition to remove trend from data set
monthly_decomposed = seasonal_decompose(monthly['bookings'], model='multiplicative')
monthly_decomposed.plot()

# augmented dickey-fuller test now that trend has been removed, this now proves that the data is stationary.
result = adfuller(monthly_decomposed.seasonal)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#plot autocorrelation plot for seasonal data without trend
pd.plotting.autocorrelation_plot(monthly_decomposed.seasonal)
