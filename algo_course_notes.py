import numpy as np

#turn a list into an array
mylist = [1, 2, 3]
x = np.array(mylist)
type(x)

#list of lists into a matrix
#can add another layer of brackets to add further dimensions
mymatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y = np.array(mymatrix)

#both work the same (step size can be added as well)
list(range(0,5))
np.arange(0,5)

#creates array of zeros as floats
np.zeros(3)
#creates a matrix of zeros
np.zeros((3,5))

#return n evenly spaced numbers between range
np.linspace(0,10,3)

#returns square matrix of zeros with diagonal of ones
#this is called an identity matrix
np.eye(4)

#lots of random methods in np- notice how a matrix is passed in. can be a singel number (array)
np.random.rand(5,4)

#random from a standard normal distribution
np.random.randn(1)

#random integers
np.random.randint(1,100)

arr = np.arange(25)
ranarr = np.random.randint(0, 50, 10)
arr

#reshape method returns the exact same values, but in a new shape
#this reshapes the arr variable above
arr.reshape(5,5)

#call shape method to check the shape of the data
arr.shape
arr.reshape(5,5).shape

#use max and min to find max and in in np, argmin and argmax return index of min/max
arr.max()
arr.argmax()

#can perform arithmetic. this adds/mults based on index location
arr + arr
arr*arr
arr ** 3

np.sqrt(arr)
np.log(arr)
np.sin(arr)

#indexing and slicing
arr[8]
arr[1:7]
arr[3:]

#broadcasting, permanent change
arr[0:5] = 100 #changes these values to 100
arr

#can take a slice of an array, then broadcast. 
#if changing a slice of an array, the original is changed as well
#in the case you want to change only a copy, use:
arr2 = arr.copy() 

#indexing matrices
mat = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
#row 0
mat[0]
#row 1, col 1 individual value
mat[1][1] #or
mat[1,1]
#take a small square of values from matrix
mat[:2, 1:]

#conditional selection
arr = np.arange(1,11)
arr > 4
boo_arr = arr > 4
#pass the conditional in to return only values > 4
arr[boo_arr]
#or
arr[arr>4]
arr[arr<=9]

#sum rows or columns
mat.sum(axis=1)
mat.sum(axis=0)
#sum all
mat.sum()


#*****PANDAS*****
import pandas as pd

labels = ['a', 'b', 'c']
mylist = [10, 20, 30]
arr = np.array([10,20,30])
d = {'a': 10, 'b': 20, 'c': 100}

#nearly the same as an np array, but has index labels
pd.Series(mylist, index=labels)
pd.Series(d)
pd.Series(data=labels)

#create series and return value from index label
ser1 = pd.Series([1, 2, 3, 4], index=['USA', 'CHINA', 'FR', 'DE'])
ser1
ser1['USA']
ser2 = pd.Series([1, 2, 3, 4], index=['USA', 'ITALY', 'FR', 'JAPAN'])
#adds where there is a match between series
ser1 + ser2

# dataframes

from numpy.random import randn

np.random.seed(101)
df = pd.DataFrame(randn(5,4), ['a', 'b', 'c', 'd', 'e'], ['w', 'x', 'y', 'z'])
df

#this is a series: use the bracket notation 
df['w']
df.w
#return multiple columns
df[['w', 'z']]

#create new column
df['new'] = df['w'] + df['y']
df

#drop column: must specifiy axis. add inplace to actually remove from df
df.drop('new', axis=1, inplace=True)

#selecting rows
#using labels indexes
df.loc['b']
#using index locations
df.iloc[1]

#subsets
df.loc['b', 'y']
df.loc[['a','b'], ['w', 'y']]

#conditional selection
df > 0
booldf = df > 0
df[booldf]
df[df > 0]
df['w'] > 0
df[df['w']>0]
df
#selects only the rows which have values in z column < 0
df[df['z']< 0]
resultdf = df[df['z']< 0]
resultdf['x']
#returns the x column from the set of rows where z < 0 all in one step
df[df['z']< 0]['x']
#returns two columns
df[df['z']< 0][['y','x']]

#multiple conditions, must use ampersand
# use pipe | for or condition
df[(df['w'] > 0) & (df['y'] > 1)]

#reset index, doesn't occur in place unless inplace argument is passed
df.reset_index()

#easy way to create a list quickly without commas and apostrophes
newindex = 'CA UT WY OR NJ'.split()
df['states'] = newindex
df

df.set_index('states')

#multiindex
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

df = pd.DataFrame(randn(6,2), hier_index, ['A','B'])
df
#call from outside index
df.loc['G1']
#then call another level deeper
df.loc['G1'].loc[1]

#indexes have no names
df.index.names
#let's change that
df.index.names = ['groups', 'num']

#cross section
df.xs('G1')
#get all rows from the index 1, regardless of group
df.xs(1, level='num')

#groupby, groups by column and aggregates
# df.groupby('Company').sum().loc['FB'] returns total sales for FB e.g.
# df.groupby('Company').describe() gives count, mean, max, percentiles

#pd.concat([list of dfs]) simply gllues dfs together
#pd.merge is like joining sql tables, a key is used
#pd.join joins dfs using index as keys

#unique values can be found by df.unique()

#apply method- this applies a function to a column

def times2(x):
    return x*2
df['w'].apply(times2)
df['w'].apply(len)
df['w'].apply(lambda x: x*2)

#return list of column names
df.columns

#sorting
df.sort_values('y')

#find nulls
df.isnull()

#read data with pandas
json = pd.read_json('example_2.json')
html = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html')
html[0]

#matplotlib
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
y = x**2

#subplots
plt.subplot(1,2,1)
plt.plot(x,y)
plt.subplot(1,2,2)
plt.plot(y,x)

#OO plotting
fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
axes1.plot(x,y)
axes1.set_title('main plot')
axes2.plot(y,x)
axes2.set_title('inner plot')


fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x,y)
axes[0].plot(x,y)
plt.tight_layout()

#size and dpi, figsize in inches
fig = plt.figure(figsize=(6,4))
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

#save fig to file
fig.savefig('mypic.jpg')

#plot customization
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, x, label = 'x', color = 'red') #can use RGB HEX codes too
ax.plot(x, x**2, label = 'x sq', linewidth=5) #default linewidth is 1
ax.plot(x,x**3, label = 'x cubed', alpha=0.5) #alpha is transparency
ax.plot(x,x**4, label = 'x^4', linestyle='--') #check doc for other styles
ax.plot(x,x**5, label = 'x^5', marker='o', markersize=1) #check doc for other styles
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.legend(loc = 0) #see documentation for locations
plt.tight_layout()

#plotting with pandas
data = pd.read_csv('forex_aug_18_all.csv', index_col=0)
data['close'].hist(bins=30)
#make pretty with seaborn
plt.style.use('seaborn')
#more examples- can use matplotlib arguments with all of these
data['close'].hist(bins=50)
data['close'].plot(kind='bar') #or
data['close'].plot.hist()
data.plot.area()
df.plot.bar()
df.plot.bar(stacked=True)
data.plot.line(x=data.index, y='close')
#this creates a scatter with the value in C column changing the color of the point
# df1.plot.scatter(x='A', y='B', c='C', cmap='coolwarm')
#same thing, but size instead of color
# df1.plot.scatter(x='A', y='B', s=df1['C']*100)
data = pd.read_csv('forex_hist.csv', index_col=0)
data.plot.box()
#hex plots
hexdata = pd.DataFrame(np.random.randn(1000,2), columns=['a', 'b'])
hexdata.plot.hexbin(x='a', y='b', gridsize=25, cmap='coolwarm')

#kernal density
hexdata.plot.density()

#time series viz
data = pd.read_csv('forex_hist.csv', index_col=0, parse_dates=True)
#set limits for plots- plot only subsections of data
data['rate'].plot(xlim=['2018-08-04', '2018-08-07'], ylim = (1.15, 1.16))
import matplotlib.dates as dates
idx = data.loc['2018-08-08':'2018-08-10'].index
stock = data.loc['2018-08-08':'2018-08-10']['rate']
fig, ax = plt.subplots()
ax.plot_date(idx,stock, '-')

#use this to clean up x axis and format as needed
#locating
ax.xaxis.set_major_locator(dates.HourLocator())
#formatting
ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%M'))
fig.autofmt_xdate()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#read in data, convert date column to a datetime object, then set it as index
df = pd.read_csv('walmart_stock.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#correct, easier method
df = pd.read_csv('walmart_stock.csv', index_col='Date', parse_dates=True)

#resampling, like grouping
df.resample(rule='A').mean() #look up table of rules, A is year end
#instead of using their aggregations, functions can be used
def first_day(entry):
    return entry[0]

df.resample('A').apply(first_day)
#visualize
df['Close'].resample('M').mean().plot.bar()

#shifting
#tshift changes the index to the frequency defined, kind of alternative to groupby
df.tshift(freq='M').head()

#rolling and expanding
df['Open'].plot(figsize=(16,6))
df.rolling(window=7).mean()['Close'].plot()
df['Close 30d mean'] = df['Close'].rolling(window=30).mean()
df[['Close 30d mean', 'Close']].plot(figsize=(16,6))
#expanding mean is a a mean of cumulative values
df['Close'].expanding().mean().plot()
#bollinger bands
df['20D MA'] = df['Close'].rolling(window=20).mean()
df['Upper'] = df['20D MA'] + 2*(df['Close'].rolling(20).std())
df['Lower'] = df['20D MA'] - 2*(df['Close'].rolling(20).std())
df[['Close', '20D MA', 'Upper', 'Lower']].tail(200).plot()

# ***** STATSMODELS ****
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = sm.datasets.macrodata.load_pandas().data
#get date index
index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
df.index = index
df['realgdp'].plot()
#get the trend of the data
gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(df['realgdp'])
df['trend'] = gdp_trend
df[['realgdp', 'trend']].plot()

#EWMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('airline_passengers.csv', index_col='Month')
df.dropna(inplace=True)
df.index = pd.to_datetime(df.index)
df['6m SMA'] = df['Thousands of Passengers'].rolling(window=6).mean()
df['12m SMA'] = df['Thousands of Passengers'].rolling(window=12).mean()
#adding weighted moving averages
df['EWMA 12'] = df['Thousands of Passengers'].ewm(span=12).mean()
df[['Thousands of Passengers', 'EWMA 12']].plot() 

#ETS decomposition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('airline_passengers.csv', index_col='Month')
df.dropna(inplace=True)
df.index = pd.to_datetime(df.index)
result = seasonal_decompose(df['Thousands of Passengers'], model='multiplicative')
result.seasonal.plot()
result.trend.plot()
#this is the group of plots with all ETS components
result.plot()
df.plot()

#ARIMA models (these don't work well for stock data)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('milk.csv')
df.drop(168, axis=0, inplace=True)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.columns = ['pounds']
df.describe().transpose()
df.plot()
time_series = df['pounds']
time_series.rolling(12).mean().plot(label='12m MA')
time_series.rolling(12).std().plot(label='12m std')
time_series.plot()
plt.legend()
decomp = seasonal_decompose(time_series)
fig = decomp.plot()

#employ dicky-fuller test
from statsmodels.tsa.stattools import adfuller


def adf_check(time_series):
    result = adfuller(time_series)
    print("Augmented Dicky-Fuller Test")
    labels = ['ADF Test statistic', 'p-value', '# of lags', '# of observations used']
    for value, label in zip(result, labels):
        print(label + " : " + str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis")
        print("Reject null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis")
        print("Fail to reject null hypothesis")
        print("Data has a unit root; it is non-stationary")

adf_check(df['pounds'])

#since data is non- stationary, try differencing, then test again.
df['first diff'] = df['pounds'] - df['pounds'].shift(1)
df['first diff'].plot()

adf_check(df['first diff'].dropna())

#this passed the test, but let's try another difference anyway
df['second diff'] = df['first diff'] - df['first diff'].shift(1)
adf_check(df['second diff'].dropna())
df['second diff'].plot()

#seasonal difference
df['seasonal diff'] = df['pounds'] - df['pounds'].shift(12)
df['seasonal diff'].plot()
adf_check(df['seasonal diff'].dropna())

#seasonal first difference
df['seasonal first diff'] = df['first diff'] - df['first diff'].shift(12)
df['seasonal first diff'].plot()
adf_check(df['seasonal first diff'].dropna())

#autocorrelation plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig_first = plot_acf(df['first diff'].dropna())
fig_seasonal_first = plot_acf(df['seasonal first diff'].dropna())

#can also be done in pandas
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['seasonal first diff'].dropna())

#partial autocorrelation plots
result = plot_pacf(df['seasonal first diff'].dropna())

plot_acf(df['seasonal first diff'].dropna())
plot_pacf(df['seasonal first diff'].dropna())

#acutally building ARIMA model
from statsmodels.tsa.arima_model import ARIMA
#seasonal model
model = sm.tsa.statespace.SARIMAX(df['pounds'], order=(0,1,0), seasonal_order=(1,1,1,12))
results = model.fit()
print(results.summary())
results.resid
results.resid.plot()
#distribution of errors is around zero, which is good!
results.resid.plot.kde()

#shows predictions versus actuals
df['forecast'] = results.predict(start=150, end=168)
df[['pounds', 'forecast']].plot()

#let's predict the future though
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1,25)]
future_df = pd.DataFrame(index=future_dates, columns=df.columns)
final_df = pd.concat([df, future_df])
final_df['forecast'] = results.predict(start=168, end=192)
final_df[['pounds', 'forecast']].plot()


#monte carlo simulation to find optimum portfolio allocation
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

aapl = pd.read_csv('AAPL_CLOSE', index_col='Date', parse_dates=True)
amzn = pd.read_csv('AMZN_CLOSE', index_col='Date', parse_dates=True)
ibm = pd.read_csv('IBM_CLOSE', index_col='Date', parse_dates=True)
csco = pd.read_csv('CISCO_CLOSE', index_col='Date', parse_dates=True)

stocks = pd.concat([aapl, csco, ibm, amzn], axis=1)
stocks.columns = ['aapl', 'csco', 'ibm', 'amzn']

#mean daily percent change
stocks.pct_change(1).mean()

#correlation
stocks.pct_change(1).corr()

#use log returns to detrend/normalize
stocks.pct_change(1).head()
log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()
log_ret.hist(bins=100)
log_ret.mean()
log_ret.cov() * 252

#random allocations
np.random.seed(101)
num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sr_arr = np.zeros(num_ports)

for ind in range(num_ports):
    
    #weights
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights) #this is to make them all sum to 1
    
    #save weights
    all_weights[ind, :] = weights
    
    #expected return
    ret_arr[ind] = np.sum((log_ret.mean() * weights) * 252)
    
    #expected volatility
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    
    #sharpe ration
    sr_arr[ind] = ret_arr[ind]/vol_arr[ind]


sr_arr.max()
sr_arr.argmax()
all_weights[1420, :]

max_sr_ret = ret_arr[sr_arr.argmax()]
max_sr_vol = vol_arr[sr_arr.argmax()]

plt.figure
plt.scatter(vol_arr, ret_arr, c=sr_arr, cmap='plasma')
plt.colorbar(label='SR')
plt.xlabel('volatility')
plt.ylabel('return')

plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')

#finding optimumn with scipy instead of a loop
#this is a lot like solver!
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

from scipy.optimize import minimize

def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    #return 0 if the sum of weights is 1
    return np.sum(weights) - 1

cons = ({'type': 'eq', 'fun':check_sum})
bounds = ((0,1), (0,1), (0,1), (0,1))
init_guess = [0.25, 0.25, 0.25, 0.25]
opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons )

opt_results
opt_results.x
get_ret_vol_sr(opt_results.x)

#efficient frontier
frontier_y = np.linspace(0, 0.3, 100)
def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]

frontier_volatility = []
for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun':check_sum},
            ({'type': 'eq', 'fun':lambda w: get_ret_vol_sr(w)[0] - possible_return}))
    result = minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons )
    frontier_volatility.append(result['fun'])

plt.figure
plt.scatter(vol_arr, ret_arr, c=sr_arr, cmap='plasma')
plt.colorbar(label='SR')
plt.xlabel('volatility')
plt.ylabel('return')

plt.plot(frontier_volatility, frontier_y, 'g--', linewidth=3)

#CAPM
from scipy import stats
import pandas as pd
import pandas_datareader as web



start = pd.to_datetime('2010-01-04')
end = pd.to_datetime('2017-07-25')

spy_etf = web.DataReader('SPY', 'robinhood', start, end)
aapl = web.DataReader('AAPL', 'quandl', start, end)
spy_etf.info()
spy_etf.head()

import matplotlib as plt
aapl['Close'].plot(label='AAPL')
spy_etf['close_price'].plot(label='SPY')
plt.legend()

aapl['cumulative'] = aapl['Close']/aapl['Close'].iloc[0]
spy_etf['cumulative'] = spy_etf['close_price']/spy_etf['close_price'].iloc[0]
aapl['cumulative'].plot()
spy_etf['cumulative'].plot()















