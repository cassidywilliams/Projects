import numpy as np
import pandas as pd
import datetime as dt
from pylab import plt
plt.style.use('seaborn')
# % matplotlib inline
import fxcmpy
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

#define number of lags to test for features. if this is the method determined to work best, add a loop to automate select number of lags
lags = 10
symbol = 'EUR/USD'

api = fxcmpy.fxcmpy(config_file='fxcm.cfg')

# Data
# think about how much data to use to fit model
try:
    raw = api.get_candles(symbol, period='m1', start='2018-09-09', end='2018-09-24')
    raw.info()
except ValueError:
    print("Cannot connect to API")

# Features
data = pd.DataFrame()
data['midclose'] = (raw['bidclose'] + raw['askclose']) / 2
data.info()

#this takes the log of (value/previous value)
data['returns'] = np.log(data / data.shift(1))
data.head()

#create list of lags, add column for each to data df and for each value with -1/0/+1
#The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.

#-----------------------------------------------------------------------------------
#MLP classification model
model = MLPClassifier(hidden_layer_sizes=[200, 200], max_iter=200)
def lagger():
    
    global cols, scores
    lag_counts = range(1, lags + 1)
    cols = []
    scores = []
    
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = np.sign(data['returns'].shift(lag))
        cols.append(col)
        data.dropna(inplace=True)
        print('Iteration number: {}'.format(lag))
        %time model.fit(data[cols], np.sign(data['returns']))
        model.predict(data[cols])
        data['prediction'] = model.predict(data[cols])
        data['prediction'].value_counts()
        score = accuracy_score(data['prediction'], np.sign(data['returns']))
        scores.append(score)
        
    plt.figure()
    plt.plot(lag_counts, scores, lw=2)
    plt.xlabel('# of Lags')
    plt.ylabel('Test Score')
    
    return scores, cols

lagger()
#---------------------------------------------------------------------------------------
#MORE FEATURES
cols.append('rolling mean 3')
cols.append('rolling mean 6')
cols.append('rolling mean 12')
cols.append('rolling std 12')
cols.append('rolling std 5')
cols.append('rolling min 5')
cols.append('rolling max 5')
cols.append('rolling median 12')

data['rolling mean 3'] = data['midclose'].rolling(3).mean()
data['rolling mean 6'] = data['midclose'].rolling(6).mean()
data['rolling mean 12'] = data['midclose'].rolling(12).mean()
data['rolling std 12'] = data['midclose'].rolling(12).std()
data['rolling std 5'] = data['midclose'].rolling(5).std()
data['rolling min 5'] = data['midclose'].rolling(5).min()
data['rolling max 5'] = data['midclose'].rolling(5).max()
data['rolling median 12'] = data['midclose'].rolling(12).median()


#linear regression model
cols = []
for lag in range(1, lags + 1):
     col = 'lag_{}'.format(lag)
     data[col] = np.sign(data['returns'].shift(lag))
     cols.append(col)
data.dropna(inplace=True)
reg = LinearRegression()
%time reg.fit(data[cols], np.sign(data['returns']))
reg.predict(data[cols])
data['prediction'] = np.sign(reg.predict(data[cols]))
score = accuracy_score(data['prediction'], np.sign(data['returns']))



#accuracy_score(data['prediction'], np.sign(data['returns']))
#---------------------------------------------------------------------------------------
#what is this for?
# #2 ** lags


# # Strategy
# uses NN MLP classifier- test prophet here. if MLP is used, how many layers and iterations are needed?
# model = MLPClassifier(hidden_layer_sizes=[200, 200], max_iter=200)
# #a list of columns is passed into the df call to handle all columns: neat
# #by calling the model.fit method, we pass in cols for regressors and sign.returns for the y
# #%time is a function used to time how long a function takes to run
# %time model.fit(data[cols], np.sign(data['returns']))
# #this predicts the direction of return, so -1/+1. these are added in a new column.
# model.predict(data[cols])
# data['prediction'] = model.predict(data[cols])
# #this simply prints a summary of counts for each value (number -1, number 0, number x, etc.)
# data['prediction'].value_counts()

# #this metric shows a percentage of values with exact matches (success rate)
# accuracy_score(data['prediction'], np.sign(data['returns']))

# Backtesting
#this is multiplying the return by the sign of prediction (as if we made the trade as predicted: what the outcome was)
# data['strategy'] = data['prediction'] * data['returns']
# #this prints the sum of returns for both normal returns and strategy columns
# np.exp(data[['returns', 'strategy']].sum())

# # plots the cumsum of strategy vs returns
# data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))

#real backtesting
def backtest(model):
    global test_data, end_result
    raw_test = api.get_candles(symbol, period='m1', start='2018-09-25', end='2018-09-27')
    test_data = pd.DataFrame()
    test_data['midclose'] = (raw_test['bidclose'] + raw_test['askclose']) / 2
    test_data['returns'] = np.log(test_data / test_data.shift(1))
    scores = []
    move = 1
    for col in cols:
        move += 1
        test_data[col] = np.sign(test_data['returns'].shift(move))
        test_data.dropna(inplace=True)
    test_data['prediction'] = model.predict(test_data[cols])
    test_data['strategy'] = np.sign(test_data['prediction']) * test_data['returns']
    np.exp(test_data[['returns', 'strategy']].sum())
    end_result = pd.DataFrame(test_data[['returns', 'strategy']].cumsum().apply(np.exp))
    test_data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    
    
backtest(reg)

# Automated Trading

# def output(data, dataframe):
#     print('%3d | %s | %s | %6.5f, %6.5f' 
#       % (len(dataframe), data['Symbol'],
#          pd.to_datetime(int(data['Updated']), unit='ms'), 
#          data['Rates'][0], data['Rates'][1]))
#
## #this uses output as a callback function so it passes the market data into the output function
#api.subscribe_market_data('EUR/USD', (output,))
#
#api.unsubscribe_market_data('EUR/USD') 

specs = ['tradeId', 'amountK', 'currency',
       'grossPL', 'isBuy', 'open', 'close', 'amountK']

order = api.create_market_buy_order('EUR/USD', 1)

#if no trades are open this triggers an error. need to handle this.
open_pos = api.get_open_positions()[specs]

#np.arange method creates an ordered array, reshape changes it into a matrix (row,column)
np.arange(10).reshape(1, -1)

threshold = 0.0001
position = 0
trades = 0
ticks = 0
min_length = lags + 1
max_open_pos = 5
PL = .1

def auto_trader(data, dataframe):
    global position, trades, ticks, min_length, threshold, open_pos, specs
    ticks += 1
    #end by default is /n, but can change the ending behaviour of a print statement
    print(ticks, end=' ')
    #this is a resample of the response passed into the auto_trader function. it resamples every 10s.
    resam = dataframe.resample('10s', label='right').last().ffill()
    try:
         open_pos = api.get_open_positions()[specs]
    except:
         open_pos = [0]
        

#    if len(open_pos) < max_open_pos:
    if len(resam) > min_length:
        min_length += 1
        resam['mid'] = (resam['Bid'] + resam['Ask']) / 2
        resam['returns'] = np.log(resam['mid'] / resam['mid'].shift(1))
        features = np.sign(resam['returns'].iloc[-(lags+1):-1])
        features = features.values.reshape(1, -1)
        signal = model.predict(features)
        print('\nNEW SIGNAL: {}'.format(signal))
        
        if position in [0, -1]:
            if signal == 1:
                if position == -1:
                    api.close_all_for_symbol(symbol)
                api.create_market_buy_order(symbol, 1)
                position = 1
                print('{} | ***PLACING BUY ORDER***'.format(dt.datetime.now()))
                    #open_pos = api.get_open_positions()[specs]
        elif position in [0, 1]:
            if signal == -1:
                if position == 1:
                    api.close_all_for_symbol(symbol)
                api.create_market_sell_order(symbol, 1)
                position = -1
                print('{} | ***PLACING SELL ORDER***'.format(dt.datetime.now()))
                    #open_pos = api.get_open_positions()[specs]
 
        if ticks > 100:
            pass
        
    # else:
    #     for index, row in open_pos.iterrows():
    #         if row[4] == True and row[3]> PL:
    #             api.close_trade(trade_id=row[0], amount=row[7])
    #             print('Position is above threshold of {}, trade closed at {} gain.'.format(threshold,(row[6]/row[5] - 1)))
    #             open_pos = api.get_open_positions()[specs]
    #         elif row[4] == False and row[3]> PL:
    #             api.close_trade(trade_id=row[0], amount=row[7])
    #             print('Position is above threshold of {}, trade closed at {} gain.'.format(threshold,(row[5]/row[6] - 1)))
    #             open_pos = api.get_open_positions()[specs]
    # else:
    #     for index, row in open_pos.iterrows():
    #         if row[4] == True and row[6]/row[5] > threshold + 1:
    #             api.close_trade(trade_id=row[0], amount=row[7])
    #             print('Position is above threshold of {}, trade closed at {} gain.'.format(threshold,(row[6]/row[5] - 1)))
    #             open_pos = api.get_open_positions()[specs]
    #         elif row[4] == False and row[5]/row[6] > threshold + 1:
    #             api.close_trade(trade_id=row[0], amount=row[7])
    #             print('Position is above threshold of {}, trade closed at {} gain.'.format(threshold,(row[5]/row[6] - 1)))
    #             open_pos = api.get_open_positions()[specs]
        

        
api.subscribe_market_data('EUR/USD', (auto_trader,))

api.unsubscribe_market_data('EUR/USD') 


api.close_all()

api.get_open_positions()

#---------------------------------------------------------------------------------------------
#TESTING LINEAR REGRESSION MODEL HERE

threshold = 0.0005
position = 0
trades = 0
ticks = 0
min_length = lags + 1
max_open_pos = 5
PL = .1

def auto_trader(data, dataframe):
    global position, trades, ticks, min_length, threshold, open_pos, specs, resam, features
    ticks += 1
    #end by default is /n, but can change the ending behaviour of a print statement
    print(ticks, end=' ')
    #this is a resample of the response passed into the auto_trader function. it resamples every 10s.
    resam = dataframe.resample('15s', label='right').last().ffill()
    try:
         open_pos = api.get_open_positions()[specs]
    except:
         open_pos = [0]
        
blah = np.sign(reg.predict(features))
#    if len(open_pos) < max_open_pos:
    if len(resam) > min_length:
        min_length += 1
        resam['mid'] = (resam['Bid'] + resam['Ask']) / 2
        resam['returns'] = np.log(resam['mid'] / resam['mid'].shift(1))
        features = np.sign(resam['returns'].iloc[-(lags+1):-1])
        features = features.values.reshape(1, -1)
        signal = np.sign(reg.predict(features))
        print('\nNEW SIGNAL: {}'.format(signal))
        
        if position in [0, -1]:
            if signal == 1:
                if position == -1:
                    api.close_all_for_symbol(symbol)
                api.create_market_buy_order(symbol, 1)
                trades += 1
                position = 1
                print('{} | ***PLACING BUY ORDER***'.format(dt.datetime.now()))
                    #open_pos = api.get_open_positions()[specs]
        elif position in [0, 1]:
            if signal == -1:
                if position == 1:
                    api.close_all_for_symbol(symbol)
                api.create_market_sell_order(symbol, 1)
                position = -1
                trades += 1
                print('{} | ***PLACING SELL ORDER***'.format(dt.datetime.now()))
                    #open_pos = api.get_open_positions()[specs]
 
        if ticks > 100:
            pass