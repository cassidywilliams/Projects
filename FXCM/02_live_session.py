
# coding: utf-8

# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>

# # ODSC &mdash; Live Session

# ## Risk Disclaimer

# Trading forex/CFDs on margin carries a high level of risk and may not be suitable for all investors as you could sustain losses in excess of deposits. Leverage can work against you. Due to the certain restrictions imposed by the local law and regulation, German resident retail client(s) could sustain a total loss of deposited funds but are not subject to subsequent payment obligations beyond the deposited funds. Be aware and fully understand all risks associated with the market and trading. Prior to trading any products, carefully consider your financial situation and experience level. Any opinions, news, research, analyses, prices, or other information is provided as general market commentary, and does not constitute investment advice. FXCM & TPQ will not accept liability for any loss or damage, including without limitation to, any loss of profit, which may arise directly or indirectly from use of or reliance on such information.

# ## Speaker Disclaimer

# The speaker is neither an employee, agent nor representative of FXCM and is therefore acting independently. The opinions given are their own, constitute general market commentary, and do not constitute the opinion or advice of FXCM or any form of personal or investment advice. FXCM assumes no responsibility for any loss or damage, including but not limited to, any loss or gain arising out of the direct or indirect use of this or any other content. Trading forex/CFDs on margin carries a high level of risk and may not be suitable for all investors as you could sustain losses in excess of deposits.

# ## The Imports

# In[ ]:


import numpy as np
import pandas as pd
import datetime as dt
from pylab import plt
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## API 

# In[ ]:


import fxcmpy


# In[ ]:


api = fxcmpy.fxcmpy(config_file='../fxcm.cfg')  # adjust path/filename


# ## Data

# In[ ]:


symbol = 'EUR/USD'


# In[ ]:


raw = api.get_candles(symbol, period='m1', start='2018-08-18', end='2018-08-30')


# In[ ]:


raw.info()


# ## Features

# In[ ]:


data = pd.DataFrame()


# In[ ]:


data['midclose'] = (raw['bidclose'] + raw['askclose']) / 2


# In[ ]:


data.info()


# In[ ]:


data['returns'] = np.log(data / data.shift(1))


# In[ ]:


data.head()


# In[ ]:


lags = 5


# In[ ]:


cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = np.sign(data['returns'].shift(lag))
    cols.append(col)


# In[ ]:


data.head(8)


# In[ ]:


2 ** lags


# In[ ]:


data.dropna(inplace=True)


# ## Strategy

# In[ ]:


from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# In[ ]:


model = MLPClassifier(hidden_layer_sizes=[200, 200], max_iter=200)


# In[ ]:


get_ipython().run_line_magic('time', "model.fit(data[cols], np.sign(data['returns']))")


# In[ ]:


model.predict(data[cols])


# In[ ]:


data['prediction'] = model.predict(data[cols])


# In[ ]:


data['prediction'].value_counts()


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(data['prediction'], np.sign(data['returns']))


# ## Backtesting

# In[ ]:


data['strategy'] = data['prediction'] * data['returns']


# In[ ]:


np.exp(data[['returns', 'strategy']].sum())


# In[ ]:


data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# ## Train-Test Split

# ## Automated Trading

# In[ ]:


def output(data, dataframe):
    print('%3d | %s | %s | %6.5f, %6.5f' 
          % (len(dataframe), data['Symbol'],
             pd.to_datetime(int(data['Updated']), unit='ms'), 
             data['Rates'][0], data['Rates'][1]))


# In[ ]:


api.subscribe_market_data('EUR/USD', (output,))


# In[ ]:


api.unsubscribe_market_data('EUR/USD') 


# In[ ]:


sel = ['tradeId', 'amountK', 'currency',
       'grossPL', 'isBuy']


# In[ ]:


order = api.create_market_buy_order('EUR/USD', 1)


# In[ ]:


api.get_open_positions()[sel]


# In[ ]:


api.close_all()


# In[ ]:


api.get_open_positions()


# In[ ]:


np.arange(10).reshape(1, -1)


# In[ ]:


position = 0
trades = 0
ticks = 0
min_length = lags + 1
def auto_trader(data, dataframe):
    global position, trades, ticks, min_length
    ticks += 1
    print(ticks, end=' ')
    resam = dataframe.resample('10s', label='right').last().ffill()
    
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
                
        elif position in [0, 1]:
            if signal == -1:
                if position == 1:
                    api.close_all_for_symbol(symbol)
                api.create_market_sell_order(symbol, 1)
                position = -1
                print('{} | ***PLACING SELL ORDER***'.format(dt.datetime.now()))
        
        if ticks > 100:
            pass
    


# In[ ]:


api.subscribe_market_data('EUR/USD', (auto_trader,))


# In[ ]:


api.unsubscribe_market_data('EUR/USD') 


# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>
# 
# <a href="http://tpq.io" target="_blank">http://tpq.io</a> | <a href="http://twitter.com/dyjh" target="_blank">@dyjh</a> | <a href="mailto:training@tpq.io">training@tpq.io</a>
