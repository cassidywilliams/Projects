import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv('forex_aug_18_all.csv', index_col=0)

#https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
#autocorrelation plotting, pick out the number of lags near where the line crosses, so 5k in this example is where we will start

#autocorrelation_plot(data)
#plt.show()

#fitting ARIMA model
#First, we fit an ARIMA(50,1,0) model. This sets the lag value to 50 for autoregression, 
#uses a difference order of 1 to make the time series stationary, and uses a moving average model of 10.



data2 = data.tail(1000)

X = data2.values
size = int(len(X) * 0.75)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []
strat = []

for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test[t])
    print('predicted=%f, expected=%f' % (yhat, test[t]))
    if yhat > test[t-1]:
        strat.append(-1)
    else: 
        strat.append(1)
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
print(model_fit.summary())

r2_score(test, predictions)