# time series regression test with Prophet

import pandas as pd
import numpy as np
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime
import holidays
from forex_python.converter import CurrencyRates

start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime.today()
daterange = pd.date_range(start_date, end_date)
dateframe = pd.DataFrame(
        {'ds': daterange,
         }) 

c = CurrencyRates()
temp_rates= []
for index, row in dateframe.iterrows():
    temp_rates.append(c.get_rate('EUR', 'USD', row[0]))
dateframe['rate'] = temp_rates
   

# define a few basic variables
#ts = datetime.datetime.now()
#rundate = str(datetime.date.today())

# build adaptive holiday dataframe
holis = []
holidate = []
for date, name in sorted(holidays.US(years=[2015, 2016, 2017, 2018]).items()):
        holis.append(name)
        holidate.append(date) 
holidaylist = pd.DataFrame(
        {'ds': holidate,
         'holiday': holis
         })  
holidaylist["ds"] = pd.to_datetime(holidaylist["ds"])
holidaylist['weekday'] = holidaylist['ds'].dt.dayofweek
lower_window = []
upper_window = []
for row in holidaylist['weekday']:
    if row == 0:
        lower_window.append(-3)
        upper_window.append(0)
    elif row == 1:
        lower_window.append(-4)
        upper_window.append(0)
    elif row == 2:
        lower_window.append(0)
        upper_window.append(0)
    elif row == 3:
        lower_window.append(-1)
        upper_window.append(2)
    elif row == 4:
        lower_window.append(-1)
        upper_window.append(0)
    elif row == 5:
        lower_window.append(0)
        upper_window.append(0)
    elif row == 6:
        lower_window.append(-1)
        upper_window.append(0)
    else:
        lower_window.append(0)
        upper_window.append(0)
holidaylist['lower_window'] = lower_window
holidaylist['upper_window'] = upper_window

del date, holidate, holis, name, lower_window, row, upper_window

# need to add logging
#import logging
#logging.basicConfig(filename='debug.log',level=logging.DEBUG)


#read .csv to dataframe
#pd.read_csv()

# import data and log-transform the y variable (bookings) to make this variable stationary
dateframe['y'] = np.log(dateframe['rate'])



# fit the model by instantiating a new prophet object. then call its fit method and pass in the historical dataframe
#m = Prophet()
m = Prophet(holidays=holidaylist)
m.add_seasonality(name='monthly', period=30.3, fourier_order=5)
m.fit(dateframe);

# predictions are then made aon a dataframe with a column called ds containing the dates for which a prediction is to be made
# choose number of days in the future with the make_future_dataframe method, by default the historical dates will be included as well
future = m.make_future_dataframe(periods=30)

# predict the yhat values. if historical dates are passed in here, then it will provide an in-sample fit. 
# the forecast object here is a new dataframe with the yhat values
forecast = m.predict(future)

# plot the forecast by calling the Prophet.plot method and passing in forecast dataframe.
m.plot(forecast);

# plot components of seasonality
m.plot_components(forecast);

# convert logarithmic forecasted values back into usable booking counts, then combine with actuals for exporting
output = forecast[['ds']]
output['forecast_value'] = np.exp(forecast['yhat'])
output['rate'] = dateframe['rate']
#output.forecast_value = output.forecast_value.round()

# export to csv for tableau use
output.to_csv('forecast_' + partner + '_' + rundate + '.csv', encoding='utf-8', index=False)

# copy the output dataframe to a new one, drop the forecasted values, then add metrics to measure performance
metrics = output.copy()
metrics.dropna(inplace=True)
metrics['e'] = metrics['rate'] - metrics['forecast_value']
metrics['p'] = np.abs(metrics['e'] / metrics['rate'])

# using sklearn metrics to test model accuracy
r2 = r2_score(metrics.forecast_value, metrics.rate)
mse = mean_squared_error(metrics.forecast_value, metrics.rate)
mae = mean_absolute_error(metrics.forecast_value, metrics.rate)
mape = np.mean(metrics['p'])
print(r2)

# create dataframe with log metrics and write them to log csv to track error
log = pd.DataFrame({'forecast_date':ts, 'partner':partner, 'r-sq':r2, 'mse':mse, 'mae':mae, 'mape':mape}, 
                    columns=['forecast_date', 'partner', 'r-sq', 'mse', 'mae', 'mape'], index = [0])
log.to_csv('forecast_log.csv', mode='a', header=False)

#initial steps to write to table in redshift. need s3 access before this will be possible.
#pr.pandas_to_redshift(data_frame = forecast,
#                        redshift_table_name = 'sandbox.partner_forecast')

# cross validation
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(m, horizon = '60 days')
help(cross_validation)

