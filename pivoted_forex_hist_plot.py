import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pytz import all_timezones

# http://www.histdata.com/f-a-q/

data = pd.read_csv('forex_aug_18_all.csv')
data['stamp'] = pd.to_datetime(data['datetime'])
data['cest'] = data['stamp'] + pd.DateOffset(hours=6)
data['date'] = [d.date() for d in data['cest']]
data['time'] = [d.time() for d in data['cest']]

data.drop(columns=['datetime', 'stamp'], axis=1, inplace=True)

piv = data.pivot(index=str('time'), columns=str('date'), values='close')
piv.reset_index(level=0, inplace=True)


piv.plot(x=piv['time'],legend=None)
