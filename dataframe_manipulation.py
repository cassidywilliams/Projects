import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('forex_hist.csv')
data.plot(grid=1)

data['new_date'] = data.ds.str[:10]
data['new_time'] = data.ds.str[10:]

group = data.groupby(['new_time']).mean()
group.plot()

data['rate'].std()
data['rate'].mean()
data['rate'].median()