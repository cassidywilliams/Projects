import pandas as pd
import reverse_geocoder as rg
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import seaborn as sns

# read in csv, add month_year column, drop rows with nulls
komoot = pd.read_csv('tour-events.csv')
komoot['timestamp']= pd.to_datetime(komoot['timestamp'])
komoot['month_year'] = komoot.timestamp.apply(lambda x: x.strftime('%Y-%m'))
komoot.dropna(inplace=True)

# use reverse geocoding to get country name for each lat/long pair and add column to df
countries = []
coordinates = komoot[['latitude', 'longitude']].apply(tuple, axis=1).tolist()
results = rg.search(coordinates)
for i in results:
    countries.append(i['cc'])
komoot['country'] = countries

# narrow down dataset to include top 5 countries
tops = komoot.groupby('country').filter(lambda x: len(x) > 10000)
tops['country'].value_counts()

#pivot tops and count unique customers
pivot = pd.pivot_table(tops, index='month_year', columns='country', values='user', aggfunc=pd.Series.nunique)
pivot.index = pd.to_datetime(pivot.index)
pivot.fillna(0, inplace=True)

pivot.plot(legend=True, figsize=(16,12))

top_country_list = list(tops.country.unique())

#ETS decomposition
result = seasonal_decompose(pivot['AT'][1:], model='multiplicative')
result.seasonal.plot()
result.trend.plot()
result.plot()

for country in top_country_list:
#create subset of data for each country
    country_data = komoot.loc[(komoot['country'] == country) & (komoot['timestamp'] > '2016-01-01')]
    
    #create cohort groups based on first use date
    country_data.set_index('user', inplace=True)
    country_data['cohort_group'] = country_data.groupby(level=0)['timestamp'].min().apply(lambda x: x.strftime('%Y-%m'))
    country_data.reset_index(inplace=True)
    country_data.head()
    
    #rollup data by cohort group and month
    grouped = country_data.groupby(['cohort_group', 'month_year'])
    
    # count the unique users and counts of events per cohort
    cohorts = grouped.agg({'user': pd.Series.nunique,
                           'timestamp': pd.Series.count
                           })
    cohorts.rename(columns={'user': 'total_users',
                            'timestamp': 'total_events'}, inplace=True)
    
    def cohort_period(df):
        df['cohort_period'] = np.arange(len(df)) + 1
        return df
    
    cohorts = cohorts.groupby(level=0).apply(cohort_period)
    
    # reindex the DataFrame
    cohorts.reset_index(inplace=True)
    cohorts.set_index(['cohort_group', 'cohort_period'], inplace=True)
    
    # create a series holding the total size of each cohort group
    cohort_group_size = cohorts['total_users'].groupby(level=0).first()
    
    user_retention = cohorts['total_users'].unstack(0).divide(cohort_group_size, axis=1)
    user_retention2 = user_retention.shift(-1)
    
    #plot average retention rates by country
    # user_retention.fillna(0, inplace=True)
    # user_retention['average'] = user_retention.mean(axis=1)
    # user_retention['average'].plot(legend=True, fontsize=14, figsize=(12,8)).legend(top_country_list)
    # plt.title('Average Retention by Country', fontsize=16)
    # plt.ylabel('Retention rate', fontsize=14)
    # plt.xlabel('Months after first use', fontsize=14)


    #plot (line) user retention by group
    user_retention.plot(figsize=(10,5))
    plt.title('{} Cohort User Retention'.format(country))
    plt.xticks(np.arange(1, 12.1, 1))
    plt.xlim(1, 12)
    plt.ylabel('% of cohort returning');
    
    #plot heat maps foe each country
    # sns.set(style='white')
    # sns.set(font_scale=.5)  
    # plt.figure(figsize=(12, 8))
    # plt.title('{} User Retention'.format(country), fontsize=14)
    # sns.heatmap(user_retention2.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%', cmap="YlGnBu")
    # plt.ylabel('Cohort Group', fontsize=12)
    # plt.xlabel('Months after first use', fontsize=12)