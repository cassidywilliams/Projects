import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm

# read csv
raw = pd.read_csv('ab_test_analysis.tsv', sep='\t', index_col='date')
raw.index = pd.to_datetime(raw.index)

# add columns with conversion rate and ones for regression
raw['conv rate'] =raw['conversions']/raw['visits']
raw['intercept'] = 1
raw['bounces'] = raw['bounce_rate'] * raw['visits']

# calculate one-tail upper z scores to compare conversions from test page versions to version 0
convert_control= sum(raw.query('version == 0')['conversions'])
convert_test = sum(raw.query('version == 2')['conversions'])
visits_control = raw.query('version == 0')['visits'].sum()
visits_test = raw.query('version == 2')['visits'].sum()
z_score, p_value = sm.stats.proportions_ztest([convert_test, convert_control], [visits_test, visits_control], alternative='larger')
z_score
p_value

#logistic regression at the level of version only
filtered_raw_version = raw.query('version in [0, 2]')
logit = smf.Logit(filtered_raw_version['conv rate'], filtered_raw_version[['intercept','version']])
results = logit.fit()
results.summary()

#logistic regression to test impact of location, device, and browser
encoded = pd.get_dummies(filtered_raw_version, columns=["location", "device", "browser"], drop_first=True)
logit = smf.Logit(encoded['conv rate'], encoded[["intercept", "version", "location_Canada", "location_Rest of World", "location_USA", "device_Mobile", "device_Tablet", "browser_Other", "browser_Safari"]])
results = logit.fit()
results.summary()

# calculate one-tail upper z scores to compare bounce rates from test page versions to version 0
convert_control= sum(raw.query('version == 0')['bounces'])
convert_test = sum(raw.query('version == 2')['bounces'])
visits_control = raw.query('version == 0')['visits'].sum()
visits_test = raw.query('version == 2')['visits'].sum()
z_score, p_value = sm.stats.proportions_ztest([convert_test, convert_control], [visits_test, visits_control], alternative='larger')
z_score
p_value





