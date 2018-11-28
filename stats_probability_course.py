import random
from matplotlib import pyplot as plt
from scipy.stats import binom
from scipy.stats import poisson
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import norm
from scipy.stats import normaltest

results = []
#probability of rolling a 5, 3 times, out of 16 rolls
#binom.pmf(3,16, 1/6)

for d in range(200):
    results.append(binom.pmf(d,200, 1/6))

plt.hist(results)


#probability of only 4 deliveries arriving bw 4 and 5. 8 are expected.
poisson.pmf(4, 8)

#probability of less than 3 deliveries (cumulative)
poisson.cdf(2,8)

#simulate a coin flip
n, p = 100, .5  # number of trials, probability of each trial
s = np.random.binomial(n, p, 10000)
plt.hist(s)


# Shapiro-Wilk Test
# Tests whether a data sample has a Gaussian distribution.

# Assumptions
# Observations in each sample are independent and identically distributed (iid).

# Interpretation
# H0: the sample has a Gaussian distribution.
# H1: the sample does not have a Gaussian distribution.

r = norm.rvs(size=5000)
plt.hist(r)
stat, p = shapiro(r)

# D’Agostino’s K^2 Test
# Tests whether a data sample has a Gaussian distribution.

# Assumptions
# Observations in each sample are independent and identically distributed (iid).

# Interpretation
# H0: the sample has a Gaussian distribution.
# H1: the sample does not have a Gaussian distribution.
w = np.random.normal(0,1, size=5000)
stat, p = normaltest(w)
plt.hist(w)


# Student’s t-test
# Tests whether the means of two independent samples are significantly different.

# Assumptions
# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample are normally distributed.
# Observations in each sample have the same variance.

# Interpretation
# H0: the means of the samples are equal.
# H1: the means of the samples are unequal.

from scipy.stats import ttest_ind
raw = pd.read_csv('t_test.csv')
data1, data2 = raw['18'], raw['24']
stat, p = ttest_ind(data1, data2)

# in this example, the p value is very low, so therefore we reject the null and support that the samples are unequal

# Analysis of Variance Test (ANOVA)
# Tests whether the means of two or more independent samples are significantly different.

# Assumptions
# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample are normally distributed.
# Observations in each sample have the same variance.

# Interpretation
# H0: the means of the samples are equal.
# H1: one or more of the means of the samples are unequal.

from scipy.stats import f_oneway
stat, p = f_oneway(data1, data2)
