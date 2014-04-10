#An example for the Kaplan-Meier estimator
from __future__ import print_function
from statsmodels.compat.python import lrange
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.sandbox.survival2 import KaplanMeier

#Getting the strike data as an array
dta = sm.datasets.strikes.load()
print('basic data')
print('\n')
dta = list(dta.values()[-1])
print(dta[lrange(5),:])
print('\n')

#Create the KaplanMeier object and fit the model

km = KaplanMeier(dta,0)
km.fit()

#show the results

km.plot()
print('basic  model')
print('\n')
km.summary()
print('\n')

#Mutiple survival curves

km2 = KaplanMeier(dta,0,exog=1)
km2.fit()
print('more than one curve')
print('\n')
km2.summary()
print('\n')
km2.plot()

#with censoring

censoring = np.ones_like(dta[:,0])
censoring[dta[:,0] > 80] = 0
dta = np.c_[dta,censoring]
print('with censoring')
print('\n')
print(dta[lrange(5),:])
print('\n')
km3 = KaplanMeier(dta,0,exog=1,censoring=2)
km3.fit()
km3.summary()
print('\n')
km3.plot()

#Test for difference of survival curves

log_rank = km3.test_diff([0.0645,-0.03957])
print('log rank test')
print('\n')
print(log_rank)
print('\n')

#The zeroth element of log_rank is the chi-square test statistic
#for the difference between the survival curves for exog = 0.0645
#and exog = -0.03957, the index one element is the degrees of freedom for
#the test, and the index two element is the p-value for the test

wilcoxon = km3.test_diff([0.0645,-0.03957], rho=1)
print('Wilcoxon')
print('\n')
print(wilcoxon)
print('\n')

#Same info as log_rank, but for Peto and Peto modification to the
#Gehan-Wilcoxon test

#User specified functions for tests

#A wider range of rates can be accessed by using the 'weight' parameter
#for the test_diff method

#For example, if the desire weights are S(t)*(1-S(t)), where S(t) is a pooled
#estimate for the survival function, this could be computed by doing

def weights(t):
    #must accept one arguement, even though it is not used here
    s = KaplanMeier(dta,0,censoring=2)
    s.fit()
    s = s.results[0][0]
    s = s * (1 - s)
    return s

#KaplanMeier provides an array of times to the weighting function
#internally, so the weighting function must accept one arguement

test = km3.test_diff([0.0645,-0.03957], weight=weights)
print('user specified weights')
print('\n')
print(test)
print('\n')

#Groups with nan names

#These can be handled by passing the data to KaplanMeier as an array of strings

groups = np.ones_like(dta[:,1])
groups = groups.astype('S4')
groups[dta[:,1] > 0] = 'high'
groups[dta[:,1] <= 0] = 'low'
dta = dta.astype('S4')
dta[:,1] = groups
print('with nan group names')
print('\n')
print(dta[lrange(5),:])
print('\n')
km4 = KaplanMeier(dta,0,exog=1,censoring=2)
km4.fit()
km4.summary()
print('\n')
km4.plot()

#show all the plots

plt.show()
