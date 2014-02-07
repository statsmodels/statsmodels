#An example for the Kaplan-Meier estimator
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.sandbox.survival2 import KaplanMeier, Survival

#Getting the strike data as an array
dta = sm.datasets.strikes.load()
print 'basic data'
print '\n'
dta = dta.values()[-1]
print dta[range(5),:]
print '\n'

#Create the KaplanMeier object and fit the model

dtas = Survival(0, censoring=None, data=dta)
km = KaplanMeier(dtas)
km_res = km.fit()

#show the results

km_res.plot()
print 'basic  model'
print '\n'
print km_res.summary()
print '\n'

#Mutiple survival curves

km2 = KaplanMeier(dtas,exog=dta[:,1])
km_res2 = km2.fit()
print 'more than one curve'
print '\n'
print km_res2.summary()
print '\n'
km_res2.plot()

#with censoring

censoring = np.ones_like(dta[:,0])
censoring[dta[:,0] > 80] = 0
dta = np.c_[dta,censoring]
dtas = Survival(0, censoring=None, data=dta)
print 'with censoring'
print '\n'
print dta[range(5),:]
print '\n'
km3 = KaplanMeier(dtas,exog=dta[:,1][:,None]) #,censoring=2)
km_res3 = km3.fit()
print km_res3.summary()
print '\n'
km_res3.plot()

#Test for difference of survival curves

log_rank = km_res3.test_diff([0.0645,-0.03957])
print 'log rank test'
print '\n'
print log_rank
print '\n'

#The zeroth element of log_rank is the chi-square test statistic
#for the difference between the survival curves for exog = 0.0645
#and exog = -0.03957, the index one element is the degrees of freedom for
#the test, and the index two element is the p-value for the test

wilcoxon = km_res3.test_diff([0.0645,-0.03957], rho=1)
print 'Wilcoxon'
print '\n'
print wilcoxon
print '\n'

#Same info as log_rank, but for Peto and Peto modification to the
#Gehan-Wilcoxon test

#User specified functions for tests

#A wider range of rates can be accessed by using the 'weight' parameter
#for the test_diff method

#For example, if the desire weights are S(t)*(1-S(t)), where S(t) is a pooled
#estimate for the survival function, this could be computed by doing

def weights(t):
    #must accept one arguement, even though it is not used here
    dtas_ = Survival(0, censoring=2, data=dta)
    s = KaplanMeier(dtas_)
    res = s.fit()
    s = s.results[0][0]
    s = s * (1 - s)
    return s

#KaplanMeier provides an array of times to the weighting function
#internally, so the weighting function must accept one arguement

test = km_res3.test_diff([0.0645,-0.03957], weight=weights)
print 'user specified weights'
print '\n'
print test
print '\n'

#Groups with nan names

#These can be handled by passing the data to KaplanMeier as an array of strings

groups = np.ones_like(dta[:,1])
groups = groups.astype('S4')
groups[dta[:,1] > 0] = 'high'
groups[dta[:,1] <= 0] = 'low'
dta = dta.astype('S4')
dta[:,1] = groups
print 'with nan group names'
print '\n'
print dta[range(5),:]
print '\n'
dtas2 = Survival(0, censoring=2, data=dta)
#convert to int
u = np.unique(dta[:,1], return_inverse=True)
km4 = KaplanMeier(dtas2, exog=u[1])
km_res4 = km4.fit()
print km_res4.summary()
print '\n'
km_res4.plot()

#show all the plots

plt.show()
