# This file tests the likelihood cross validation bandwidth estimate of
# the np package in R (called through rpy2) and the statmodels likelihood cross validation procedure
# for continuous variables using the Epanechnikov kernel

# The estimates of the two packages seem to be very close but not identical. This could be due to the fact
# that different numerical optimization algorithms are used 

# Currently the test only works for the continuous Epanechnikov Kernel
# TODO: Extend for categorical kernels!

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
import scikits.statsmodels.api as sm
import bandwidths

NP=importr('np')
r=robjects.r


y=np.random.normal(2,1,size=200)  #generate random data
D={"S": robjects.FloatVector(y)}
df=robjects.DataFrame(D)
formula=r('~S')
r_bw=NP.npudensbw(formula, dat=df, bwmethod='cv.ml', ckertype='epanechnikov')  #obtain R's estimate of the BW

#print 'NP bandwidth is ', r_bw[0][0]

sm_bw=bandwidths.bw_likelihood_cv(y,'continuous')  #obtain SM's estimate of the bandwidth

#print "Statsmodels bandwidth is ", sm_bw

print "------------------------"*4
print 'the estimate by R is: ', r_bw[0][0], '||||||', 'the estimate by SM is: ', sm_bw
print "------------------------"*4
#err += abs((sm_bw-r_bw[0][0])/sm_bw)  

#y_ord = np.random.binomial(n=3,p=0.3,size=100)
#sm_bw_unord = bandwidths.bw_likelihood_cv(y_ord,'ordered',c=3)
