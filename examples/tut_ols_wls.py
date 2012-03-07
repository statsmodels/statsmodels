'''Compare OLS and WLS

Note: uncomment plt.show() to display graphs
'''

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

nsample = 50
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, (x1-5)**2, np.ones(nsample)]

sig = 0.5
beta = [0.5, -0.0, 5.]
y_true2 = np.dot(X, beta)
y2 = y_true2 + sig*1. * np.random.normal(size=nsample)

plt.figure()
plt.plot(x1, y2, 'o', x1, y_true2, 'b-')

res2 = sm.OLS(y2, X).fit()
print res2.params
print res2.bse
#print res.predict
plt.plot(x1, res2.fittedvalues, 'r--')

# Example WLS: Heteroscedasticity 2 groups
# ----------------------------------------

#model assumption:
# * identical coefficients
# * misspecificaion: true model is quadratic, estimate only linear
# * independent noise/error term
# * two groups for error variance, low and high variance groups


#..np.random.seed(123456789)
np.random.seed(0)
#..9876789) #9876543)
beta = [0.5, -0.01, 5.]
y_true2 = np.dot(X, beta)
w = np.ones(nsample)
w[nsample*6/10:] = 3
#..y2[:nsample*6/10] = y_true2[:nsample*6/10] + sig*1. * np.random.normal(size=nsample*6/10)
#..y2[nsample*6/10:] = y_true2[nsample*6/10:] + sig*4. * np.random.normal(size=nsample*4/10)
y2 = y_true2 + sig*w* np.random.normal(size=nsample)
X2 = X[:,[0,2]]

# OLS estimate
# ^^^^^^^^^^^^
# unbiased parameter estimated, biased parameter covariance, standard errors

print 'OLS'
plt.figure()
plt.plot(x1, y2, 'o', x1, y_true2, 'b-')
res2 = sm.OLS(y2, X[:,[0,2]]).fit()
print 'OLS beta estimates'
print res2.params
print 'OLS stddev of beta'
print res2.bse

# heteroscedasticity corrected standard errors for OLS
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#OLS standard errors are inconsistent (?) with heteroscedasticity
#use correction
#sandwich estimators of parameter covariance matrix
print 'heteroscedasticity corrected standard error of beta estimates'
print res2.HC0_se
print res2.HC1_se
print res2.HC2_se
print res2.HC3_se

#print res.predict
#plt.plot(x1, res2.fittedvalues, '--')


#WLS knowing the true variance ratio of heteroscedasticity
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print '\nWLS'
res3 = sm.WLS(y2, X[:,[0,2]], 1./w).fit()
print 'WLS beta estimates'
print res3.params
print 'WLS stddev of beta'
print res3.bse
#print res.predict
#plt.plot(x1, res3.fittedvalues, '--.')

#Detour write function for prediction standard errors

#Prediction Interval for OLS
#---------------------------
covb = res2.cov_params()
# full covariance:
#predvar = res2.mse_resid + np.diag(np.dot(X2,np.dot(covb,X2.T)))
# predication variance only
predvar = res2.mse_resid + (X2 * np.dot(covb,X2.T).T).sum(1)
predstd = np.sqrt(predvar)
tppf = stats.t.ppf(0.975, res2.df_resid)
plt.plot(x1, res2.fittedvalues, 'r--')
plt.plot(x1, res2.fittedvalues + tppf * predstd, 'r--')
plt.plot(x1, res2.fittedvalues - tppf * predstd, 'r--')


#..Prediction Interval for WLS
#..---------------------------
#..covb = res3.cov_params()
##.. full covariance:
##..predvar = res3.mse_resid + np.diag(np.dot(X2,np.dot(covb,X2.T)))
##.. predication variance only
#..predvar = res3.mse_resid*w + (X2 * np.dot(covb,X2.T).T).sum(1)
#..predstd = np.sqrt(predvar)
#..tppf = stats.t.ppf(0.975, res3.df_resid)
#..plt.plot(x1, res3.fittedvalues, 'g--.')
#..plt.plot(x1, res3.fittedvalues + tppf * predstd, 'g--')
#..plt.plot(x1, res3.fittedvalues - tppf * predstd, 'g--')
#..plt.title('blue: true, red: OLS, green: WLS')

prstd, iv_l, iv_u = wls_prediction_std(res3)
plt.plot(x1, res3.fittedvalues, 'g--.')
plt.plot(x1, iv_u, 'g--')
plt.plot(x1, iv_l, 'g--')
plt.title('blue: true, red: OLS, green: WLS')


# 2-stage least squares for FGLS (FWLS)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print '\n2-stage least squares for FGLS (FWLS)'
resid1 = res2.resid[w==1.]
var1 = resid1.var(ddof=int(res2.df_model)+1)
resid2 = res2.resid[w!=1.]
var2 = resid2.var(ddof=int(res2.df_model)+1)
west = w.copy()
west[w!=1.] = np.sqrt(var2)/np.sqrt(var1)
res3 = sm.WLS(y2, X[:,[0,2]], 1./west).fit()
print 'feasible WLS beta estimates'
print res3.params
print 'feasible WLS stddev of beta'
print res3.bse


#..plt.show()
