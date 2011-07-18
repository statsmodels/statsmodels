"""Example for scikits.statsmodels.sandbox.survival2.CoxPH"""

#An example for the CoxPH class

import numpy as np
from scikits.statsmodels.sandbox.survival2 import CoxPH, Survival, CoxResults
from scikits.statsmodels.datasets import ovarian_cancer
from scipy import stats

##Get the data
##The ovarian_cancer dataset contains data on cancer survival
##and gene expression
dta = ovarian_cancer.load()
darray = np.asarray(dta['data'])

##darray is a numpy array whose index zero column is a vector
##survival times, and whose index one column is a vector of
##censoring indicators, 0 for censored, 1 for an event.
##The other columns contain expression values for various genes

##Now, get an array of the exogenous variables. We'll use the
##first 4 genes in darray

exog = darray[:,range(2,6)]

##Fit the model
##0 is the index in darray of the times, and 1 is the index
##of the censoring variable
surv = Survival(0, censoring=1, data=darray)
##CoxPH takes a Survival object as input
cox = CoxPH(surv, exog)
##cox = CoxPH(surv, range(2,6), data=darray)
##is equivalent
results = cox.fit()
##results is a CoxResults object
print "estimated parameters"
print results.params

##Test global null and individual coefficients
wald = results.wald_test()
df = len(results.params)
pval = stats.chi2.sf(wald, df)
print "wald test"
print "test stat:", wald
print "df:", df
print "p-value:", pval
##respective function are also available for score
##and likelihood-ratio tests
con_int = results.conf_int()
print "confidence intervals for parameters"
print con_int
test_coeffs = results.test_coefficients()
print "tests for individual coefficients"
print test_coeffs
##for test_coeffs
##index zero column is the coefficient
##index one column is the standard error of the coefficient
##index two column is the z-score
##index three column is the p-value

##Planned developments for the CoxPH class:
##Currently only works for right-censored data. Handling for more
##complex censoring is forthcoming
##stratified Cox models
##fitting with time-dependent covariates and time-dependent
##coefficients
##quick summary method
##Residuals and diagnostic plots
##Hypothesis tests against restricted models
