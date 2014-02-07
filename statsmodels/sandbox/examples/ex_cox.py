"""Example for scikits.statsmodels.sandbox.survival2.CoxPH"""

#An example for the CoxPH class

import numpy as np
from statsmodels.sandbox.survival2 import CoxPH, Survival, CoxResults
from statsmodels.datasets import ovarian_cancer
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

##Maybe we are interested in some classification of "expressed"
##and "not expressed" rather than just the floating point expression
##values. The dataset does not included this, so, for demonstration
##purposes, say that a gene is "expressed" (1) if its expression value
##is greater than the mean for the gene, and not expressed (0)
##otherwise. We can create an indicator variable "expressed", that
##represents whether the first gene in the dataset is "expressed"

gene1 = exog[:,0]
expressed = gene1 > gene1.mean()
##replacing the column for th first gene with the indicatore variable
exog[:,0] = expressed
##create a model with the new exogenous variable
cox2 = CoxPH(surv, exog)

##We can now create a stratified cox model using the "stratify" method
##stratify using the index 0 variable (the indicator variable)
##copy=False alters the object cox. Default is copy=True, to create
##a new CoxPH object
cox2.stratify(0,copy=False)
##This can also be achieved by specifying an argument to the
##"strata" parameter when the model is instantiated

##The stratified model now has the stratified variable stored seperately
print "\nstratified cox model"
print "sratified variable"
print cox2.strata[range(5),:]
print "unstratified variables"
print cox2.exog[range(5),:]

##Just like the previous model, we can fit the model
results2 = cox2.fit()

##This new CoxResults object has access to all the methods of a standard
##cox model
print "estimated parameters with stratification"
print results2.params

##Planned developments for the CoxPH class:
##Currently only works for right-censored data. Handling for more
##complex censoring is forthcoming
##fitting with time-dependent covariates and time-dependent
##coefficients
##quick summary method
##Residuals and diagnostic plots
##Hypothesis tests against restricted models

print results2.summary()
results2.diagnostics()
do_plots = True
if do_plots:
    import matplotlib.pyplot as plt
    plt.figure()
    results2.martingale_plot(1)
    plt.title('martingale plot')
    plt.figure()
    results2.deviance_plot()
    plt.title('deviance plot')
    plt.show()
