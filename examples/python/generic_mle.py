
## Maximum Likelihood Estimation (Generic models)

# This tutorial explains how to quickly implement new maximum likelihood models in `statsmodels`. We give two examples: 
# 
# 1. Probit model for binary dependent variables
# 2. Negative binomial model for count data
# 
# The `GenericLikelihoodModel` class eases the process by providing tools such as automatic numeric differentiation and a unified interface to ``scipy`` optimization functions. Using ``statsmodels``, users can fit new MLE models simply by "plugging-in" a log-likelihood function. 

# ## Example 1: Probit model
from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


# The ``Spector`` dataset is distributed with ``statsmodels``. You can access a vector of values for the dependent variable (``endog``) and a matrix of regressors (``exog``) like this:

data = sm.datasets.spector.load_pandas()
exog = data.exog
endog = data.endog
print(sm.datasets.spector.NOTE)
print(data.exog.head())


# Them, we add a constant to the matrix of regressors:

exog = sm.add_constant(exog, prepend=True)


# To create your own Likelihood Model, you simply need to overwrite the loglike method.

class MyProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        return stats.norm.logcdf(q*np.dot(exog, params)).sum()


# Estimate the model and print(a summary:

sm_probit_manual = MyProbit(endog, exog).fit()
print(sm_probit_manual.summary())


# Compare your Probit implementation to ``statsmodels``' "canned" implementation:

sm_probit_canned = sm.Probit(endog, exog).fit()


print(sm_probit_canned.params)
print(sm_probit_manual.params)


print(sm_probit_canned.cov_params())
print(sm_probit_manual.cov_params())


# Notice that the ``GenericMaximumLikelihood`` class provides automatic differentiation, so we didn't have to provide Hessian or Score functions in order to calculate the covariance estimates.

# 
# 
# ## Example 2: Negative Binomial Regression for Count Data
# 
# Consider a negative binomial regression model for count data with
# log-likelihood (type NB-2) function expressed as:
# 
# $$
#     \mathcal{L}(\beta_j; y, \alpha) = \sum_{i=1}^n y_i ln 
#     \left ( \frac{\alpha exp(X_i'\beta)}{1+\alpha exp(X_i'\beta)} \right ) -
#     \frac{1}{\alpha} ln(1+\alpha exp(X_i'\beta)) + ln \Gamma (y_i + 1/\alpha) - ln \Gamma (y_i+1) - ln \Gamma (1/\alpha)
# $$
# 
# with a matrix of regressors $X$, a vector of coefficients $\beta$,
# and the negative binomial heterogeneity parameter $\alpha$. 
# 
# Using the ``nbinom`` distribution from ``scipy``, we can write this likelihood
# simply as:
# 

import numpy as np
from scipy.stats import nbinom


def _ll_nb2(y, X, beta, alph):
    mu = np.exp(np.dot(X, beta))
    size = 1/alph
    prob = size/(size+mu)
    ll = nbinom.logpmf(y, size, prob)
    return ll


# ### New Model Class
# 
# We create a new model class which inherits from ``GenericLikelihoodModel``:

from statsmodels.base.model import GenericLikelihoodModel


class NBin(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(NBin, self).__init__(endog, exog, **kwds)
        
    def nloglikeobs(self, params):
        alph = params[-1]
        beta = params[:-1]
        ll = _ll_nb2(self.endog, self.exog, beta, alph)
        return -ll 
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('alpha')
        if start_params == None:
            # Reasonable starting values
            start_params = np.append(np.zeros(self.exog.shape[1]), .5)
            # intercept
            start_params[-2] = np.log(self.endog.mean())
        return super(NBin, self).fit(start_params=start_params, 
                                     maxiter=maxiter, maxfun=maxfun, 
                                     **kwds) 


# Two important things to notice: 
# 
# + ``nloglikeobs``: This function should return one evaluation of the negative log-likelihood function per observation in your dataset (i.e. rows of the endog/X matrix). 
# + ``start_params``: A one-dimensional array of starting values needs to be provided. The size of this array determines the number of parameters that will be used in optimization.
#    
# That's it! You're done!
# 
# ### Usage Example
# 
# The [Medpar](http://vincentarelbundock.github.com/Rdatasets/doc/COUNT/medpar.html)
# dataset is hosted in CSV format at the [Rdatasets repository](http://vincentarelbundock.github.com/Rdatasets). We use the ``read_csv``
# function from the [Pandas library](http://pandas.pydata.org) to load the data
# in memory. We then print(the first few columns: 
# 

import statsmodels.api as sm


medpar = sm.datasets.get_rdataset("medpar", "COUNT", cache=True).data

medpar.head()


# The model we are interested in has a vector of non-negative integers as
# dependent variable (``los``), and 5 regressors: ``Intercept``, ``type2``,
# ``type3``, ``hmo``, ``white``.
# 
# For estimation, we need to create two variables to hold our regressors and the outcome variable. These can be ndarrays or pandas objects.

y = medpar.los
X = medpar[["type2", "type3", "hmo", "white"]]
X["constant"] = 1


# Then, we fit the model and extract some information: 

mod = NBin(y, X)
res = mod.fit()


#  Extract parameter estimates, standard errors, p-values, AIC, etc.:

print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('P-values: ', res.pvalues)
print('AIC: ', res.aic)


# As usual, you can obtain a full list of available information by typing
# ``dir(res)``.
# We can also look at the summary of the estimation results.

print(res.summary())


# ### Testing

# We can check the results by using the statsmodels implementation of the Negative Binomial model, which uses the analytic score function and Hessian.

res_nbin = sm.NegativeBinomial(y, X).fit(disp=0)
print(res_nbin.summary())


print(res_nbin.params)


print(res_nbin.bse)


# Or we could compare them to results obtained using the MASS implementation for R:
# 
#     url = 'http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/medpar.csv'
#     medpar = read.csv(url)
#     f = los~factor(type)+hmo+white
#     
#     library(MASS)
#     mod = glm.nb(f, medpar)
#     coef(summary(mod))
#                      Estimate Std. Error   z value      Pr(>|z|)
#     (Intercept)    2.31027893 0.06744676 34.253370 3.885556e-257
#     factor(type)2  0.22124898 0.05045746  4.384861  1.160597e-05
#     factor(type)3  0.70615882 0.07599849  9.291748  1.517751e-20
#     hmo           -0.06795522 0.05321375 -1.277024  2.015939e-01
#     white         -0.12906544 0.06836272 -1.887951  5.903257e-02
# 
# ### Numerical precision 
# 
# The ``statsmodels`` generic MLE and ``R`` parameter estimates agree up to the fourth decimal. The standard errors, however, agree only up to the second decimal. This discrepancy is the result of imprecision in our Hessian numerical estimates. In the current context, the difference between ``MASS`` and ``statsmodels`` standard error estimates is substantively irrelevant, but it highlights the fact that users who need very precise estimates may not always want to rely on default settings when using numerical derivatives. In such cases, it is better to use analytical derivatives with the ``LikelihoodModel`` class.
# 
