'''Generic Maximum Likelihood Models'''

#This tutorial explains how to quickly implement new maximum likelihood models
#in ``statsmodels``. The `GenericLikelihoodModel
#<../../dev/generated/statsmodels.base.model.GenericLikelihoodModel.html#statsmodels.base.model.GenericLikelihoodModel>`_
#class eases the process by providing tools such as automatic numeric
#differentiation and a unified interface to ``scipy`` optimization functions.
#Using ``statsmodels``, users can fit new MLE models simply by "plugging-in" a
#log-likelihood function. 

#
#Negative Binomial Regression for Count Data
#-------------------------------------------

#Consider a negative binomial regression model for count data with
#log-likelihood (type NB-2) function expressed as:

#.. math::
#
#    \mathcal{L}(\beta_j; y, \alpha) = \sum_{i=1}^n y_i ln 
#    \left ( \frac{\alpha exp(X_i'\beta)}{1+\alpha exp(X_i'\beta)} \right ) -
#    \frac{1}{\alpha} ln(1+\alpha exp(X_i'\beta)) \\
#    + ln \Gamma (y_i + 1/\alpha) - ln \Gamma (y_i+1) - ln \Gamma (1/\alpha)

#with a matrix of regressors :math:`X`, a vector of coefficients :math:`\beta`,
#and the negative binomial heterogeneity parameter :math:`\alpha`. 

#Using the ``nbinom`` distribution from ``scipy``, we can write this likelihood
#simply as:

import numpy as np
from scipy.stats import nbinom
def _ll_nb2(y, X, beta, alph):
    mu = np.exp(np.dot(X, beta))
    size = 1/alph
    prob = size/(size+mu)
    ll = nbinom.logpmf(y, size, prob)
    return ll


#New Model Class
#---------------

#We create a new model class which inherits from ``GenericLikelihoodModel``:

from statsmodels.base.model import GenericLikelihoodModel
class NBin(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(NBin, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        alph = params[-1]
        beta = params[:self.exog.shape[1]]
        ll = _ll_nb2(self.endog, self.exog, beta, alph)
        return -ll 
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params == None:
            # Reasonable starting values
            start_params = np.append(np.zeros(self.exog.shape[1]), .5)
            start_params[0] = np.log(self.endog.mean())
        return super(NBin, self).fit(start_params=start_params, 
                                     maxiter=maxiter, maxfun=maxfun, 
                                     **kwds) 

#Two important things to notice: 

#+ ``nloglikeobs``: This function should return one evaluation of the negative log-likelihood function per observation in your dataset (i.e. rows of the endog/X matrix). 
#+ ``start_params``: A one-dimensional array of starting values needs to be provided. The size of this array determines the number of parameters that will be used in optimization.
   
#That's it! You're done!

#Usage Example
#-------------

#The `Medpar <http://vincentarelbundock.github.com/Rdatasets/doc/medpar.html>`_
#dataset is hosted in CSV format at the `Rdatasets repository
#<http://vincentarelbundock.github.com/Rdatasets>`_. We use the ``read_csv``
#function from the `Pandas library <http://pandas.pydata.org>`_ to load the data
#in memory. We then print the first few columns: 

import pandas as pd
url = 'http://vincentarelbundock.github.com/Rdatasets/csv/medpar.csv'
medpar = pd.read_csv(url)
medpar.head()

#The model we are interested in has a vector of non-negative integers as
#dependent variable (``los``), and 5 regressors: ``Intercept``, ``type2``,
#``type3``, ``hmo``, ``white``.

#For estimation, we need to create 2 numpy arrays (pandas DataFrame should also
#work): a 1d array of length *N* to hold ``los`` values, and a *N* by 5
#array to hold our 5 regressors. These arrays can be constructed manually or
#using any number of helper functions; the details matter little for our current
#purposes.  Here, we build the arrays we need using the `Patsy
#<http://patsy.readthedocs.org>`_ package:

import patsy
y, X = patsy.dmatrices('los~type2+type3+hmo+white', medpar)
print y[:5]
print X[:5] 

#Then, we fit the model and extract some information: 

mod = NBin(y, X)
res = mod.fit()

# Extract parameter estimates, standard errors, p-values, AIC, etc.:
res.params
res.bse
res.pvalues
res.aic

#As usual, you can obtain a full list of available information by typing
#``dir(res)``. 
# 
#To ensure that the above results are sound, we compare them to results
# obtained using the MASS implementation for R:: 
#
#    url = 'http://vincentarelbundock.github.com/Rdatasets/csv/medpar.csv'
#    medpar = read.csv(url)
#    f = los~factor(type)+hmo+white
#    
#    library(MASS)
#    mod = glm.nb(f, medpar)
#    coef(summary(mod))
#                     Estimate Std. Error   z value      Pr(>|z|)
#    (Intercept)    2.31027893 0.06744676 34.253370 3.885556e-257
#    factor(type)2  0.22124898 0.05045746  4.384861  1.160597e-05
#    factor(type)3  0.70615882 0.07599849  9.291748  1.517751e-20
#    hmo           -0.06795522 0.05321375 -1.277024  2.015939e-01
#    white         -0.12906544 0.06836272 -1.887951  5.903257e-02

#Numerical precision 
#^^^^^^^^^^^^^^^^^^^ 

#The ``statsmodels`` and ``R`` parameter estimates agree up to the fourth
#decimal. The standard errors, however, agree only up to the second decimal.
#This discrepancy may be the result of imprecision in our Hessian numerical
#estimates. In the current context, the difference between ``MASS`` and
#``statsmodels`` standard error estimates is substantively irrelevant, but it
#highlights the fact that users who need very precise estimates may not always
#want to rely on default settings when using numerical derivatives. In such
#cases, it may be better to use analytical derivatives with the `LikelihoodModel
#<../../dev/generated/statsmodels.base.model.GenericLikelihoodModel.html#statsmodels.base.model.GenericLikelihoodModel>`_
#class. 


