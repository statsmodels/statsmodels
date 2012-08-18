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
import numpy as np
class NBin(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(NBin, self).__init__(endog, exog, **kwds)
        # Reasonable starting values
        self.start_params = np.append(np.zeros(self.exog.shape[1]), .5)
        self.start_params[0] = np.log(self.endog.mean())
    def nloglikeobs(self, params):
        alph = params[-1]
        beta = params[:self.exog.shape[1]]
        ll = _ll_nb2(self.endog, self.exog, beta, alph)
        return -ll 
    def fit(self, **kwds):
        return super(NBin, self).fit(start_params=self.start_params, 
                                     maxiter=10000, maxfun=5000, **kwds) 

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
#To be sure that the above results are sound, compare to an equivalent model
#estimated with R::
#
#    > library(MASS)
#    > library(COUNT)
#    > data(medpar)
#    > f <- los~factor(type)+hmo+white
#    > ml.nb2(f, medpar)
#    Estimate         SE         Z        LCL         UCL
#    (Intercept)    2.31214519 0.06794358 34.030372  2.1789758 2.445314604
#    factor(type)2  0.22049993 0.05056730  4.360524  0.1213880 0.319611832
#    factor(type)3  0.70437929 0.07606068  9.260754  0.5553003 0.853458232
#    hmo           -0.06809686 0.05323976 -1.279060 -0.1724468 0.036253069
#    white         -0.13052184 0.06853619 -1.904422 -0.2648528 0.003809104
#    alpha          0.44522693 0.01978011 22.508817  0.4064579 0.483995950


