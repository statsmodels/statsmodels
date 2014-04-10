# -*- coding: utf-8 -*-
'''
Author: Vincent Arel-Bundock <varel@umich.edu>
Date: 2012-08-25

This example file implements 5 variations of the negative binomial regression
model for count data: NB-P, NB-1, NB-2, geometric and left-truncated.

The NBin class inherits from the GenericMaximumLikelihood statsmodels class
which provides automatic numerical differentiation for the score and hessian.

NB-1, NB-2 and geometric are implemented as special cases of the NB-P model
described in Greene (2008) Functional forms for the negative binomial model for
count data. Economics Letters, v99n3.

Tests are included to check how NB-1, NB-2 and geometric coefficient estimates
compare to equivalent models in R. Results usually agree up to the 4th digit.

The NB-P and left-truncated model results have not been compared to other
implementations. Note that NB-P appears to only have been implemented in the
LIMDEP software.
'''

import numpy as np
from scipy.special import gammaln
from scipy.stats import nbinom
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModelResults
import statsmodels.api as sm

#### Negative Binomial Log-likelihoods ####
def _ll_nbp(y, X, beta, alph, Q):
    '''
    Negative Binomial Log-likelihood -- type P

    References:

    Greene, W. 2008. "Functional forms for the negtive binomial model
        for count data". Economics Letters. Volume 99, Number 3, pp.585-590.
    Hilbe, J.M. 2011. "Negative binomial regression". Cambridge University Press.

    Following notation in Greene (2008), with negative binomial heterogeneity
	parameter :math:`\alpha`:

    .. math::

        \lambda_i = exp(X\beta)\\
        \theta = 1 / \alpha \\
        g_i = \theta \lambda_i^Q \\
        w_i = g_i/(g_i + \lambda_i) \\
        r_i = \theta / (\theta+\lambda_i) \\
        ln \mathcal{L}_i = ln \Gamma(y_i+g_i) - ln \Gamma(1+y_i) + g_iln (r_i) + y_i ln(1-r_i)
    '''
    mu = np.exp(np.dot(X, beta))
    size = 1/alph*mu**Q
    prob = size/(size+mu)
    ll = nbinom.logpmf(y, size, prob)
    return ll
def _ll_nb1(y, X, beta, alph):
    '''Negative Binomial regression (type 1 likelihood)'''
    ll = _ll_nbp(y, X, beta, alph, Q=1)
    return ll
def _ll_nb2(y, X, beta, alph):
    '''Negative Binomial regression (type 2 likelihood)'''
    ll = _ll_nbp(y, X, beta, alph, Q=0)
    return ll
def _ll_geom(y, X, beta):
    '''Geometric regression'''
    ll = _ll_nbp(y, X, beta, alph=1, Q=0)
    return ll
def _ll_nbt(y, X, beta, alph, C=0):
    '''
    Negative Binomial (truncated)

    Truncated densities for count models (Cameron & Trivedi, 2005, 680):

    .. math::

        f(y|\beta, y \geq C+1) = \frac{f(y|\beta)}{1-F(C|\beta)}
    '''
    Q = 0
    mu = np.exp(np.dot(X, beta))
    size = 1/alph*mu**Q
    prob = size/(size+mu)
    ll = nbinom.logpmf(y, size, prob) - np.log(1 - nbinom.cdf(C, size, prob))
    return ll

#### Model Classes ####
class NBin(GenericLikelihoodModel):
    '''
    Negative Binomial regression

    Parameters
    ----------
    endog : array-like
        1-d array of the response variable.
    exog : array-like
        `exog` is an n x p array where n is the number of observations and p
        is the number of regressors including the intercept if one is
        included in the data.
    ll_type: string
        log-likelihood type
        `nb2`: Negative Binomial type-2 (most common)
        `nb1`: Negative Binomial type-1
        `nbp`: Negative Binomial type-P (Greene, 2008)
        `nbt`: Left-truncated Negative Binomial (type-2)
        `geom`: Geometric regression model
    C: integer
        Cut-point for `nbt` model
    '''
    def __init__(self, endog, exog, ll_type='nb2', C=0, **kwds):
        self.exog = np.array(exog)
        self.endog = np.array(endog)
        self.C = C
        super(NBin, self).__init__(endog, exog, **kwds)
        # Check user input
        if ll_type not in ['nb2', 'nb1', 'nbp', 'nbt', 'geom']:
            raise NameError('Valid ll_type are: nb2, nb1, nbp,  nbt, geom')
        self.ll_type = ll_type
        # Starting values (assumes first column of exog is constant)
        if ll_type == 'geom':
            self.start_params_default = np.zeros(self.exog.shape[1])
        elif ll_type == 'nbp':
            # Greene recommends starting NB-P at NB-2
            start_mod = NBin(endog, exog, 'nb2')
            start_res = start_mod.fit(disp=False)
            self.start_params_default = np.append(start_res.params, 0)
        else:
            self.start_params_default = np.append(np.zeros(self.exog.shape[1]), .5)
        self.start_params_default[0] = np.log(self.endog.mean())
        # Define loglik based on ll_type argument
        if ll_type == 'nb1':
            self.ll_func = _ll_nb1
        elif ll_type == 'nb2':
            self.ll_func = _ll_nb2
        elif ll_type == 'geom':
            self.ll_func = _ll_geom
        elif ll_type == 'nbp':
            self.ll_func = _ll_nbp
        elif ll_type == 'nbt':
            self.ll_func = _ll_nbt
    def nloglikeobs(self, params):
        alph = params[-1]
        beta = params[:self.exog.shape[1]]
        if self.ll_type == 'geom':
            return -self.ll_func(self.endog, self.exog, beta)
        elif self.ll_type == 'nbt':
            return -self.ll_func(self.endog, self.exog, beta, alph, self.C)
        elif self.ll_type == 'nbp':
            Q = params[-2]
            return -self.ll_func(self.endog, self.exog, beta, alph, Q)
        else:
            return -self.ll_func(self.endog, self.exog, beta, alph)
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params==None:
            countfit = super(NBin, self).fit(start_params=self.start_params_default,
                                             maxiter=maxiter, maxfun=maxfun, **kwds)
        else:
            countfit = super(NBin, self).fit(start_params=start_params,
                                             maxiter=maxiter, maxfun=maxfun, **kwds)
        countfit = CountResults(self, countfit)
        return countfit

class CountResults(GenericLikelihoodModelResults):
    def __init__(self, model, mlefit):
        self.model = model
        self.__dict__.update(mlefit.__dict__)
    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):
        top_left = [('Dep. Variable:', None),
                     ('Model:', [self.model.__class__.__name__]),
                     ('Method:', ['MLE']),
                     ('Date:', None),
                     ('Time:', None),
                     ('Converged:', ["%s" % self.mle_retvals['converged']])
                      ]
        top_right = [('No. Observations:', None),
                     ('Log-Likelihood:', None),
                     ]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"
        #boiler plate
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        # for top of table
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, #[],
                          yname=yname, xname=xname, title=title)
        # for parameters, etc
        smry.add_table_params(self, yname=yname_list, xname=xname, alpha=alpha,
                             use_t=True)
        return smry

#### Score function for NB-P ####
from scipy.special import digamma
def _score_nbp(y, X, beta, thet, Q):
    '''
    Negative Binomial Score -- type P likelihood from Greene (2007)
    .. math::

        \lambda_i = exp(X\beta)\\
        g_i = \theta \lambda_i^Q \\
        w_i = g_i/(g_i + \lambda_i) \\
        r_i = \theta / (\theta+\lambda_i) \\
        A_i = \left [ \Psi(y_i+g_i) - \Psi(g_i) + ln w_i \right ] \\
        B_i = \left [ g_i (1-w_i) - y_iw_i \right ] \\
        \partial ln \mathcal{L}_i / \partial
            \begin{pmatrix} \lambda_i \\ \theta \\ Q \end{pmatrix}=
            [A_i+B_i]
            \begin{pmatrix} Q/\lambda_i \\ 1/\theta \\ ln(\lambda_i) \end{pmatrix}
            -B_i
            \begin{pmatrix} 1/\lambda_i\\ 0 \\ 0 \end{pmatrix} \\
        \frac{\partial \lambda}{\partial \beta} = \lambda_i \mathbf{x}_i \\
        \frac{\partial \mathcal{L}_i}{\partial \beta} =
            \left (\frac{\partial\mathcal{L}_i}{\partial \lambda_i} \right )
            \frac{\partial \lambda_i}{\partial \beta}
    '''
    lamb = np.exp(np.dot(X, beta))
    g = thet * lamb**Q
    w = g / (g + lamb)
    r = thet / (thet+lamb)
    A = digamma(y+g) - digamma(g) + np.log(w)
    B = g*(1-w) - y*w
    dl = (A+B) * Q/lamb - B * 1/lamb
    dt = (A+B) * 1/thet
    dq = (A+B) * np.log(lamb)
    db = X * (dl * lamb)[:,np.newaxis]
    sc = np.array([dt.sum(), dq.sum()])
    sc = np.concatenate([db.sum(axis=0), sc])
    return sc

#### Tests ####
from statsmodels.compat.python import urlopen
from numpy.testing import assert_almost_equal
import pandas
import patsy
medpar = pandas.read_csv(urlopen('http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/medpar.csv'))
mdvis = pandas.read_csv(urlopen('http://vincentarelbundock.github.com/Rdatasets/csv/COUNT/mdvis.csv'))

# NB-2
'''
# R v2.15.1
library(MASS)
library(COUNT)
data(medpar)
f <- los~factor(type)+hmo+white
mod <- glm.nb(f, medpar)
summary(mod)
Call:
glm.nb(formula = f, data = medpar, init.theta = 2.243376203,
    link = log)

Deviance Residuals:
    Min       1Q   Median       3Q      Max
-2.4671  -0.9090  -0.2693   0.4320   3.8668

Coefficients:
              Estimate Std. Error z value Pr(>|z|)
(Intercept)    2.31028    0.06745  34.253  < 2e-16 ***
factor(type)2  0.22125    0.05046   4.385 1.16e-05 ***
factor(type)3  0.70616    0.07600   9.292  < 2e-16 ***
hmo           -0.06796    0.05321  -1.277    0.202
white         -0.12907    0.06836  -1.888    0.059 .
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for Negative Binomial(2.2434) family taken to be 1)

    Null deviance: 1691.1  on 1494  degrees of freedom
Residual deviance: 1568.1  on 1490  degrees of freedom
AIC: 9607

Number of Fisher Scoring iterations: 1


              Theta:  2.2434
          Std. Err.:  0.0997

 2 x log-likelihood:  -9594.9530
'''

def test_nb2():
    y, X = patsy.dmatrices('los ~ C(type) + hmo + white', medpar)
    y = np.array(y)[:,0]
    nb2 = NBin(y,X,'nb2').fit(maxiter=10000, maxfun=5000)
    assert_almost_equal(nb2.params,
                        [2.31027893349935, 0.221248978197356, 0.706158824346228,
                         -0.067955221930748, -0.129065442248951, 0.4457567],
                        decimal=2)

# NB-1
'''
# R v2.15.1
# COUNT v1.2.3
library(COUNT)
data(medpar)
f <- los~factor(type)+hmo+white
ml.nb1(f, medpar)

                 Estimate         SE          Z         LCL         UCL
(Intercept)    2.34918407 0.06023641 38.9994023  2.23112070  2.46724744
factor(type)2  0.16175471 0.04585569  3.5274735  0.07187757  0.25163186
factor(type)3  0.41879257 0.06553258  6.3906006  0.29034871  0.54723643
hmo           -0.04533566 0.05004714 -0.9058592 -0.14342805  0.05275673
white         -0.12951295 0.06071130 -2.1332593 -0.24850710 -0.01051880
alpha          4.57898241 0.22015968 20.7984603  4.14746943  5.01049539
'''

#def test_nb1():
    #y, X = patsy.dmatrices('los ~ C(type) + hmo + white', medpar)
    #y = np.array(y)[:,0]
    ## TODO: Test fails with some of the other optimization methods
    #nb1 = NBin(y,X,'nb1').fit(method='ncg', maxiter=10000, maxfun=5000)
    #assert_almost_equal(nb1.params,
						#[2.34918407014186, 0.161754714412848, 0.418792569970658,
                         #-0.0453356614650342, -0.129512952033423, 4.57898241219275],
						#decimal=2)

# NB-Geometric
'''
MASS v7.3-20
R v2.15.1
library(MASS)
data(medpar)
f <- los~factor(type)+hmo+white
mod <- glm(f, family=negative.binomial(1), data=medpar)
summary(mod)
Call:
glm(formula = f, family = negative.binomial(1), data = medpar)

Deviance Residuals:
    Min       1Q   Median       3Q      Max
-1.7942  -0.6545  -0.1896   0.3044   2.6844

Coefficients:
              Estimate Std. Error t value Pr(>|t|)
(Intercept)    2.30849    0.07071  32.649  < 2e-16 ***
factor(type)2  0.22121    0.05283   4.187 2.99e-05 ***
factor(type)3  0.70599    0.08092   8.724  < 2e-16 ***
hmo           -0.06779    0.05521  -1.228   0.2197
white         -0.12709    0.07169  -1.773   0.0765 .
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for Negative Binomial(1) family taken to be 0.5409721)

    Null deviance: 872.29  on 1494  degrees of freedom
Residual deviance: 811.95  on 1490  degrees of freedom
AIC: 9927.3

Number of Fisher Scoring iterations: 5
'''

#def test_geom():
    #y, X = patsy.dmatrices('los ~ C(type) + hmo + white', medpar)
    #y = np.array(y)[:,0]
    ## TODO: remove alph from geom params
    #geom = NBin(y,X,'geom').fit(maxiter=10000, maxfun=5000)
    #assert_almost_equal(geom.params,
						#[2.3084850946241, 0.221206159108742, 0.705986369841159,
                         #-0.0677871843613577, -0.127088772164963],
						#decimal=4)

test_nb2()
