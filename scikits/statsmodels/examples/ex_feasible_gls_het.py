# -*- coding: utf-8 -*-
"""Examples for linear model with heteroscedasticity estimated by feasible GLS

These are examples to check the results during developement.

The assumptions:

We have a linear model y = X*beta where the variance of an observation depends
on some explanatory variable Z (`exog_var`).
linear_model.WLS estimated the model for a given weight matrix
here we want to estimate also the weight matrix by two step or iterative WLS

Created on Wed Dec 21 12:28:17 2011

Author: Josef Perktold

"""

import numpy as np
from numpy.testing import assert_almost_equal

from scikits.statsmodels.regression.linear_model import OLS, WLS, GLS
from scikits.statsmodels.regression.feasible_gls import GLSHet, GLSHet2

examples = ['ex1']

if 'ex1' in examples:
    #from tut_ols_wls
    nsample = 100
    sig = 0.5
    x1 = np.linspace(0, 20, nsample)
    X = np.c_[x1, (x1-5)**2, np.ones(nsample)]
    np.random.seed(0)#9876789) #9876543)
    beta = [0.5, -0.015, 1.]
    y_true2 = np.dot(X, beta)
    w = np.ones(nsample)
    w[nsample*6//10:] = 4
    #y2[:nsample*6/10] = y_true2[:nsample*6/10] + sig*1. * np.random.normal(size=nsample*6/10)
    #y2[nsample*6/10:] = y_true2[nsample*6/10:] + sig*4. * np.random.normal(size=nsample*4/10)
    y2 = y_true2 + sig*w* np.random.normal(size=nsample)
    X2 = X[:,[0,2]]

    res_ols = OLS(y2, X2).fit()
    print 'OLS beta estimates'
    print res_ols.params
    print 'OLS stddev of beta'
    print res_ols.bse
    print '\nWLS'
    mod0 = GLSHet2(y2, X2, exog_var=w)
    res0 = mod0.fit()
    mod1 = GLSHet(y2, X2, exog_var=w)
    res1 = mod1.iterative_fit(2)
    print 'WLS beta estimates'
    print res1.params
    print res0.params
    print 'WLS stddev of beta'
    print res1.bse
    #compare with previous version GLSHet2, refactoring check
    #assert_almost_equal(res1.params, np.array([ 0.37642521,  1.51447662]))
    #this fails ???  more iterations? different starting weights?


    print res1.model.weights/res1.model.weights.max()
    #why is the error so small in the estimated weights ?
    assert_almost_equal(res1.model.weights/res1.model.weights.max(), 1./w, 14)
    print 'residual regression params'
    print res1.results_residual_regression.params
    print 'scale of model ?'
    print res1.scale
    print 'unweighted residual variance, note unweighted mean is not zero'
    print res1.resid.var()
    #Note weighted mean is zero:
    #(res1.model.weights * res1.resid).mean()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x1, y2, 'o')
    plt.plot(x1, y_true2, 'b-', label='true')
    plt.plot(x1, res1.fittedvalues, 'r-', label='fwls')
    plt.plot(x1, res_ols.fittedvalues, '--', label='ols')
    plt.legend()


    plt.show()
