# -*- coding: utf-8 -*-
"""
Author: Josef Perktold
License: BSD-3

"""

from __future__ import print_function
import numpy as np
from scipy import stats

class NonlinearDeltaCov(object):
    '''Asymptotic covariance by Deltamethod

    the function is designed for 2d array, with rows equal to
    the number of equations and columns equal to the number
    of parameters. 1d params work by chance ?

    fun: R^{m*k) -> R^{m}  where m is number of equations and k is
    the number of parameters.

    equations follow Greene

    This class does not use any caching. The intended usage is as a helper
    function. Extra methods have been added for convenience but might move
    to calling functions.

    '''
    def __init__(self, fun, params, cov_params, grad=None, func_args=None):
        self.fun = fun
        self.params = params
        self.cov_params = cov_params
        self._grad = grad
        self.func_args = func_args if func_args is not None else ()
        if func_args is not None:
            raise NotImplementedError('func_args not yet implemented')

    def grad(self, params=None, **kwds):

        if params is None:
            params = self.params
        if self._grad is not None:
            return self._grad(params)
        else:
            # copied from discrete_margins
            try:
                from statsmodels.tools.numdiff import approx_fprime_cs
                jac = approx_fprime_cs(params, self.fun)
            except TypeError:  # norm.cdf doesn't take complex values
                from statsmodels.tools.numdiff import approx_fprime
                jac = approx_fprime(params, self.fun)

            return jac

    def cov(self):
        g = self.grad()
        covar = np.dot(np.dot(g, self.cov_params), g.T)
        return covar

    def predicted(self):
        # rename: misnomer, this is the MLE of the fun
        predicted = self.fun(self.params)

        # TODO: why do I need to squeeze in poisson example
        if predicted.ndim > 1:
            predicted = predicted.squeeze()
        return predicted

    def wald_test(self, value):
        # TODO: add use_t option or not?
        m = self.predicted()
        v = self.cov()
        df_constraints = np.size(m)
        diff = m - value
        lmstat = np.dot(np.dot(diff.T, np.linalg.inv(v)), diff)
        return lmstat, stats.chi2.sf(lmstat, df_constraints)


    def se_vectorized(self):
        """standard error for each equation (row) treated separately

        """
        g = self.grad()
        var = (np.dot(g, self.cov_params) * g).sum(-1)
        return np.sqrt(var)


    def conf_int(self, alpha=0.05, use_t=False, df=None, var_extra=None,
                 predicted=None, se=None):
        # TODO: predicted and se as arguments to avoid duplicate calculations, keep?
        if not use_t:
            dist = stats.norm
            dist_args = ()
        else:
            if df is None:
                raise ValueError('t distribution requires df')
            dist = stats.t
            dist_args = (df,)

        if predicted is None:
            predicted = self.predicted()
        if se is None:
            se = self.se_vectorized()
        if var_extra is not None:
            se = np.sqrt(se**2 + var_extra)

        q = dist.ppf(1 - alpha / 2., *dist_args)
        lower = predicted - q * se
        upper = predicted + q * se
        ci = np.column_stack((lower, upper))
        if ci.shape[1] !=2:
            raise RuntimeError('something wrong: ci not 2 columns')
        return ci


    def summary(self, xname=None, alpha=0.05, title=None, use_t=False, df=None):
        """Summarize the Results of the hypothesis test
        """
        # this is an experimental reuse of ContrastResults
        from statsmodels.stats.contrast import ContrastResults
        predicted = self.predicted()
        se = self.se_vectorized()
        # TODO check shape for scalar case, ContrastResults requires iterable
        predicted = np.atleast_1d(predicted)
        if predicted.ndim > 1:
            predicted = predicted.squeeze()
        se = np.atleast_1d(se)

        statistic = predicted / se
        if use_t:
            df_resid = df
            cr = ContrastResults(effect=predicted, t=statistic, sd=se,
                                   df_denom=df_resid)
        else:
            cr = ContrastResults(effect=predicted, statistic=statistic, sd=se,
                                   df_denom=None, distribution='norm')

        return cr.summary(xname=xname, alpha=alpha, title=title)
