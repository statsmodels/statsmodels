# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:53:27 2018

Author: Josef Perktold

"""


import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd

from statsmodels.discrete.discrete_model import Poisson
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.base._screening import VariableScreening


class PoissonPenalized(PenalizedMixin, Poisson):
    pass


def test_poisson_screening():
    # this is mostly a dump of my trial notebook
    # number of exog candidates is reduced to 500 to reduce time
    np.random.seed(987865)

    nobs, k_vars = 100, 500
    k_nonzero = 5
    x = (np.random.rand(nobs, k_vars) + 1.* (np.random.rand(nobs, 1)-0.5)) * 2 - 1
    x *= 1.2

    x = (x - x.mean()) / x.std(0)
    x[:, 0] = 1
    beta = np.zeros(k_vars)
    idx_non_zero_true = [0, 100, 300, 400, 411]
    beta[idx_non_zero_true] = 1. / np.arange(1, k_nonzero + 1)
    beta = np.sqrt(beta)  # make small coefficients larger
    linpred = x.dot(beta)
    mu = np.exp(linpred)
    y = np.random.poisson(mu)

    xnames_true = ['var%4d' % ii for ii in idx_non_zero_true]
    xnames_true[0] = 'const'
    parameters = pd.DataFrame(beta[idx_non_zero_true], index=xnames_true, columns=['true'])

    xframe_true = pd.DataFrame(x[:, idx_non_zero_true], columns=xnames_true)
    res_oracle = Poisson(y, xframe_true).fit()
    parameters['oracle'] = res_oracle.params

    mod_initial = PoissonPenalized(y, np.ones(nobs), pen_weight=nobs * 5)
    base_class = Poisson

    screener = VariableScreening(mod_initial, base_class)
    exog_candidates = x[:, 1:]
    res_screen = screener.screen_vars(exog_candidates, maxiter=10)

    res_screen.idx_nonzero

    res_screen.results_final


    xnames = ['var%4d' % ii for ii in res_screen.idx_nonzero]
    xnames[0] = 'const'

    # smoke test
    res_screen.results_final.summary(xname=xnames)
    res_screen.results_pen.summary()
    assert_equal(res_screen.results_final.mle_retvals['converged'], True)

    ps = pd.Series(res_screen.results_final.params, index=xnames, name='final')
    parameters = parameters.join(ps, how='outer')

    assert_allclose(parameters['oracle'], parameters['final'], atol=5e-6)
