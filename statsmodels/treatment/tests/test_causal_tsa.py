# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:51:24 2017

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from statsmodels.treatment.panel_effect import (
            PanelATTBasic, OLSNonNegative, OLSSimplexConstrained)

# hack to switch off print
skip_print = True #False
def myprint(*x):
    if not skip_print:
        print(*x)

def test_causal_tsa():
    # this just dumps the example into a test
    # assert is that estimated ATT is within rtol=0.1 of true/DGP value
    nobs = 50
    nobs_pre = 30
    k_factors = 2
    k_control = 10
    effect_trend = 0.5
    shift = 1 / effect_trend
    trend = 0.5 * np.arange(nobs)
    #trend_new = np.maximum(np.arange(nobs) - nobs_pre, 0)
    trend_new = np.zeros(nobs)
    trend_new[nobs_pre:] = np.arange(nobs - nobs_pre) + shift
    np.random.seed(987456)

    sig_f = 0.5
    factor = sig_f * np.random.randn(nobs, k_factors)
    load = 1 + np.random.randn(k_factors , k_control + 1)
    noise_x = 0.1 * np.random.randn(nobs, k_control)

    x0 = factor.dot(load[:, 0]) + 0.1 * np.random.randn(nobs)
    x1 = factor.dot(load[:, 1:]) + 0.2 * noise_x
    #ex1 = np.column_stack((np.ones(nobs), trend, x1))
    #beta = np.ones(1 + 1 + 1)
    sig_e = 0.5
    y0 = trend + x0 - effect_trend * trend_new + sig_e * np.random.randn(nobs)
    y1 = (1 + 0.25 * np.random.rand(k_control)) * trend[:, None] + x1
    y1 += sig_e * np.random.randn(nobs, k_control)
    att_true = -(effect_trend * trend_new)[nobs_pre:].mean()
    myprint('ATT true', att_true)


    mod = PanelATTBasic(y0, y1, nobs_pre)
    res = mod.fit()
    myprint('\nwith OLS')
    myprint('ATT', res.att)




    res2 = mod.fit(regularization=(1, 0.9))
    myprint('\nwith elastic net penalization')
    myprint(np.nonzero(res2.res_fit.params))
    myprint('ATT', res2.att)
    #myprint(res2.res_fit.params)


    check_constraint = False
    if check_constraint:
        res_nn = OLSNonNegative(y0[:nobs_pre], y1[:nobs_pre]).fit()
        myprint(res_nn.params)
        myprint(np.nonzero(res_nn.params))

        res_simplex = OLSSimplexConstrained(y0[:nobs_pre], y1[:nobs_pre]).fit()
        myprint(res_simplex.params)
        myprint(np.nonzero(res_simplex.params))


    res3 = mod.fit(constraints='nonneg')
    myprint('\nwith nonnegativity constraints')
    myprint(res3.res_fit.model.__class__)
    myprint(np.nonzero(res3.res_fit.params))
    myprint('ATT', res3.att)

    res4 = mod.fit(constraints='simplex')
    myprint('\nwith simplex constraints')
    myprint(res4.res_fit.model.__class__)
    myprint(np.nonzero(res4.res_fit.params))
    myprint('ATT', res4.att)

    from numpy.testing import assert_allclose
    assert_allclose(res.att, att_true, rtol=0.1)
    assert_allclose(res2.att, att_true, rtol=0.1)
    assert_allclose(res3.att, att_true, rtol=0.1)
    assert_allclose(res4.att, att_true, rtol=0.1)

    try:
        import matplotlib.pyplot as plt
        fig = res.plot(loc_legend="upper left")
        #fig.show()
        plt.close(fig)
        fig = res2.plot(loc_legend="upper left")
        #fig.show()
        plt.close(fig)
        fig = res3.plot(loc_legend="upper left")
        plt.close(fig)
        fig = res4.plot(loc_legend="upper left")
        plt.close(fig)
    except ImportError:
        pass

