# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:51:24 2017

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose
from statsmodels.treatment.panel_effect import (
            PanelATTBasic, OLSNonNegative, OLSSimplexConstrained)


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
    trend_new = np.zeros(nobs)
    trend_new[nobs_pre:] = np.arange(nobs - nobs_pre) + shift
    np.random.seed(987456)

    sig_f = 0.5
    factor = sig_f * np.random.randn(nobs, k_factors)
    load = 1 + np.random.randn(k_factors , k_control + 1)
    noise_x = 0.1 * np.random.randn(nobs, k_control)

    x0 = factor.dot(load[:, 0]) + 0.1 * np.random.randn(nobs)
    x1 = factor.dot(load[:, 1:]) + 0.2 * noise_x

    sig_e = 0.5
    y0 = trend + x0 - effect_trend * trend_new + sig_e * np.random.randn(nobs)
    y1 = (1 + 0.25 * np.random.rand(k_control)) * trend[:, None] + x1
    y1 += sig_e * np.random.randn(nobs, k_control)
    att_true = -(effect_trend * trend_new)[nobs_pre:].mean()

    mod = PanelATTBasic(y0, y1, nobs_pre)
    res = mod.fit()

    res2 = mod.fit(regularization=(1, 0.9))

    check_constraint = True
    if check_constraint:
        # rough check, the tolerances are large because of the prediction error
        res_nn = OLSNonNegative(y0[:nobs_pre], y1[:nobs_pre]).fit()
        fitted = res_nn.predict()
        assert_allclose(fitted, y0[:nobs_pre], atol=2, rtol=0.3)

        res_simplex = OLSSimplexConstrained(y0[:nobs_pre], y1[:nobs_pre]).fit()
        fitted = res_simplex.predict()
        assert_allclose(fitted, y0[:nobs_pre], atol=2.5, rtol=0.3)


    res3 = mod.fit(constraints='nonneg')

    res4 = mod.fit(constraints='simplex')

    assert_allclose(res.att, att_true, rtol=0.1)
    assert_allclose(res2.att, att_true, rtol=0.1)
    assert_allclose(res3.att, att_true, rtol=0.1)
    assert_allclose(res4.att, att_true, rtol=0.1)

    diff_true = -effect_trend * trend_new
    atol = 0.35
    sf = 1 / y0.mean() # scaling for errors relative to endog.mean()
    assert_allclose(res.prediction_error * sf, diff_true * sf, atol=atol, rtol=0.1)
    assert_allclose(res2.prediction_error * sf, diff_true * sf, atol=atol, rtol=0.1)
    assert_allclose(res3.prediction_error * sf, diff_true * sf, atol=atol, rtol=0.1)
    assert_allclose(res4.prediction_error * sf, diff_true * sf, atol=atol, rtol=0.1)

    try:
        import matplotlib.pyplot as plt
        fig = res.plot(loc_legend="upper left")
        plt.close(fig)
        fig = res2.plot(loc_legend="upper left")
        plt.close(fig)
        fig = res3.plot(loc_legend="upper left")
        plt.close(fig)
        fig = res4.plot(loc_legend="upper left")
        plt.close(fig)
    except ImportError:
        pass

