"""
General tests for Markov switching models

Author: Chad Fulton
License: BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching import markov_regression
from numpy.testing import assert_equal, assert_allclose, assert_raises


fedfunds = [1.03, 0.99, 1.34, 1.5, 1.94, 2.36, 2.48, 2.69, 2.81, 2.93, 2.93,
            3.0, 3.23, 3.25, 1.86, 0.94, 1.32, 2.16, 2.57, 3.08, 3.58, 3.99,
            3.93, 3.7, 2.94, 2.3, 2.0, 1.73, 1.68, 2.4, 2.46, 2.61, 2.85,
            2.92, 2.97, 2.96, 3.33, 3.45, 3.46, 3.49, 3.46, 3.58, 3.97, 4.08,
            4.07, 4.17, 4.56, 4.91, 5.41, 5.56, 4.82, 3.99, 3.89, 4.17, 4.79,
            5.98, 5.94, 5.92, 6.57, 8.33, 8.98, 8.94, 8.57, 7.88, 6.7, 5.57,
            3.86, 4.56, 5.47, 4.75, 3.54, 4.3, 4.74, 5.14, 6.54, 7.82, 10.56,
            10.0, 9.32, 11.25, 12.09, 9.35, 6.3, 5.42, 6.16, 5.41, 4.83, 5.2,
            5.28, 4.87, 4.66, 5.16, 5.82, 6.51, 6.76, 7.28, 8.1, 9.58, 10.07,
            10.18, 10.95, 13.58, 15.05, 12.69, 9.84, 15.85, 16.57, 17.78,
            17.58, 13.59, 14.23, 14.51, 11.01, 9.29, 8.65, 8.8, 9.46, 9.43,
            9.69, 10.56, 11.39, 9.27, 8.48, 7.92, 7.9, 8.1, 7.83, 6.92, 6.21,
            6.27, 6.22, 6.65, 6.84, 6.92, 6.66, 7.16, 7.98, 8.47, 9.44, 9.73,
            9.08, 8.61, 8.25, 8.24, 8.16, 7.74, 6.43, 5.86, 5.64, 4.82, 4.02,
            3.77, 3.26, 3.04, 3.04, 3.0, 3.06, 2.99, 3.21, 3.94, 4.49, 5.17,
            5.81, 6.02, 5.8, 5.72, 5.36, 5.24, 5.31, 5.28, 5.28, 5.52, 5.53,
            5.51, 5.52, 5.5, 5.53, 4.86, 4.73, 4.75, 5.09, 5.31, 5.68, 6.27,
            6.52, 6.47, 5.59, 4.33, 3.5, 2.13, 1.73, 1.75, 1.74, 1.44, 1.25,
            1.25, 1.02, 1.0, 1.0, 1.01, 1.43, 1.95, 2.47, 2.94, 3.46, 3.98,
            4.46, 4.91, 5.25, 5.25, 5.26, 5.25, 5.07, 4.5, 3.18, 2.09, 1.94,
            0.51, 0.18, 0.18, 0.16, 0.12, 0.13, 0.19, 0.19, 0.19]


def test_fedfunds():
    mod = markov_regression.MarkovRegression(fedfunds, k_regimes=2)

    # Test loglike against Stata
    # See http://www.stata.com/manuals14/tsmswitch.pdf
    params = np.r_[.9820939, .0503587, 3.70877, 9.556793, 2.107562**2]
    assert_allclose(mod.loglike(params), -508.63592, atol=5)

    # Test fitting against Stata
    res = mod.fit(disp=False)
    assert_allclose(res.llf, -508.63592, atol=5)

    # Test EM fitting (smoke test)
    res_em = mod.fit_em()
    assert_allclose(res_em.llf, -508.65856, atol=5)
