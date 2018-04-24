# -*- coding: utf-8 -*-

import time
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.discrete._count_tsa import predict_exparma

def test_exparma():
    nobs = 100000
    k_ar, k_ma = 1, 1
    k_exog = 4
    y = np.random.randn(nobs)
    x = np.random.randn(nobs, k_exog)
    params = 0.1 * np.ones(k_ar + k_ma + k_exog)
    for ii in range(5):
        t0 = time.time()
        fitted = predict_exparma(y, x, params, k_ar, k_ma)
        t1 = time.time()
        print('time', t1 - t0)

    f = np.exp(0.1* (y[-11:-1] + fitted[-11:-1] + x[-10:].sum(1)))
    assert_allclose(fitted[-10:], f, rtol=1e-14)
    assert_array_less(t1-t0, 0.01)
