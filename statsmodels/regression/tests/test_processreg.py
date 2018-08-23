# -*- coding: utf-8 -*-

from statsmodels.regression.process_reg import (ProcessRegression,
                                                GaussianCovariance)
import numpy as np
import pandas as pd
import collections
import statsmodels.tools.numdiff as nd
from numpy.testing import assert_allclose, assert_equal


def setup1(n):

    q = 4

    mn_par = np.r_[1, 0, -1, 0]
    sd_par = np.r_[1, 0]
    sm_par = np.r_[0.5, -0.1]

    groups = np.kron(np.arange(n // 5), np.ones(5))
    time = np.kron(np.ones(n // 5), np.arange(5))

    x_mean = np.random.normal(size=(n, q))
    x_sd = 0.2 * np.random.normal(size=(n, 2))
    x_sm = np.ones((n, 2))
    x_sm[:, 1] = time

    mn = np.dot(x_mean, mn_par)
    sd = np.exp(np.dot(x_sd, sd_par))
    sm = np.exp(np.dot(x_sm, sm_par))

    y = mn.copy()

    gc = GaussianCovariance()

    ix = collections.defaultdict(lambda: [])
    for i, g in enumerate(groups):
        ix[g].append(i)

    for g, ii in ix.items():
        c = gc.get_cov(time[ii], sd[ii], sm[ii])
        r = np.linalg.cholesky(c)
        y[ii] += np.dot(r, np.random.normal(size=len(ii)))

    return y, x_mean, x_sd, x_sm, time, groups


def test_arrays():

    np.random.seed(8996)

    y, x_mean, x_sc, x_sm, time, groups = setup1(1000)

    preg = ProcessRegression(y, x_mean, x_sc, x_sm, time, groups)

    f = preg.fit()
    f.summary()  # Smoke test

    # Compare the parameter estimates to population values.
    epar = np.r_[1, 0, -1, 0, 1, 0, 0.5, -0.1]
    assert_allclose(f.params, epar, atol=0.1, rtol=1)

    # Test the fitted covariance matrix
    cv = f.covariance(time[0:5], x_sc[0:5, :], x_sm[0:5, :])
    assert_allclose(cv, cv.T)  # Check symmetry
    a, _ = np.linalg.eig(cv)
    assert_equal(a > 0, True)  # Check PSD

    # Test predict
    yhat = f.predict()
    assert_equal(np.corrcoef(yhat, y)[0, 1] > 0.75, True)
    yhatm = f.predict(exog=x_mean)
    assert_equal(yhat, yhatm)
    yhat0 = preg.predict(params=f.params, exog=x_mean)
    assert_equal(yhat, yhat0)

    # Smoke test t-test
    f.t_test(np.eye(len(f.params)))

def test_formulas():

    np.random.seed(8482)

    y, x_mean, x_sd, x_sm, time, groups = setup1(1000)

    df = pd.DataFrame({
        "y": y,
        "x1": x_mean[:, 0],
        "x2": x_mean[:, 1],
        "x3": x_mean[:, 2],
        "x4": x_mean[:, 3],
        "xsd1": x_sd[:, 0],
        "xsd2": x_sd[:, 1],
        "xsm1": x_sm[:, 0],
        "xsm2": x_sm[:, 1],
        "time": time,
        "groups": groups
    })

    mean_formula = "y ~ x1 + x2 + x3 + x4"
    scale_formula = "0 + xsd1 + xsd2"
    smooth_formula = "0 + xsm1 + xsm2"
    preg = ProcessRegression.from_formula(
        mean_formula,
        data=df,
        scale_formula=scale_formula,
        smooth_formula=smooth_formula,
        time="time",
        groups="groups")
    f = preg.fit()
    f.summary()  # Smoke test

    # Compare the parameter estimates to population values.
    epar = np.r_[0, 1, 0, -1, 0, 1, 0, 0.5, -0.1]
    assert_allclose(f.params, epar, atol=0.1, rtol=1)

    # Test the fitted covariance matrix
    cv = f.covariance(df.time.iloc[0:5], df.iloc[0:5, :],
                      df.iloc[0:5, :])
    assert_allclose(cv, cv.T)
    a, _ = np.linalg.eig(cv)
    assert_equal(a > 0, True)

    # Test predict
    yhat = f.predict()
    assert_equal(np.corrcoef(yhat, y)[0, 1] > 0.75, True)
    yhatm = f.predict(exog=df)
    assert_equal(yhat, yhatm)
    yhat0 = preg.predict(params=f.params, exog=df)
    assert_equal(yhat, yhat0)

    # Smoke test t-test
    f.t_test(np.eye(len(f.params)))


# Test the score functions using numerical derivatives.
def test_score_numdiff():

    y, x_mean, x_sd, x_sm, time, groups = setup1(1000)

    preg = ProcessRegression(y, x_mean, x_sd, x_sm, time, groups)

    def loglike(x):
        return preg.loglike(x)
    q = x_mean.shape[1] + x_sd.shape[1] + x_sm.shape[1]

    np.random.seed(342)

    for _ in range(5):
        par0 = preg._get_start()
        par = par0 + 0.1 * np.random.normal(size=q)
        score = preg.score(par)
        score_nd = nd.approx_fprime(par, loglike, epsilon=1e-7)
        assert_allclose(score, score_nd, atol=1e-3, rtol=1e-4)
