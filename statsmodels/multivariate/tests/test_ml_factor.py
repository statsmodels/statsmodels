import numpy as np
from statsmodels.multivariate.factor import Factor, _rotate
from numpy.testing import assert_allclose, assert_equal


def test_rotate():

    np.random.seed(23324)

    gamma = np.random.normal(size=(10, 4))
    sigma = np.linspace(1, 3, 10)
    g = _rotate(gamma, sigma)
    f = np.dot(g.T, g/sigma[:, None])
    f -= np.diag(np.diag(f))
    assert_allclose(f, np.zeros_like(f), rtol=1e-8, atol=1e-8)


def test_exact():
    # Test if we can recover exact factor-structured matrices with
    # default starting values.

    np.random.seed(23324)

    # Works for larger k_var but slow for routine testing.
    for k_var in 5, 10, 25:
        for n_factor in 1, 2, 3:
            gamma = np.random.normal(size=(k_var, n_factor))
            sigma2 = np.linspace(1, 2, k_var)
            gamma = _rotate(gamma, sigma2)
            c = np.dot(gamma, gamma.T)
            c.flat[::c.shape[0]+1] += sigma2
            s = np.sqrt(np.diag(c))
            c /= np.outer(s, s)
            fa = Factor(corr=c, n_factor=n_factor, method='ml')
            rslt = fa.fit()
            assert_allclose(rslt.fitted_cov, c, rtol=1e-4, atol=1e-4)
            # print(rslt.summary()) # DEBUG


def test_1factor():
    """
    # R code:
    r = 0.4
    p = 4
    ii = seq(0, p-1)
    ii = outer(ii, ii, "-")
    ii = abs(ii)
    cm = r^ii
    factanal(covmat=cm, factors=1)
    """

    r = 0.4
    p = 4
    ii = np.arange(p)
    cm = r ** np.abs(np.subtract.outer(ii, ii))

    fa = Factor(corr=cm, n_factor=1, method='ml')
    rslt = fa.fit()

    if rslt.loadings[0, 0] < 0:
        rslt.loadings[:, 0] *= -1

    load = np.r_[0.401, 0.646, 0.646, 0.401]
    uniq = np.r_[0.839, 0.582, 0.582, 0.839]
    assert_allclose(load, rslt.loadings[:, 0], rtol=1e-3, atol=1e-3)
    assert_allclose(uniq, rslt.uniqueness, rtol=1e-3, atol=1e-3)

    assert_equal(rslt.df, 2)


def test_2factor():
    """
    # R code:
    r = 0.4
    p = 6
    ii = seq(0, p-1)
    ii = outer(ii, ii, "-")
    ii = abs(ii)
    cm = r^ii
    factanal(covmat=cm, factors=2)
    """

    r = 0.4
    p = 6
    ii = np.arange(p)
    cm = r ** np.abs(np.subtract.outer(ii, ii))

    fa = Factor(corr=cm, n_factor=2, nobs=100, method='ml')
    rslt = fa.fit()

    for j in 0, 1:
        if rslt.loadings[0, j] < 0:
            rslt.loadings[:, j] *= -1

    uniq = np.r_[0.782, 0.367, 0.696, 0.696, 0.367, 0.782]
    assert_allclose(uniq, rslt.uniqueness, rtol=1e-3, atol=1e-3)

    load = np.r_[0.323, 0.586, 0.519, 0.519, 0.586, 0.323]
    assert_allclose(load, rslt.loadings[:, 0], rtol=1e-3, atol=1e-3)
    load = np.r_[0.337, 0.538, 0.187, -0.187, -0.538, -0.337]
    assert_allclose(load, rslt.loadings[:, 1], rtol=1e-3, atol=1e-3)

    assert_equal(rslt.df, 4)

    # Smoke test for standard errors
    rslt.uniq_stderr
    rslt.load_stderr
