import numpy as np
from numpy.testing import assert_equal, assert_
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _gen_grad, _gen_wdesign_mat, _gen_gamma_hat, _gen_tau_hat, _gen_theta_hat, _est_regularized_distributed, _join_debiased, distributed_estimation


def exogGen(exog, partitions):
    """partitions exog data"""

    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)

    ii = 0
    while ii < n_exog:
        jj = int(min(ii + n_part, n_exog))
        yield exog[ii:jj, :]
        ii += int(n_part)

def endogGen(endog, partitions):
    """partitions endog data"""

    n_endog = endog.shape[0]
    n_part = np.ceil(n_endog / partitions)

    ii = 0
    while ii < n_endog:
        jj = int(min(ii + n_part, n_endog))
        yield endog[ii:jj]
        ii += int(n_part)


def test_gen_grad():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    grad = _gen_grad(mod, beta, 50, 0.01, 1, {})
    assert_equal(grad.shape, beta.shape)


def test_gen_wdesign_mat():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    dmat = _gen_wdesign_mat(mod, beta, {})
    assert_equal(dmat.shape, X.shape)

    mod = GLM(y, X, family=Binomial())
    dmat = _gen_wdesign_mat(mod, beta, {})
    assert_equal(dmat.shape, X.shape)


def test_gen_gamma_hat():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    ghat = _gen_gamma_hat(X, 0, 3, 50, 0.01)
    assert_equal(ghat.shape, (2,))


def test_gen_tau_hat():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    ghat = np.random.normal(size=2)
    that = _gen_tau_hat(X, ghat, 0, 3, 50, 0.01)
    assert_(isinstance(that, float))


def test_gen_theta_hat():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    ghat_l = []
    that_l = []
    for i in range(3):
        ghat = _gen_gamma_hat(X, i, 3, 50, 0.01)
        that = _gen_tau_hat(X, ghat, i, 3, 50, 0.01)
        ghat_l.append(ghat)
        that_l.append(that)
    theta_hat = _gen_theta_hat(ghat_l, that_l, 3)
    assert_equal(theta_hat.shape, np.eye(3).shape)


def test_est_regularized_distributed():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res = _est_regularized_distributed(mod, 0, 2, fit_kwds={"alpha": 0.5})
    bhat = res[0]
    grad = res[1]
    ghat_l = res[2]
    that_l = res[3]

    assert_(isinstance(res, tuple))
    assert_equal(bhat.shape, beta.shape)
    assert_equal(grad.shape, beta.shape)
    assert_(isinstance(ghat_l, list))
    assert_(isinstance(that_l, list))
    assert_equal(len(ghat_l), len(that_l))
    assert_equal(ghat_l[0].shape, (2,))
    assert_(isinstance(that_l[0], float))

    mod = GLM(y, X, family=Binomial())
    res = _est_regularized_distributed(mod, 0, 2, fit_kwds={"alpha": 0.5})
    bhat = res[0]
    grad = res[1]
    ghat_l = res[2]
    that_l = res[3]

    assert_(isinstance(res, tuple))
    assert_equal(bhat.shape, beta.shape)
    assert_equal(grad.shape, beta.shape)
    assert_(isinstance(ghat_l, list))
    assert_(isinstance(that_l, list))
    assert_equal(len(ghat_l), len(that_l))
    assert_equal(ghat_l[0].shape, (2,))
    assert_(isinstance(that_l[0], float))


def test_join_debiased():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res_l = []
    for i in range(2):
        res = _est_regularized_distributed(mod, i, 2, fit_kwds={"alpha": 0.5})
        res_l.append(res)
    joined = _join_debiased(res_l, 2)
    assert_equal(joined.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res_l = []
    for i in range(2):
        res = _est_regularized_distributed(mod, i, 2, fit_kwds={"alpha": 0.5})
        res_l.append(res)
    joined = _join_debiased(res_l, 2)
    assert_equal(joined.shape, beta.shape)


def test_distributed_estimation():

    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)

    fit = distributed_estimation(zip(endogGen(y, 1), exogGen(X, 1)), 1, model_class=OLS, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)
    fit = distributed_estimation(zip(endogGen(y, 2), exogGen(X, 2)), 2, model_class=OLS, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)
    fit = distributed_estimation(zip(endogGen(y, 3), exogGen(X, 3)), 3, model_class=OLS, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)
    fit = distributed_estimation(zip(endogGen(y, 50), exogGen(X, 50)), 50, model_class=OLS, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)

    fit = distributed_estimation(zip(endogGen(y, 1), exogGen(X, 1)), 1, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)
    fit = distributed_estimation(zip(endogGen(y, 2), exogGen(X, 2)), 2, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)
    fit = distributed_estimation(zip(endogGen(y, 3), exogGen(X, 3)), 3, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)
    fit = distributed_estimation(zip(endogGen(y, 50), exogGen(X, 50)), 50, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    assert_equal(fit.shape, beta.shape)
