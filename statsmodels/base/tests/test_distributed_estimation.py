import numpy as np
from numpy.testing import assert_equal, assert_
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, _calc_wdesign_mat, _calc_nodewise_row, _calc_nodewise_weight, _calc_approx_inv_cov, _est_regularized_debiased, _join_debiased, DistributedModel


def _exog_gen(exog, partitions):
    """partitions exog data"""

    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)

    ii = 0
    while ii < n_exog:
        jj = int(min(ii + n_part, n_exog))
        yield exog[ii:jj, :]
        ii += int(n_part)

def _endog_gen(endog, partitions):
    """partitions endog data"""

    n_endog = endog.shape[0]
    n_part = np.ceil(n_endog / partitions)

    ii = 0
    while ii < n_endog:
        jj = int(min(ii + n_part, n_endog))
        yield endog[ii:jj]
        ii += int(n_part)


def test_calc_grad():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    grad = _calc_grad(mod, beta, 0.01, 1, {})
    assert_equal(grad.shape, beta.shape)


def test_calc_wdesign_mat():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    dmat = _calc_wdesign_mat(mod, beta, {})
    assert_equal(dmat.shape, X.shape)

    mod = GLM(y, X, family=Binomial())
    dmat = _calc_wdesign_mat(mod, beta, {})
    assert_equal(dmat.shape, X.shape)


def test_calc_nodewise_row():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    ghat = _calc_nodewise_row(X, 0, 0.01)
    assert_equal(ghat.shape, (2,))


def test_calc_nodewise_weight():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    ghat = np.random.normal(size=2)
    that = _calc_nodewise_weight(X, ghat, 0, 0.01)
    assert_(isinstance(that, float))


def test_calc_approx_inv_cov():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    ghat_l = []
    that_l = []
    for i in range(3):
        ghat = _calc_nodewise_row(X, i, 0.01)
        that = _calc_nodewise_weight(X, ghat, i, 0.01)
        ghat_l.append(ghat)
        that_l.append(that)
    theta_hat = _calc_approx_inv_cov(ghat_l, that_l, 3)
    assert_equal(theta_hat.shape, np.eye(3).shape)


def test_est_regularized_debiased():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res = _est_regularized_debiased(mod, 0, 2, fit_kwds={"alpha": 0.5})
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
    res = _est_regularized_debiased(mod, 0, 2, fit_kwds={"alpha": 0.5})
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

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res_l = []
    for i in range(2):
        res = _est_regularized_debiased(mod, i, 2, fit_kwds={"alpha": 0.5})
        res_l.append(res)
    joined = _join_debiased(res_l, 2)
    assert_equal(joined.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res_l = []
    for i in range(2):
        res = _est_regularized_debiased(mod, i, 2, fit_kwds={"alpha": 0.5})
        res_l.append(res)
    joined = _join_debiased(res_l, 2)
    assert_equal(joined.shape, beta.shape)


def test_fit_sequential():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)


def test_fit_joblib():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=OLS, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    fit = mod.fit_distributed(parallel_method="sequential")
    assert_equal(fit.shape, beta.shape)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    assert_equal(fit.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.5})
    fit = mod.fit_distributed(parallel_method="joblib")
    assert_equal(fit.shape, beta.shape)
