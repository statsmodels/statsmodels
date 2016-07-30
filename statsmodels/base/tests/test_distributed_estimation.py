import numpy as np
from numpy.testing import assert_equal, assert_
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, _calc_wdesign_mat, _est_regularized_debiased, _join_debiased, _est_regularized_naive, _est_unregularized_naive, _join_naive, DistributedModel


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


def test_est_regularized_naive():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res = _est_regularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})
    bhat = res[0]

    assert_equal(res.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res = _est_regularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})

    assert_equal(res.shape, beta.shape)


def test_est_unregularized_naive():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res = _est_unregularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})

    assert_equal(res.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res = _est_unregularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})

    assert_equal(res.shape, beta.shape)


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
    joined = _join_debiased(res_l)
    assert_equal(joined.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res_l = []
    for i in range(2):
        res = _est_regularized_debiased(mod, i, 2, fit_kwds={"alpha": 0.5})
        res_l.append(res)
    joined = _join_debiased(res_l)
    assert_equal(joined.shape, beta.shape)


def test_join_naive():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res_l = []
    for i in range(2):
        res = _est_regularized_naive(mod, i, 2, fit_kwds={"alpha": 0.5})
        res_l.append(res)
    joined = _join_naive(res_l)
    assert_equal(joined.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res_l = []
    for i in range(2):
        res = _est_regularized_naive(mod, i, 2, fit_kwds={"alpha": 0.5})
        res_l.append(res)
    joined = _join_naive(res_l)
    assert_equal(joined.shape, beta.shape)



def test_fit_sequential():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=OLS)
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=OLS)
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=OLS)
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=OLS)
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="sequential", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)


def test_fit_joblib():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=OLS)
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=OLS)
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=OLS)
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=OLS)
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)

    mod = DistributedModel(zip(_endog_gen(y, 1), _exog_gen(X, 1)), 1, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 2), _exog_gen(X, 2)), 2, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 3), _exog_gen(X, 3)), 3, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(zip(_endog_gen(y, 50), _exog_gen(X, 50)), 50, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)


def test_debiased_v_average():

    np.random.seed(435265)
    N = 200
    p = 10
    m = 4

    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)

    db_mod = DistributedModel(zip(_endog_gen(y, m), _exog_gen(X, m)), m)
    fitOLSdb = db_mod.fit(fit_kwds={"alpha": 0.2})
    olsdb = np.linalg.norm(fitOLSdb.params - beta)
    n_mod = DistributedModel(zip(_endog_gen(y, m), _exog_gen(X, m)), m, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitOLSn = n_mod.fit(fit_kwds={"alpha": 0.2})
    olsn = np.linalg.norm(fitOLSn.params - beta)

    assert_(olsdb < olsn)

    prob = 1 / (1 + np.exp(-X.dot(beta) + np.random.normal(size=N)))
    y = 1. * (prob > 0.5)

    db_mod = DistributedModel(zip(_endog_gen(y, m), _exog_gen(X, m)), m, model_class=GLM, init_kwds={"family": Binomial()})
    fitGLMdb = db_mod.fit(fit_kwds={"alpha": 0.2})
    glmdb = np.linalg.norm(fitGLMdb.params - beta)
    n_mod = DistributedModel(zip(_endog_gen(y, m), _exog_gen(X, m)), m, model_class=GLM, init_kwds={"family": Binomial()}, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitGLMn = n_mod.fit(fit_kwds={"alpha": 0.2})
    glmn = np.linalg.norm(fitGLMn.params - beta)

    assert_(glmdb < glmn)
