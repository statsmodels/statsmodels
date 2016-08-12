import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
    _calc_wdesign_mat, _est_regularized_debiased, _join_debiased, \
    _est_regularized_naive, _est_unregularized_naive, _join_naive, \
    DistributedModel


def _data_gen(endog, exog, partitions):
    """partitions data"""

    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)

    ii = 0
    while ii < n_exog:
        jj = int(min(ii + n_part, n_exog))
        yield endog[ii:jj], exog[ii:jj, :]
        ii += int(n_part)


def test_calc_grad():

    # seperately tests that _calc_grad returns
    # sensible results

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    grad = _calc_grad(mod, beta, 0.01, 1, {})
    assert_equal(grad.shape, beta.shape)


def test_calc_wdesign_mat():

    # seperately tests that _calc_wdesign_mat
    # returns sensible results

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

    # tests that the shape of all the intermediate steps
    # remains correct for regularized debiased estimation,
    # does this for OLS and GLM

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

    # tests that the shape of all the intermediate steps
    # remains correct for regularized naive estimation,
    # does this for OLS and GLM

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    res = _est_regularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})

    assert_equal(res.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res = _est_regularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})

    assert_equal(res.shape, beta.shape)


def test_est_unregularized_naive():

    # tests that the shape of all the intermediate steps
    # remains correct for unregularized naive estimation,
    # does this for OLS and GLM

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

    # tests that the shape of all the intermediate steps
    # remains correct for debiased join, does this for OLS and GLM

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

    # tests that the shape of all the intermediate steps
    # remains correct for naive join, does this for OLS and GLM

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

    # tests that the shape of all the intermediate steps
    # remains correct for sequential fit, does this for OLS and GLM
    # and a variety of model sizes

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)

    mod = DistributedModel(1, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 1), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(2, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 2), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(3, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 3), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(50, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 50), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)

    mod = DistributedModel(1, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 1), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(2, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 2), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(3, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 3), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(50, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 50), parallel_method="sequential",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)


def test_fit_joblib():

    # tests that the shape of all the intermediate steps
    # remains correct for joblib fit, does this for OLS and GLM
    # and a variety of model sizes

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)

    mod = DistributedModel(1, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 1), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(2, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 2), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(3, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 3), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(50, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 50), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)

    mod = DistributedModel(1, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 1), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(2, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 2), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(3, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 3), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)
    mod = DistributedModel(50, model_class=GLM,
                           init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 50), parallel_method="joblib",
                  fit_kwds={"alpha": 0.5})
    assert_equal(fit.params.shape, beta.shape)


def test_single_partition():

    # tests that the results make sense if we have a single partition

    np.random.seed(435265)
    N = 200
    p = 10
    m = 1

    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)

    # test regularized OLS v. naive
    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0})

    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive,
                              join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0})

    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit(alpha=0)

    assert_allclose(fitOLSdb.params, fitOLS.params)
    assert_allclose(fitOLSnv.params, fitOLS.params)

    # test regularized
    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive,
                              join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.1})

    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)

    assert_allclose(fitOLSnv.params, fitOLS.params)


def test_larger_p():

    # tests when p > N / m for the debiased and naive case

    np.random.seed(435265)
    N = 40
    p = 40
    m = 5

    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)

    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.1})
    assert_equal(np.sum(np.isnan(fitOLSdb.params)), 0)

    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive,
                              join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.1})
    assert_equal(np.sum(np.isnan(fitOLSnv.params)), 0)


def test_non_zero_params():

    # tests that the thresholding does not cause any issues

    np.random.seed(435265)
    N = 200
    p = 10
    m = 5

    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)

    db_mod = DistributedModel(m, join_kwds={"threshold": 0.13})
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.1})
    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)

    nz_params_db = 1 * (fitOLSdb.params != 0)
    nz_params_ols = 1 * (fitOLS.params != 0)

    assert_allclose(nz_params_db, nz_params_ols)


def test_repeat_partition():

    # tests that if we use identical partitions the average is the same
    # as the estimate for the full data

    np.random.seed(435265)
    N = 200
    p = 10
    m = 1

    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)

    def _rep_data_gen(endog, exog, partitions):
        """partitions data"""

        n_exog = exog.shape[0]
        n_part = np.ceil(n_exog / partitions)

        ii = 0
        while ii < n_exog:
            yield endog, exog
            ii += int(n_part)

    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive,
                              join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_rep_data_gen(y, X, m), fit_kwds={"alpha": 0.1})

    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)

    assert_allclose(fitOLSnv.params, fitOLS.params)


def test_debiased_v_average():

    # tests that the debiased method performs better than the standard
    # average.  Does this for both OLS and GLM.

    np.random.seed(435265)
    N = 200
    p = 10
    m = 4

    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)

    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    olsdb = np.linalg.norm(fitOLSdb.params - beta)
    n_mod = DistributedModel(m, estimation_method=_est_regularized_naive,
                             join_method=_join_naive)
    fitOLSn = n_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    olsn = np.linalg.norm(fitOLSn.params - beta)

    assert_(olsdb < olsn)

    prob = 1 / (1 + np.exp(-X.dot(beta) + np.random.normal(size=N)))
    y = 1. * (prob > 0.5)

    db_mod = DistributedModel(m, model_class=GLM,
                              init_kwds={"family": Binomial()})
    fitGLMdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    glmdb = np.linalg.norm(fitGLMdb.params - beta)
    n_mod = DistributedModel(m, model_class=GLM,
                             init_kwds={"family": Binomial()},
                             estimation_method=_est_regularized_naive,
                             join_method=_join_naive)
    fitGLMn = n_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    glmn = np.linalg.norm(fitGLMn.params - beta)

    assert_(glmdb < glmn)
