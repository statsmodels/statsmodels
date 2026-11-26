import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pytest

from statsmodels.base.distributed_estimation import (
    DistributedModel,
    DistributedResults,
    _calc_grad,
    _calc_wdesign_mat,
    _est_regularized_debiased,
    _est_regularized_naive,
    _est_unregularized_naive,
    _join_debiased,
    _join_naive,
)
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM


def _data_gen(endog, exog, partitions):
    """partitions data"""

    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)

    n_part = np.floor(n_exog / partitions)
    rem = n_exog - n_part * partitions

    stp = 0

    while stp < (partitions - 1):
        ii = int(n_part * stp)
        jj = int(n_part * (stp + 1))
        yield endog[ii:jj], exog[ii:jj, :]
        stp += 1

    ii = int(n_part * stp)
    jj = int(n_part * (stp + 1) + rem)
    yield endog[ii:jj], exog[ii:jj, :]


def test_calc_grad():

    # separately tests that _calc_grad returns
    # sensible results
    #
    # regression test

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)
    beta = rs.normal(size=3)
    mod = OLS(y, X)
    grad = _calc_grad(mod, beta, 0.01, 1, {})
    assert_allclose(grad, np.array([19.75816, -6.62307, 7.324644]), atol=1e-6, rtol=0)


def test_calc_wdesign_mat():

    # separately tests that _calc_wdesign_mat
    # returns sensible results
    #
    # regression test

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(3, 3))
    y = rs.randint(0, 2, size=3)
    beta = rs.normal(size=3)
    mod = OLS(y, X)
    dmat = _calc_wdesign_mat(mod, beta, {})
    assert_allclose(
        dmat,
        np.array(
            [
                [1.306314, -0.024897, 1.326498],
                [-0.539219, -0.483028, -0.703503],
                [-3.327987, 0.524541, -0.139761],
            ]
        ),
        atol=1e-6,
        rtol=0,
    )

    mod = GLM(y, X, family=Binomial())
    dmat = _calc_wdesign_mat(mod, beta, {})
    assert_allclose(
        dmat,
        np.array(
            [
                [0.408616, -0.007788, 0.41493],
                [-0.263292, -0.235854, -0.343509],
                [-0.11241, 0.017718, -0.004721],
            ]
        ),
        atol=1e-6,
        rtol=0,
    )


def test_est_regularized_debiased():

    # tests that the shape of all the intermediate steps
    # remains correct for regularized debiased estimation,
    # does this for OLS and GLM

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)
    beta = rs.normal(size=3)
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

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)
    beta = rs.normal(size=3)
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

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)
    beta = rs.normal(size=3)
    mod = OLS(y, X)
    res = _est_unregularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})

    assert_equal(res.shape, beta.shape)

    mod = GLM(y, X, family=Binomial())
    res = _est_unregularized_naive(mod, 0, 2, fit_kwds={"alpha": 0.5})

    assert_equal(res.shape, beta.shape)


def test_join_debiased():

    # tests that the results of all the intermediate steps
    # remains correct for debiased join, does this for OLS and GLM
    #
    # regression test

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)
    mod = OLS(y, X)
    res_l = []
    for i in range(2):
        res = _est_regularized_debiased(mod, i, 2, fit_kwds={"alpha": 0.1})
        res_l.append(res)
    joined = _join_debiased(res_l)
    assert_allclose(
        joined, np.array([-0.167548, -0.016567, -0.34414]), atol=1e-6, rtol=0
    )

    mod = GLM(y, X, family=Binomial())
    res_l = []
    for i in range(2):
        res = _est_regularized_debiased(mod, i, 2, fit_kwds={"alpha": 0.1})
        res_l.append(res)
    joined = _join_debiased(res_l)
    assert_allclose(
        joined, np.array([-0.164515, -0.412854, -0.223955]), atol=1e-6, rtol=0
    )


def test_join_naive():

    # tests that the results of all the intermediate steps
    # remains correct for naive join, does this for OLS and GLM
    #
    # regression test

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)
    mod = OLS(y, X)
    res_l = []
    for i in range(2):
        res = _est_regularized_naive(mod, i, 2, fit_kwds={"alpha": 0.1})
        res_l.append(res)
    joined = _join_naive(res_l)
    assert_allclose(joined, np.array([-0.020757, 0.0, 0.0]), atol=1e-6, rtol=0)

    mod = GLM(y, X, family=Binomial())
    res_l = []
    for i in range(2):
        res = _est_regularized_naive(mod, i, 2, fit_kwds={"alpha": 0.1})
        res_l.append(res)
    joined = _join_naive(res_l)
    assert_allclose(joined, np.array([0.0, 0.0, 0.0]), atol=1e-6, rtol=0)


def test_fit_sequential():

    # tests that the results of all the intermediate steps
    # remains correct for sequential fit, does this for OLS and GLM
    # and a variety of model sizes
    #
    # regression test

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)

    mod = DistributedModel(1, model_class=OLS)
    fit = mod.fit(
        _data_gen(y, X, 1), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    assert_allclose(
        fit.params, np.array([-0.191606, -0.012565, -0.351398]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(2, model_class=OLS)
    fit = mod.fit(
        _data_gen(y, X, 2), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    assert_allclose(
        fit.params, np.array([-0.157416, -0.029643, -0.471653]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(3, model_class=OLS)
    fit = mod.fit(
        _data_gen(y, X, 3), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    assert_allclose(
        fit.params, np.array([-0.124891, -0.050934, -0.403354]), atol=1e-6, rtol=0
    )

    mod = DistributedModel(1, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(
        _data_gen(y, X, 1), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    assert_allclose(
        fit.params, np.array([-0.164515, -0.412854, -0.223955]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(2, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(
        _data_gen(y, X, 2), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    assert_allclose(
        fit.params, np.array([-0.142513, -0.360324, -0.295485]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(3, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(
        _data_gen(y, X, 3), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    assert_allclose(
        fit.params, np.array([-0.110487, -0.306431, -0.243921]), atol=1e-6, rtol=0
    )


def test_fit_invalid_parallel_method_raises():
    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)

    mod = DistributedModel(1, model_class=OLS)
    with pytest.raises(ValueError, match="parallel_method"):
        mod.fit(_data_gen(y, X, 1), parallel_method="not-a-method")


@pytest.mark.joblib
def test_fit_joblib_single_partition_matches_sequential():
    # test_fit_joblib below covers several partition counts but is marked
    # slow; with a single partition, joblib's Parallel(n_jobs=1) runs
    # in-process (no subprocess spawned), so this exercises the same
    # fit_joblib code path quickly. Cross check against the (separately
    # tested) sequential path, since both partition schemes combine the
    # same per-partition estimates with the same join_method and so must
    # agree exactly for a single partition.
    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)

    mod = DistributedModel(1, model_class=OLS)
    fit_seq = mod.fit(
        _data_gen(y, X, 1), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    fit_par = mod.fit(
        _data_gen(y, X, 1), parallel_method="joblib", fit_kwds={"alpha": 0.5}
    )

    assert_allclose(fit_par.params, fit_seq.params)


def test_distributed_results_predict():
    # DistributedResults is not used as any default results_class, so it
    # must be requested explicitly; predict() should just delegate to the
    # (fake) model's own predict, which for OLS is exog @ params.
    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)

    mod = DistributedModel(2, model_class=OLS, results_class=DistributedResults)
    fit = mod.fit(
        _data_gen(y, X, 2), parallel_method="sequential", fit_kwds={"alpha": 0.5}
    )
    assert isinstance(fit, DistributedResults)

    Xnew = rs.normal(size=(5, 3))
    pred = fit.predict(Xnew)

    assert_allclose(pred, Xnew.dot(fit.params))


@pytest.mark.joblib
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Uses joblib")
def test_fit_joblib():

    # tests that the results of all the intermediate steps
    # remains correct for joblib fit, does this for OLS and GLM
    # and a variety of model sizes
    #
    # regression test

    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)

    mod = DistributedModel(1, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 1), parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_allclose(
        fit.params, np.array([-0.191606, -0.012565, -0.351398]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(2, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 2), parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_allclose(
        fit.params, np.array([-0.157416, -0.029643, -0.471653]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(3, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 3), parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_allclose(
        fit.params, np.array([-0.124891, -0.050934, -0.403354]), atol=1e-6, rtol=0
    )

    mod = DistributedModel(1, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 1), parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_allclose(
        fit.params, np.array([-0.164515, -0.412854, -0.223955]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(2, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 2), parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_allclose(
        fit.params, np.array([-0.142513, -0.360324, -0.295485]), atol=1e-6, rtol=0
    )
    mod = DistributedModel(3, model_class=GLM, init_kwds={"family": Binomial()})
    fit = mod.fit(_data_gen(y, X, 3), parallel_method="joblib", fit_kwds={"alpha": 0.5})
    assert_allclose(
        fit.params, np.array([-0.110487, -0.306431, -0.243921]), atol=1e-6, rtol=0
    )


def test_single_partition():

    # tests that the results make sense if we have a single partition

    rs = np.random.RandomState(435265)
    N = 200
    p = 10
    m = 1

    beta = rs.normal(size=p)
    beta = beta * rs.randint(0, 2, p)
    X = rs.normal(size=(N, p))
    y = X.dot(beta) + rs.normal(size=N)

    # test regularized OLS v. naive
    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0})

    nv_mod = DistributedModel(
        m, estimation_method=_est_regularized_naive, join_method=_join_naive
    )
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0})

    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit(alpha=0)

    assert_allclose(fitOLSdb.params, fitOLS.params)
    assert_allclose(fitOLSnv.params, fitOLS.params)

    # test regularized
    nv_mod = DistributedModel(
        m, estimation_method=_est_regularized_naive, join_method=_join_naive
    )
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.1})

    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)

    assert_allclose(fitOLSnv.params, fitOLS.params)


def test_larger_p():

    # tests when p > N / m for the debiased and naive case

    rs = np.random.RandomState(435265)
    N = 40
    p = 40
    m = 5

    beta = rs.normal(size=p)
    beta = beta * rs.randint(0, 2, p)
    X = rs.normal(size=(N, p))
    y = X.dot(beta) + rs.normal(size=N)

    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.1})
    assert_equal(np.sum(np.isnan(fitOLSdb.params)), 0)

    nv_mod = DistributedModel(
        m, estimation_method=_est_regularized_naive, join_method=_join_naive
    )
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.1})
    assert_equal(np.sum(np.isnan(fitOLSnv.params)), 0)


def test_non_zero_params():

    # tests that the thresholding does not cause any issues

    rs = np.random.RandomState(435265)
    N = 200
    p = 10
    m = 5

    beta = rs.normal(size=p)
    beta = beta * rs.randint(0, 2, p)
    X = rs.normal(size=(N, p))
    y = X.dot(beta) + rs.normal(size=N)

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

    rs = np.random.RandomState(435265)
    N = 200
    p = 10
    m = 1

    beta = rs.normal(size=p)
    beta = beta * rs.randint(0, 2, p)
    X = rs.normal(size=(N, p))
    y = X.dot(beta) + rs.normal(size=N)

    def _rep_data_gen(endog, exog, partitions):
        """partitions data"""

        n_exog = exog.shape[0]
        n_part = np.ceil(n_exog / partitions)

        ii = 0
        while ii < n_exog:
            yield endog, exog
            ii += int(n_part)

    nv_mod = DistributedModel(
        m, estimation_method=_est_regularized_naive, join_method=_join_naive
    )
    fitOLSnv = nv_mod.fit(_rep_data_gen(y, X, m), fit_kwds={"alpha": 0.1})

    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)

    assert_allclose(fitOLSnv.params, fitOLS.params)


def test_debiased_v_average():

    # tests that the debiased method performs better than the standard
    # average.  Does this for both OLS and GLM.

    rs = np.random.RandomState(435265)
    N = 200
    p = 10
    m = 4

    beta = rs.normal(size=p)
    beta = beta * rs.randint(0, 2, p)
    X = rs.normal(size=(N, p))
    y = X.dot(beta) + rs.normal(size=N)

    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    olsdb = np.linalg.norm(fitOLSdb.params - beta)
    n_mod = DistributedModel(
        m, estimation_method=_est_regularized_naive, join_method=_join_naive
    )
    fitOLSn = n_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    olsn = np.linalg.norm(fitOLSn.params - beta)

    assert_(olsdb < olsn)

    prob = 1 / (1 + np.exp(-X.dot(beta) + rs.normal(size=N)))
    y = 1.0 * (prob > 0.5)

    db_mod = DistributedModel(m, model_class=GLM, init_kwds={"family": Binomial()})
    fitGLMdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    glmdb = np.linalg.norm(fitGLMdb.params - beta)
    n_mod = DistributedModel(
        m,
        model_class=GLM,
        init_kwds={"family": Binomial()},
        estimation_method=_est_regularized_naive,
        join_method=_join_naive,
    )
    fitGLMn = n_mod.fit(_data_gen(y, X, m), fit_kwds={"alpha": 0.2})
    glmn = np.linalg.norm(fitGLMn.params - beta)

    assert_(glmdb < glmn)


def _gee_data(n_groups=100, m=4, partitions=2):
    rs = np.random.RandomState(8734)
    groups = np.repeat(np.arange(n_groups), m)
    n = n_groups * m
    X = np.column_stack((np.ones(n), rs.normal(size=(n, 2))))
    y = X.dot([0.5, 1.0, -0.5]) + np.repeat(rs.normal(size=n_groups), m)
    y += rs.normal(size=n)
    blocks = np.array_split(np.arange(n_groups), partitions)
    masks = [np.isin(groups, b) for b in blocks]
    return y, X, groups, masks


def test_gee_unregularized_naive():
    # GH 10778
    y, X, groups, masks = _gee_data()

    mod = DistributedModel(
        len(masks),
        model_class=GEE,
        init_kwds={"cov_struct": Exchangeable()},
        estimation_method=_est_unregularized_naive,
        join_method=_join_naive,
    )
    fit = mod.fit(
        ((y[k], X[k]) for k in masks),
        init_kwds_generator=({"groups": groups[k]} for k in masks),
    )

    params_l = [
        GEE(y[k], X[k], groups=groups[k], cov_struct=Exchangeable()).fit().params
        for k in masks
    ]
    assert_allclose(fit.params, np.mean(params_l, axis=0), rtol=1e-10)
    assert_allclose(fit.predict(X[:5]), X[:5].dot(fit.params), rtol=1e-10)


def test_gee_debiased_raises_before_fitting():
    # GH 10778
    y, X, groups, masks = _gee_data()
    consumed = []

    def data_gen():
        for k in masks:
            consumed.append(k)
            yield y[k], X[k]

    mod = DistributedModel(
        len(masks), model_class=GEE, init_kwds={"cov_struct": Exchangeable()}
    )
    with pytest.raises(NotImplementedError, match="GEE"):
        mod.fit(
            data_gen(),
            fit_kwds={"alpha": 0.1},
            init_kwds_generator=({"groups": groups[k]} for k in masks),
        )
    assert_equal(len(consumed), 0)


class _OLSKeywordFit(OLS):
    # alpha reaches fit_regularized only through **kwargs
    def fit_regularized(self, **kwargs):
        return super().fit_regularized(**kwargs)


class _OLSNoAlpha(OLS):
    # neither an alpha parameter nor **kwargs
    def fit_regularized(self, method="elastic_net"):
        return super().fit_regularized(method=method)


@pytest.mark.parametrize(
    "model_class, raises",
    [
        (OLS, False),
        (_OLSKeywordFit, False),
        (_OLSNoAlpha, True),
        (GEE, True),
        (RLM, True),  # no fit_regularized at all
    ],
)
def test_check_estimation_method_signature(model_class, raises):
    # GH 10778
    mod = DistributedModel(2, model_class=model_class)
    if raises:
        with pytest.raises(NotImplementedError, match=model_class.__name__):
            mod._check_estimation_method()
    else:
        mod._check_estimation_method()


def test_fit_regularized_with_keywords_only():
    # GH 10778, the signature check accepts fit_regularized(**kwargs)
    rs = np.random.RandomState(435265)
    X = rs.normal(size=(50, 3))
    y = rs.randint(0, 2, size=50)

    fits = [
        DistributedModel(2, model_class=model_class).fit(
            _data_gen(y, X, 2), fit_kwds={"alpha": 0.5}
        )
        for model_class in (OLS, _OLSKeywordFit)
    ]
    assert_allclose(fits[1].params, fits[0].params, rtol=1e-12)
    assert_allclose(
        fits[1].params, np.array([-0.157416, -0.029643, -0.471653]), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("init_groups, expected", [(None, [0]), (np.array([3]), [3])])
def test_dummy_model_groups(init_groups, expected):
    # GH 10778, groups is required by GEE: the placeholder model uses the
    # groups of init_kwds if there are any and one group otherwise
    init_kwds = {} if init_groups is None else {"groups": init_groups}
    mod = DistributedModel(2, model_class=GEE, init_kwds=init_kwds)
    assert_equal(mod._dummy_model().groups, expected)
