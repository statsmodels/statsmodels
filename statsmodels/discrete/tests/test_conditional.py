from itertools import combinations

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.discrete.conditional_models import (
    ConditionalLogit,
    ConditionalMNLogit,
    ConditionalPoisson,
)
from statsmodels.iolib.summary import Summary
from statsmodels.tools.numdiff import approx_fprime


@pytest.mark.parametrize("method", ["loglike", "score"])
@pytest.mark.parametrize("n_success", [1, 2, 1499])
@pytest.mark.parametrize("use_offset", [False, True])
def test_logit_large_group(method, n_success, use_offset):
    nobs = 1500
    exog = np.linspace(-1, 1, nobs)[:, None]
    endog = np.zeros(nobs)
    endog[:n_success] = 1
    params = np.array([0.2])
    offset = np.linspace(-0.3, 0.3, nobs) if use_offset else np.zeros(nobs)
    model = ConditionalLogit(endog, exog, groups=np.zeros(nobs), offset=offset)

    # The one-success, two-success and one-failure cases have closed forms.
    # For two successes, sum_{i<j} w_i*w_j = ((sum w)**2 - sum w**2) / 2.
    linear = exog @ params + offset
    weights = np.exp(linear)
    if n_success == 1:
        denom = weights.sum()
        mean_sum = weights @ exog / denom
    elif n_success == 2:
        denom = (weights.sum() ** 2 - (weights**2).sum()) / 2
        mean_sum = (weights.sum() * (weights @ exog) - weights**2 @ exog) / denom
    else:
        inv_weights = 1 / weights
        denom = np.exp(linear.sum()) * inv_weights.sum()
        mean_sum = exog.sum(axis=0) - inv_weights @ exog / inv_weights.sum()
    if method == "loglike":
        expected = endog @ linear - np.log(denom)
    else:
        expected = endog @ exog - mean_sum
    assert_allclose(getattr(model, method)(params), expected, rtol=1e-12)


@pytest.mark.parametrize("n_success", [1, 2, 4, 5])
@pytest.mark.parametrize("use_offset", [False, True])
def test_logit_enumerated_likelihood(n_success, use_offset):
    exog = np.column_stack((np.linspace(-1, 1, 6), [0, 1, -1, 0.5, 2, -0.5]))
    endog = np.zeros(6)
    endog[:n_success] = 1
    params = np.array([0.2, -0.3])
    offset = np.linspace(-0.4, 0.2, 6) if use_offset else np.zeros(6)
    model = ConditionalLogit(endog, exog, groups=np.zeros(6), offset=offset)

    # Enumerating every possible success subset is independent of the dynamic
    # program. Its weighted covariate mean and covariance give score/Hessian.
    subsets = np.array(list(combinations(range(6), n_success)))
    subset_exog = exog[subsets].sum(axis=1)
    linear = exog @ params + offset
    weights = np.exp(linear[subsets].sum(axis=1))
    mean_exog = weights @ subset_exog / weights.sum()
    expected_score = endog @ exog - mean_exog
    expected_hess = (
        np.outer(mean_exog, mean_exog)
        - (subset_exog.T * weights) @ subset_exog / weights.sum()
    )
    expected_ll = endog @ linear - np.log(weights.sum())
    assert_allclose(model.loglike(params), expected_ll, rtol=1e-13)
    assert_allclose(model.score(params), expected_score, rtol=1e-13, atol=1e-14)

    for j in range(len(params)):
        complex_params = params.astype(complex)
        complex_params[j] += 1e-20j
        assert_allclose(
            model.loglike(complex_params).imag / 1e-20,
            expected_score[j],
            rtol=1e-12,
            atol=1e-13,
        )
        assert_allclose(
            model.score(complex_params).imag / 1e-20,
            expected_hess[:, j],
            rtol=1e-12,
            atol=1e-13,
        )


def test_logit_1d():

    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]

    model = ConditionalLogit(y, x, groups=g)

    # Check the gradient for the denominator of the partial likelihood
    for x in -1, 0, 1, 2:
        params = np.r_[x,]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x)).squeeze()
        assert_allclose(grad, ngrad)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        grad = approx_fprime(np.r_[x,], model.loglike).squeeze()
        score = model.score(np.r_[x,])
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[0.9272407], rtol=1e-5)
    assert_allclose(result.bse, np.r_[1.295155], rtol=1e-5)


def test_logit_formula():
    """Test that ConditionalLogit uses the right environment for formulas"""

    def times_two(x):
        return 2 * x

    groups = np.repeat([0, 1], 50)
    exog = np.linspace(-2, 2, len(groups))

    # Random endog with within-group variance, a deterministic threshold on exog
    # is perfectly separated and one group has no variation
    rng = np.random.default_rng(123)
    prob = 1 / (1 + np.exp(-(0.5 * exog + groups - 0.5)))
    endog = rng.binomial(1, prob)

    data = pd.DataFrame({"exog": exog, "groups": groups, "endog": endog})

    result_direct = ConditionalLogit(endog, times_two(exog), groups=groups).fit()

    result_formula = ConditionalLogit.from_formula(
        "endog ~ 0 + times_two(exog)", groups="groups", data=data
    ).fit()

    assert_allclose(result_direct.params, result_formula.params)
    assert_allclose(result_direct.bse, result_formula.bse)


def test_logit_2d():

    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    x = np.empty((10, 2))
    x[:, 0] = x1
    x[:, 1] = x2

    model = ConditionalLogit(y, x, groups=g)

    # Check the gradient for the denominator of the partial likelihood
    for x in -1, 0, 1, 2:
        params = np.r_[x, -1.5 * x]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x))
        assert_allclose(grad, ngrad, rtol=1e-5)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        params = np.r_[-0.5 * x, 0.5 * x]
        grad = approx_fprime(params, model.loglike)
        score = model.score(params)
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[1.011074, 1.236758], rtol=1e-3)
    assert_allclose(result.bse, np.r_[1.420784, 1.361738], rtol=1e-5)

    result.summary()


def test_formula():

    for j in 0, 1:

        rs = np.random.RandomState(34234)
        n = 200
        y = rs.randint(0, 2, size=n)
        x1 = rs.normal(size=n)
        x2 = rs.normal(size=n)
        g = rs.randint(0, 25, size=n)

        x = np.hstack((x1[:, None], x2[:, None]))
        if j == 0:
            model1 = ConditionalLogit(y, x, groups=g)
        else:
            model1 = ConditionalPoisson(y, x, groups=g)
        result1 = model1.fit()

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
        if j == 0:
            model2 = ConditionalLogit.from_formula(
                "y ~ 0 + x1 + x2", groups="g", data=df
            )
        else:
            model2 = ConditionalPoisson.from_formula(
                "y ~ 0 + x1 + x2", groups="g", data=df
            )
        result2 = model2.fit()

        assert_allclose(result1.params, result2.params, rtol=1e-5)
        assert_allclose(result1.bse, result2.bse, rtol=1e-5)
        assert_allclose(result1.cov_params(), result2.cov_params(), rtol=1e-5)
        assert_allclose(result1.tvalues, result2.tvalues, rtol=1e-5)


def test_poisson_1d():

    y = np.r_[3, 1, 1, 4, 5, 2, 0, 1, 6, 2]
    g = np.r_[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]

    model = ConditionalPoisson(y, x, groups=g)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        grad = approx_fprime(np.r_[x,], model.loglike).squeeze()
        score = model.score(np.r_[x,])
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[0.6466272], rtol=1e-4)
    assert_allclose(result.bse, np.r_[0.4170918], rtol=1e-5)


def test_poisson_2d():

    y = np.r_[3, 1, 4, 8, 2, 5, 4, 7, 2, 6]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[2, 1, 0, 0, 1, 2, 3, 2, 0, 1]
    x = np.empty((10, 2))
    x[:, 0] = x1
    x[:, 1] = x2

    model = ConditionalPoisson(y, x, groups=g)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        params = np.r_[-0.5 * x, 0.5 * x]
        grad = approx_fprime(params, model.loglike)
        score = model.score(params)
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[-0.9478957, -0.0134279], rtol=1e-3)
    assert_allclose(result.bse, np.r_[0.3874942, 0.1686712], rtol=1e-5)

    result.summary()


def test_lasso_logistic():

    rs = np.random.RandomState(3423948)

    n = 200
    groups = np.arange(10)
    groups = np.kron(groups, np.ones(n // 10))
    group_effects = rs.normal(size=10)
    group_effects = np.kron(group_effects, np.ones(n // 10))

    x = rs.normal(size=(n, 4))
    params = np.r_[0, 0, 1, 0]
    lin_pred = np.dot(x, params) + group_effects

    mean = 1 / (1 + np.exp(-lin_pred))
    y = (rs.uniform(size=n) < mean).astype(int)

    model0 = ConditionalLogit(y, x, groups=groups)
    result0 = model0.fit()

    # Should be the same as model0
    model1 = ConditionalLogit(y, x, groups=groups)
    result1 = model1.fit_regularized(L1_wt=0, alpha=0)

    assert_allclose(result0.params, result1.params, rtol=1e-3)

    model2 = ConditionalLogit(y, x, groups=groups)
    result2 = model2.fit_regularized(L1_wt=1, alpha=0.05)

    # Rxegression test
    assert_allclose(result2.params, np.r_[0, 0, 0.55235152, 0], rtol=1e-4)

    # Test with formula
    df = pd.DataFrame(
        {
            "y": y,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x3": x[:, 2],
            "x4": x[:, 3],
            "groups": groups,
        }
    )
    fml = "y ~ 0 + x1 + x2 + x3 + x4"
    model3 = ConditionalLogit.from_formula(fml, groups="groups", data=df)
    result3 = model3.fit_regularized(L1_wt=1, alpha=0.05)
    assert_allclose(result2.params, result3.params)


def test_lasso_poisson():

    rs = np.random.RandomState(342394)

    n = 200
    groups = np.arange(10)
    groups = np.kron(groups, np.ones(n // 10))
    group_effects = rs.normal(size=10)
    group_effects = np.kron(group_effects, np.ones(n // 10))

    x = rs.normal(size=(n, 4))
    params = np.r_[0, 0, 1, 0]
    lin_pred = np.dot(x, params) + group_effects

    mean = np.exp(lin_pred)
    y = rs.poisson(mean)

    model0 = ConditionalPoisson(y, x, groups=groups)
    result0 = model0.fit()

    # Should be the same as model0
    model1 = ConditionalPoisson(y, x, groups=groups)
    result1 = model1.fit_regularized(L1_wt=0, alpha=0)

    assert_allclose(result0.params, result1.params, rtol=1e-3)

    model2 = ConditionalPoisson(y, x, groups=groups)
    result2 = model2.fit_regularized(L1_wt=1, alpha=0.2)

    # Regression test
    assert_allclose(result2.params, np.r_[0, 0, 0.91697508, 0], rtol=1e-4)

    # Test with formula
    df = pd.DataFrame(
        {
            "y": y,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x3": x[:, 2],
            "x4": x[:, 3],
            "groups": groups,
        }
    )
    fml = "y ~ 0 + x1 + x2 + x3 + x4"
    model3 = ConditionalPoisson.from_formula(fml, groups="groups", data=df)
    result3 = model3.fit_regularized(L1_wt=1, alpha=0.2)
    assert_allclose(result2.params, result3.params)


def test_fit_regularized_invalid_method():
    rs = np.random.RandomState(3423948)
    n = 200
    groups = np.kron(np.arange(10), np.ones(n // 10))
    x = rs.normal(size=(n, 2))
    y = (rs.uniform(size=n) < 0.5).astype(int)

    model = ConditionalLogit(y, x, groups=groups)
    with pytest.raises(ValueError, match="method"):
        model.fit_regularized(method="not-a-method")


def gen_mnlogit(n):

    rs = np.random.RandomState(235)

    g = np.kron(np.ones(5), np.arange(n // 5))
    x1 = rs.normal(size=n)
    x2 = rs.normal(size=n)
    xm = np.concatenate((x1[:, None], x2[:, None]), axis=1)
    pa = np.array([[0, 1, -1], [0, 2, -1]])
    lpr = np.dot(xm, pa)
    pr = np.exp(lpr)
    pr /= pr.sum(1)[:, None]
    cpr = pr.cumsum(1)
    y = 2 * np.ones(n)
    u = rs.uniform(size=n)
    y[u < cpr[:, 2]] = 2
    y[u < cpr[:, 1]] = 1
    y[u < cpr[:, 0]] = 0

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
    return df, rs


def test_conditional_mnlogit_grad():
    df, rs = gen_mnlogit(90)
    model = ConditionalMNLogit.from_formula("y ~ 0 + x1 + x2", groups="g", data=df)

    # Compare the gradients to numeric gradients
    for _ in range(5):
        za = rs.normal(size=4)
        grad = model.score(za)
        ngrad = approx_fprime(za, model.loglike)
        assert_allclose(grad, ngrad, rtol=1e-5, atol=1e-3)


def test_conditional_mnlogit_2d():

    df, rs = gen_mnlogit(90)
    model = ConditionalMNLogit.from_formula("y ~ 0 + x1 + x2", groups="g", data=df)
    result = model.fit(rng=rs)

    # Regression tests
    assert_allclose(
        result.params,
        np.asarray([[0.75592035, -1.58565494], [1.82919869, -1.32594231]]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert_allclose(
        result.bse,
        np.asarray([[0.68099698, 0.70142727], [0.65190315, 0.59653771]]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_conditional_mnlogit_3d():
    df, rs = gen_mnlogit(90)
    df["x3"] = rs.normal(size=df.shape[0])
    model = ConditionalMNLogit.from_formula("y ~ 0 + x1 + x2 + x3", groups="g", data=df)
    result = model.fit(rng=rs)

    # Regression tests
    assert_allclose(
        result.params,
        np.asarray(
            [[0.729629, -1.633673], [1.879019, -1.327163], [-0.114124, -0.109378]]
        ),
        atol=1e-5,
        rtol=1e-5,
    )

    assert_allclose(
        result.bse,
        np.asarray([[0.682965, 0.60472], [0.672947, 0.42401], [0.722631, 0.33663]]),
        atol=1e-5,
        rtol=1e-5,
    )

    # Smoke test
    result.summary()


@pytest.mark.parametrize(
    "rng",
    [0, np.random.RandomState(0), np.random.default_rng(0)],
    ids=["int", "randomstate", "generator"],
)
def test_conditional_mnlogit_fit_rng_types(recwarn, rng):
    # GH: fit() used to special-case rng=None with an unconditional
    # FutureWarning and global np.random state instead of routing
    # through check_random_state like every other rng-accepting
    # function, so the default call emitted a warning that no other
    # canonical rng parameter does.
    df, _ = gen_mnlogit(60)
    model = ConditionalMNLogit.from_formula("y ~ 0 + x1 + x2", groups="g", data=df)
    model.fit(rng=rng, disp=False)
    future_warnings = [w for w in recwarn.list if issubclass(w.category, FutureWarning)]
    assert not future_warnings, [str(w.message) for w in future_warnings]


@pytest.mark.singleton_randomstate
def test_conditional_mnlogit_fit_warn():
    # GH: fit() used to special-case rng=None with an unconditional
    # FutureWarning and global np.random state instead of routing
    # through check_random_state like every other rng-accepting
    # function, so the default call emitted a warning that no other
    # canonical rng parameter does.
    df, _ = gen_mnlogit(60)
    model = ConditionalMNLogit.from_formula("y ~ 0 + x1 + x2", groups="g", data=df)
    with pytest.warns(FutureWarning, match="When start_params is not specified,"):
        model.fit(rng=None, disp=False)


def test_conditional_mnlogit_fit_rng_reproducible():
    df, _ = gen_mnlogit(60)
    model = ConditionalMNLogit.from_formula("y ~ 0 + x1 + x2", groups="g", data=df)
    res1 = model.fit(rng=0, disp=False)
    res2 = model.fit(rng=0, disp=False)
    assert_allclose(res1.params, res2.params)


def test_skip_hessian():
    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]

    model = ConditionalLogit(y, x, groups=g)
    result_no_hess = model.fit(skip_hessian=True)
    result_hess = model.fit(skip_hessian=False)
    assert result_no_hess.normalized_cov_params is None
    assert isinstance(result_hess.normalized_cov_params, np.ndarray)


def _fit_conditional_logit_for_summary():
    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    x = np.column_stack((x1, x2))
    return ConditionalLogit(y, x, groups=g).fit()


def _fit_conditional_poisson_for_summary():
    y = np.r_[3, 1, 4, 8, 2, 5, 4, 7, 2, 6]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[2, 1, 0, 0, 1, 2, 3, 2, 0, 1]
    x = np.column_stack((x1, x2))
    return ConditionalPoisson(y, x, groups=g).fit()


def _fit_conditional_mnlogit_for_summary():
    df, rs = gen_mnlogit(90)
    model = ConditionalMNLogit.from_formula("y ~ 0 + x1 + x2", groups="g", data=df)
    return model.fit(rng=rs)


@pytest.mark.parametrize(
    "fit_func",
    [
        _fit_conditional_logit_for_summary,
        _fit_conditional_poisson_for_summary,
        _fit_conditional_mnlogit_for_summary,
    ],
    ids=["ConditionalLogit", "ConditionalPoisson", "ConditionalMNLogit"],
)
def test_summary_after_remove_data(fit_func):
    # summary() must still work after remove_data() has been called
    res = fit_func()
    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)
