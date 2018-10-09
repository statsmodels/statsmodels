import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd

def test1d():

    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]

    model = ConditionalLogit(y, x, g)

    # Check the gradient for the denominator of the partial likelihood
    for x in -1, 0, 1, 2:
        params = np.r_[x,]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x))
        assert_allclose(grad, ngrad)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        grad = approx_fprime(np.r_[x,], model.loglike)
        score = model.score(np.r_[x,])
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[0.9272407], rtol=1e-5)
    assert_allclose(result.bse, np.r_[1.295155], rtol=1e-5)


def test2d():

    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    x = np.empty((10, 2))
    x[:, 0] = x1
    x[:, 1] = x2

    model = ConditionalLogit(y, x, g)

    # Check the gradient for the denominator of the partial likelihood
    for x in -1, 0, 1, 2:
        params = np.r_[x, -1.5*x]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x))
        assert_allclose(grad, ngrad, rtol=1e-5)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        params = np.r_[-0.5*x, 0.5*x]
        grad = approx_fprime(params, model.loglike)
        score = model.score(params)
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[1.011074, 1.236758], rtol=1e-3)
    assert_allclose(result.bse, np.r_[1.420784, 1.361738], rtol=1e-5)

    result.summary()

def test_formula():

    np.random.seed(34234)
    n = 200
    y = np.random.randint(0, 2, size=n)
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    g = np.random.randint(0, 25, size=n)

    x = np.hstack((x1[:, None], x2[:, None]))
    model1 = ConditionalLogit(y, x, g)
    result1 = model1.fit()

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
    model2 = ConditionalLogit.from_formula("y ~ 0 + x1 + x2", groups="g", data=df)
    result2 = model2.fit()

    assert_allclose(result1.params, result2.params, rtol=1e-5)
    assert_allclose(result1.bse, result2.bse, rtol=1e-5)
    assert_allclose(result1.cov_params(), result2.cov_params(), rtol=1e-5)
    assert_allclose(result1.tvalues, result2.tvalues, rtol=1e-5)
