"""
Results and models pickled by earlier versions of statsmodels

Each test removes the attributes that the objects of an earlier version do not
have, as found by comparing pickles written by statsmodels 0.14.6 and 0.15.0
with objects created by this version, then pickles and loads the object.
"""

import pickle
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM

NOBS = 200


@pytest.fixture
def data():
    rs = np.random.RandomState(987689)
    exog = np.column_stack([np.ones(NOBS), rs.standard_normal((NOBS, 2))])
    endog = rs.poisson(np.exp(0.2 * exog.sum(1)))
    offset = 0.2 * rs.standard_normal(NOBS)
    exposure = np.exp(0.5 * rs.random_sample(NOBS))
    groups = np.repeat(np.arange(NOBS // 5), 5)
    return endog, exog, offset, exposure, groups


def roundtrip(obj, remove=(), remove_if_none=()):
    # Remove the attributes that the object would not have had in an earlier
    # version, then pickle and load it
    state = obj.__dict__
    for name in remove:
        del state[name]
    for name in remove_if_none:
        if state.get(name, 0) is None:
            del state[name]
    return pickle.loads(pickle.dumps(obj))


# Attributes that GLM.__init__ sets since 0.15.0. Models pickled by 0.14
# also lack offset and exposure when they were not given.
GLM_014_MISSING = [
    "_has_offset",
    "_has_exposure",
    "_offset_name",
    "_exposure_name",
    "_freq_weights_name",
    "_var_weights_name",
    "_optim_hessian",
    "_tmp_like_exog",
]


def glm_roundtrip_014(res):
    res._results.model = roundtrip(
        res.model, GLM_014_MISSING, remove_if_none=["offset", "exposure"]
    )
    return res


@pytest.mark.parametrize("kind", ["none", "offset", "exposure"])
def test_glm_014(data, kind):
    endog, exog, offset, exposure, _ = data
    kwargs = {}
    if kind == "offset":
        kwargs["offset"] = offset
    elif kind == "exposure":
        kwargs["exposure"] = exposure
    res = GLM(endog, exog, family=families.Poisson(), **kwargs).fit()
    expected = res.predict()
    llf = res.llf
    res = glm_roundtrip_014(res)
    model = res.model
    assert model._has_offset is (kind == "offset")
    assert model._has_exposure is (kind == "exposure")
    if kind != "offset":
        assert model.offset is None
    if kind != "exposure":
        assert model.exposure is None
    assert model._tmp_like_exog.shape == exog.shape
    # in-sample prediction uses the offset or exposure of the model
    assert_allclose(res.predict(), expected, rtol=1e-13)
    assert_allclose(res.llf, llf, rtol=1e-13)
    res.summary()
    # the loaded model can be fit again
    res2 = model.fit()
    assert_allclose(res2.params, res.params, rtol=1e-10)


def test_glm_014_remove_data(data):
    endog, exog, offset, _, _ = data
    res = GLM(endog, exog, family=families.Poisson(), offset=offset).fit()
    res.summary()
    res.remove_data()
    res = glm_roundtrip_014(res)
    assert res.model._has_offset
    assert res.model._tmp_like_exog is None
    res.summary()
    assert_allclose(
        res.predict(exog[:5], offset=offset[:5]),
        np.exp(exog[:5] @ res.params + offset[:5]),
        rtol=1e-13,
    )


def test_gee_014(data):
    endog, exog, offset, _, groups = data
    res = GEE(endog, exog, groups=groups, family=families.Poisson(), offset=offset)
    res = res.fit()
    expected = res.predict()
    res = glm_roundtrip_014(res)
    assert res.model._has_offset
    assert_allclose(res.predict(), expected, rtol=1e-13)
    res.summary()


def test_glm_new_pickle_unchanged(data):
    # A model pickled by this version keeps the values it had
    endog, exog, offset, _, _ = data
    model = GLM(endog, exog, family=families.Poisson(), offset=offset)
    model._offset_name = "name"
    model2 = pickle.loads(pickle.dumps(model))
    assert model2._offset_name == "name"
    assert model2._has_offset
    assert model2.exposure is None


@pytest.mark.parametrize(
    "model, separation", [("logit", False), ("probit", False), ("logit", True)]
)
def test_binary_014(model, separation):
    # 0.14 computed the separation warning in summary, from the data
    rs = np.random.RandomState(987689)
    exog = np.column_stack([np.ones(NOBS), rs.standard_normal((NOBS, 2))])
    noise = 0.05 if separation else 1.0
    endog = (exog[:, 1] + noise * rs.standard_normal(NOBS) > 0).astype(float)
    mod = {"logit": Logit, "probit": Probit}[model](endog, exog)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=0, maxiter=200)
    expected = res.summary().extra_txt
    assert bool(expected) is separation
    res._results = roundtrip(res._results, ["_separation_etext"])
    assert res.summary().extra_txt == expected
