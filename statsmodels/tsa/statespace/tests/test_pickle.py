"""
Tests for python wrapper of state space representation and filtering

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.
"""
import copy
import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.testing import warns_or_none
from statsmodels.tsa.statespace import (
    dynamic_factor,
    exponential_smoothing,
    sarimax,
    varmax,
)
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.mlemodel import MLEResults
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .results import results_kalman_filter


@pytest.fixture
def data():
    true = results_kalman_filter.uc_uni
    data_ = pd.DataFrame(
        true["data"],
        index=pd.date_range("1947-01-01", "1995-07-01", freq="QS"),
        columns=["GDP"]
    )
    data_["lgdp"] = np.log(data_["GDP"])
    return data_


def test_pickle_fit_sarimax(data):
    # Fit an ARIMA(1,1,0) to log GDP
    mod = sarimax.SARIMAX(data["lgdp"], order=(1, 1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))

    # The start values are already at the maximum. L-BFGS reports convergence
    # with SciPy 1.8, but newer SciPy stops at the start values with a
    # line-search warning (warnflag 2), as Newton's method does. The two fits
    # agree either way, which is what is tested.
    with warns_or_none(ConvergenceWarning, match="Maximum Likelihood"):
        res = mod.fit(disp=-1, full_output=True, method="lbfgs")
        pkl_res = pkl_mod.fit(disp=-1, full_output=True, method="lbfgs")

    assert_allclose(res.llf_obs, pkl_res.llf_obs)
    assert_allclose(res.tvalues, pkl_res.tvalues)
    assert_allclose(res.smoothed_state, pkl_res.smoothed_state)
    assert_allclose(res.resid.values, pkl_res.resid.values)
    assert_allclose(res.impulse_responses(10), res.impulse_responses(10))


def test_unobserved_components_pickle():
    # Tests for missing data
    nobs = 20
    k_endog = 1
    rg = np.random.RandomState(1208)
    endog = rg.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog2 = rg.normal(size=(nobs, 2))

    index = pd.date_range("1970-01-01", freq="QS", periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        UnobservedComponents(endog, "llevel", exog=exog2),
        UnobservedComponents(endog_pd, "llevel", exog=exog2_pd),
    ]

    for mod in models:
        # Smoke tests
        pkl_mod = pickle.loads(pickle.dumps(mod))
        assert_equal(mod.start_params, pkl_mod.start_params)
        res = mod.fit(disp=False)
        pkl_res = pkl_mod.fit(disp=False)

        assert_allclose(res.llf_obs, pkl_res.llf_obs)
        assert_allclose(res.tvalues, pkl_res.tvalues)
        assert_allclose(res.smoothed_state, pkl_res.smoothed_state)
        assert_allclose(res.resid, pkl_res.resid)
        assert_allclose(res.impulse_responses(10), res.impulse_responses(10))


def test_kalman_filter_pickle(data):
    # Construct the statespace representation
    true = results_kalman_filter.uc_uni
    k_states = 4
    model = KalmanFilter(k_endog=1, k_states=k_states)
    model.bind(data["lgdp"].values)

    model.design[:, :, 0] = [1, 1, 0, 0]
    model.transition[([0, 0, 1, 1, 2, 3],
                      [0, 3, 1, 2, 1, 3],
                      [0, 0, 0, 0, 0, 0])] = [1, 1, 0, 0, 1, 1]
    model.selection = np.eye(model.k_states)

    # Update matrices with given parameters
    (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = np.array(
        true["parameters"]
    )
    model.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
    model.state_cov[
        np.diag_indices(k_states) + (np.zeros(k_states, dtype=int),)] = [
        sigma_v ** 2, sigma_e ** 2, 0, sigma_w ** 2
    ]

    # Initialization
    initial_state = np.zeros((k_states,))
    initial_state_cov = np.eye(k_states) * 100

    # Initialization: modification
    initial_state_cov = np.dot(
        np.dot(model.transition[:, :, 0], initial_state_cov),
        model.transition[:, :, 0].T
    )
    model.initialize_known(initial_state, initial_state_cov)
    pkl_mod = pickle.loads(pickle.dumps(model))

    results = model.filter()
    pkl_results = pkl_mod.filter()

    assert_allclose(results.llf_obs[true["start"]:].sum(),
                    pkl_results.llf_obs[true["start"]:].sum())
    assert_allclose(results.filtered_state[0][true["start"]:],
                    pkl_results.filtered_state[0][true["start"]:])
    assert_allclose(results.filtered_state[1][true["start"]:],
                    pkl_results.filtered_state[1][true["start"]:])
    assert_allclose(results.filtered_state[3][true["start"]:],
                    pkl_results.filtered_state[3][true["start"]:])


def test_representation_pickle():
    nobs = 10
    k_endog = 2
    arr = np.arange(nobs * k_endog).reshape(k_endog, nobs) * 1.
    endog = np.asfortranarray(arr)
    mod = Representation(endog, k_states=2)
    pkl_mod = pickle.loads(pickle.dumps(mod))

    assert_equal(mod.nobs, pkl_mod.nobs)
    assert_equal(mod.k_endog, pkl_mod.k_endog)

    mod._initialize_representation()
    pkl_mod._initialize_representation()
    assert_equal(mod.design, pkl_mod.design)
    assert_equal(mod.obs_intercept, pkl_mod.obs_intercept)
    assert_equal(mod.initial_variance, pkl_mod.initial_variance)


# Parameters at which a model is used before it is copied, and different
# parameters at which the copy is then evaluated
USED_MODEL_PARAMS = {
    "sarimax": ([1.0, 0.5, 0.2, 1.0], [2.0, 0.8, -0.3, 0.5]),
    "unobserved_components": ([1.0, 0.5, 0.1], [0.2, 1.5, -0.5]),
    "varmax": (
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0],
        [0.1, -0.1, -0.4, 0.1, 0.2, 0.5, 0.8, 0.3, 1.2],
    ),
    "dynamic_factor": ([1.0, 0.5, 1.0, 1.0, 0.5], [0.3, -0.8, 0.5, 2.0, -0.6]),
    "exponential_smoothing": ([0.1, 0.01, 10.0, 0.0], [0.6, 0.2, 9.0, -0.5]),
}


def _make_model(name):
    # The same data on every call, so that a fresh model is the reference
    rng = np.random.default_rng(20261119)
    nobs = 60
    endog = np.cumsum(rng.standard_normal(nobs)) + 10
    exog = rng.standard_normal(nobs)
    endog2 = rng.standard_normal((nobs, 2))
    if name == "sarimax":
        return sarimax.SARIMAX(endog, order=(1, 0, 1), trend="c")
    if name == "unobserved_components":
        # The regression term is a time-varying observation intercept
        return UnobservedComponents(endog, "llevel", exog=exog)
    if name == "varmax":
        return varmax.VARMAX(endog2, order=(1, 0))
    if name == "dynamic_factor":
        return dynamic_factor.DynamicFactor(endog2, k_factors=1, factor_order=1)
    # The known initialization is updated with the parameters
    return exponential_smoothing.ExponentialSmoothing(endog, trend=True)


@pytest.mark.parametrize("roundtrip", ["pickle", "deepcopy"])
@pytest.mark.parametrize("used_by", ["loglike", "smooth"])
@pytest.mark.parametrize("name", list(USED_MODEL_PARAMS))
def test_model_copied_after_use(name, used_by, roundtrip):
    # A model that was copied after it had been used ignored new parameters
    # and kept using the system matrices of its last evaluation
    params0, params1 = USED_MODEL_PARAMS[name]
    mod = _make_model(name)
    getattr(mod, used_by)(params0)
    if roundtrip == "pickle":
        copied = pickle.loads(pickle.dumps(mod))
    else:
        copied = copy.deepcopy(mod)

    fresh = _make_model(name)
    assert_allclose(copied.loglike(params1), fresh.loglike(params1), rtol=1e-10)
    assert_allclose(
        copied.filter(params1).llf, fresh.filter(params1).llf, rtol=1e-10
    )
    smoothed2 = copied.smooth(params1)
    smoothed = fresh.smooth(params1)
    assert_allclose(smoothed2.smoothed_state, smoothed.smoothed_state, rtol=1e-10)
    # The standard errors use the complex-step representation
    assert_allclose(smoothed2.bse, smoothed.bse, rtol=1e-10)


@pytest.mark.parametrize("roundtrip", ["pickle", "save"])
@pytest.mark.parametrize("name", list(USED_MODEL_PARAMS))
def test_results_copied_after_use(name, roundtrip, tmp_path):
    params0, params1 = USED_MODEL_PARAMS[name]
    res = _make_model(name).smooth(params0)
    if roundtrip == "pickle":
        res2 = pickle.loads(pickle.dumps(res))
    else:
        path = str(tmp_path / "results.pkl")
        res.save(path)
        res2 = MLEResults.load(path)

    # The stored output is unchanged
    assert_allclose(res2.llf, res.llf, rtol=1e-10)
    assert_allclose(res2.smoothed_state, res.smoothed_state, rtol=1e-10)

    # and the model follows new parameters
    fresh = _make_model(name)
    assert_allclose(res2.model.loglike(params1), fresh.loglike(params1), rtol=1e-10)
    smoothed2 = res2.model.smooth(params1)
    smoothed = fresh.smooth(params1)
    assert_allclose(smoothed2.llf, smoothed.llf, rtol=1e-10)
    assert_allclose(smoothed2.smoothed_state, smoothed.smoothed_state, rtol=1e-10)


def test_fitted_results_pickled():
    # The model of pickled fitted results returned the fitted loglikelihood
    # for any parameters
    res = _make_model("sarimax").fit(disp=False)
    res2 = pickle.loads(pickle.dumps(res))
    params1 = USED_MODEL_PARAMS["sarimax"][1]
    fresh = _make_model("sarimax")
    assert_allclose(res2.model.loglike(params1), fresh.loglike(params1), rtol=1e-10)


def _roundtrip(mod, roundtrip):
    if roundtrip == "pickle":
        return pickle.loads(pickle.dumps(mod))
    return copy.deepcopy(mod)


@pytest.mark.parametrize("roundtrip", ["pickle", "deepcopy"])
@pytest.mark.parametrize("name", list(USED_MODEL_PARAMS))
def test_model_copied_before_initialization(name, roundtrip):
    # simulation_smoother creates the compiled statespace object without
    # initializing it. A copy of the model restored it without its arrays for
    # the initial state, so the copy could not compute the loglikelihood
    params0, params1 = USED_MODEL_PARAMS[name]
    mod = _make_model(name)
    mod.update(params0)
    mod.simulation_smoother()
    copied = _roundtrip(mod, roundtrip)

    fresh = _make_model(name)
    assert_allclose(copied.loglike(params1), fresh.loglike(params1), rtol=1e-10)
    smoothed2 = copied.smooth(params1)
    smoothed = fresh.smooth(params1)
    assert_allclose(smoothed2.smoothed_state, smoothed.smoothed_state, rtol=1e-10)


@pytest.mark.parametrize("roundtrip", ["pickle", "deepcopy"])
def test_model_copied_after_incomplete_initialization(roundtrip):
    # The compiled statespace object also stays uninitialized when filtering
    # fails because the initialization is incomplete
    params0, params1 = USED_MODEL_PARAMS["sarimax"]
    mod = _make_model("sarimax")
    mod.ssm.initialization = Initialization(mod.k_states)
    with pytest.raises(RuntimeError, match="Initialization is incomplete"):
        mod.loglike(params0)
    mod.ssm.initialize_stationary()
    copied = _roundtrip(mod, roundtrip)

    fresh = _make_model("sarimax")
    fresh.ssm.initialize_stationary()
    assert_allclose(copied.loglike(params1), fresh.loglike(params1), rtol=1e-10)
