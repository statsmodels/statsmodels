"""
Tests for the generic MLEModel

Author: Chad Fulton
License: Simplified-BSD
"""
from statsmodels.compat.pandas import MONTH_END

import inspect
from pathlib import Path
import re
from types import SimpleNamespace
import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_equal,
)
import pandas as pd
import pytest
from scipy.linalg import solve_discrete_lyapunov

from statsmodels.datasets import macrodata, nile
from statsmodels.iolib.summary import Summary
from statsmodels.tools.numdiff import approx_fprime, approx_hess3
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.testing import warns_or_none
from statsmodels.tsa.statespace import (
    dynamic_factor,
    kalman_filter,
    kalman_smoother,
    sarimax,
    structural,
    varmax,
)
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.mlemodel import (
    MLEModel,
    MLEResults,
    MLEResultsWrapper,
)
from statsmodels.tsa.statespace.news import NewsResults
from statsmodels.tsa.statespace.tests.results import (
    results_sarimax,
    results_var_misc,
)

current_path = Path(__file__).resolve().parent

# Basic kwargs
kwargs = {
    "k_states": 1,
    "design": [[1]],
    "transition": [[1]],
    "selection": [[1]],
    "state_cov": [[1]],
    "initialization": "approximate_diffuse",
}


def get_dummy_mod(fit=True, pandas=False):
    # This tests time-varying parameters regression when in fact the parameters
    # are not time-varying, and in fact the regression fit is perfect
    endog = np.arange(100) * 1.0
    exog = 2 * endog

    if pandas:
        index = pd.date_range("1960-01-01", periods=100, freq="MS")
        endog = pd.Series(endog, index=index)
        exog = pd.Series(exog, index=index)

    mod = sarimax.SARIMAX(
        endog,
        exog=exog,
        order=(0, 0, 0),
        time_varying_regression=True,
        mle_regression=False,
        use_exact_diffuse=True,
    )

    if fit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mod.fit(disp=-1)
    else:
        res = None

    return mod, res


def test_init_matrices_time_invariant():
    # Test setting state space system matrices in __init__, with time-invariant
    # matrices
    k_endog = 2
    k_states = 3
    k_posdef = 1

    endog = np.zeros((10, 2))
    obs_intercept = np.arange(k_endog) * 1.0
    design = np.reshape(np.arange(k_endog * k_states) * 1.0, (k_endog, k_states))
    obs_cov = np.reshape(np.arange(k_endog**2) * 1.0, (k_endog, k_endog))
    state_intercept = np.arange(k_states) * 1.0
    transition = np.reshape(np.arange(k_states**2) * 1.0, (k_states, k_states))
    selection = np.reshape(np.arange(k_states * k_posdef) * 1.0, (k_states, k_posdef))
    state_cov = np.reshape(np.arange(k_posdef**2) * 1.0, (k_posdef, k_posdef))

    mod = MLEModel(
        endog,
        k_states=k_states,
        k_posdef=k_posdef,
        obs_intercept=obs_intercept,
        design=design,
        obs_cov=obs_cov,
        state_intercept=state_intercept,
        transition=transition,
        selection=selection,
        state_cov=state_cov,
    )

    assert_allclose(mod["obs_intercept"], obs_intercept)
    assert_allclose(mod["design"], design)
    assert_allclose(mod["obs_cov"], obs_cov)
    assert_allclose(mod["state_intercept"], state_intercept)
    assert_allclose(mod["transition"], transition)
    assert_allclose(mod["selection"], selection)
    assert_allclose(mod["state_cov"], state_cov)


def test_init_matrices_time_varying():
    # Test setting state space system matrices in __init__, with time-varying
    # matrices
    nobs = 10
    k_endog = 2
    k_states = 3
    k_posdef = 1

    endog = np.zeros((10, 2))
    obs_intercept = np.reshape(np.arange(k_endog * nobs) * 1.0, (k_endog, nobs))
    design = np.reshape(
        np.arange(k_endog * k_states * nobs) * 1.0, (k_endog, k_states, nobs)
    )
    obs_cov = np.reshape(np.arange(k_endog**2 * nobs) * 1.0, (k_endog, k_endog, nobs))
    state_intercept = np.reshape(np.arange(k_states * nobs) * 1.0, (k_states, nobs))
    transition = np.reshape(
        np.arange(k_states**2 * nobs) * 1.0, (k_states, k_states, nobs)
    )
    selection = np.reshape(
        np.arange(k_states * k_posdef * nobs) * 1.0, (k_states, k_posdef, nobs)
    )
    state_cov = np.reshape(
        np.arange(k_posdef**2 * nobs) * 1.0, (k_posdef, k_posdef, nobs)
    )

    mod = MLEModel(
        endog,
        k_states=k_states,
        k_posdef=k_posdef,
        obs_intercept=obs_intercept,
        design=design,
        obs_cov=obs_cov,
        state_intercept=state_intercept,
        transition=transition,
        selection=selection,
        state_cov=state_cov,
    )

    assert_allclose(mod["obs_intercept"], obs_intercept)
    assert_allclose(mod["design"], design)
    assert_allclose(mod["obs_cov"], obs_cov)
    assert_allclose(mod["state_intercept"], state_intercept)
    assert_allclose(mod["transition"], transition)
    assert_allclose(mod["selection"], selection)
    assert_allclose(mod["state_cov"], state_cov)


def test_wrapping():
    # Test the wrapping of various Representation / KalmanFilter /
    # KalmanSmoother methods / attributes
    mod, _ = get_dummy_mod(fit=False)

    # Test that we can get the design matrix
    assert_equal(mod["design", 0, 0], 2.0 * np.arange(100))

    # Test that we can set individual elements of the design matrix
    mod["design", 0, 0, :] = 2
    assert_equal(mod.ssm["design", 0, 0, :], 2)
    assert_equal(mod.ssm["design"].shape, (1, 1, 100))

    # Test that we can set the entire design matrix
    mod["design"] = [[3.0]]
    assert_equal(mod.ssm["design", 0, 0], 3.0)
    # (Now it's no longer time-varying, so only 2-dim)
    assert_equal(mod.ssm["design"].shape, (1, 1))

    # Test that we can change the following properties: loglikelihood_burn,
    # initial_variance, tolerance
    assert_equal(mod.loglikelihood_burn, 0)
    mod.loglikelihood_burn = 1
    assert_equal(mod.ssm.loglikelihood_burn, 1)

    assert_equal(mod.tolerance, mod.ssm.tolerance)
    mod.tolerance = 0.123
    assert_equal(mod.ssm.tolerance, 0.123)

    assert_equal(mod.initial_variance, 1e10)
    mod.initial_variance = 1e12
    assert_equal(mod.ssm.initial_variance, 1e12)

    # Test that we can use the following wrappers: initialization,
    # initialize_known, initialize_stationary, initialize_approximate_diffuse

    # Initialization starts off as none
    assert_equal(isinstance(mod.initialization, object), True)

    # Since the SARIMAX model may be fully stationary or may have diffuse
    # elements, it uses a custom initialization by default, but it can be
    # overridden by users
    mod.initialize_default()  # no-op here

    mod.initialize_approximate_diffuse(1e5)
    assert_equal(mod.initialization.initialization_type, "approximate_diffuse")
    assert_equal(mod.initialization.approximate_diffuse_variance, 1e5)

    mod.initialize_known([5.0], [[40]])
    assert_equal(mod.initialization.initialization_type, "known")
    assert_equal(mod.initialization.constant, [5.0])
    assert_equal(mod.initialization.stationary_cov, [[40]])

    mod.initialize_stationary()
    assert_equal(mod.initialization.initialization_type, "stationary")

    # Test that we can use the following wrapper methods: set_filter_method,
    # set_stability_method, set_conserve_memory, set_smoother_output

    # The defaults are as follows:
    assert_equal(mod.ssm.filter_method, kalman_filter.FILTER_CONVENTIONAL)
    assert_equal(mod.ssm.stability_method, kalman_filter.STABILITY_FORCE_SYMMETRY)
    assert_equal(mod.ssm.conserve_memory, kalman_filter.MEMORY_STORE_ALL)
    assert_equal(mod.ssm.smoother_output, kalman_smoother.SMOOTHER_ALL)

    # Now, create the Cython filter object and assert that they have
    # transferred correctly
    mod.ssm._initialize_filter()
    kf = mod.ssm._kalman_filter
    assert_equal(kf.filter_method, kalman_filter.FILTER_CONVENTIONAL)
    assert_equal(kf.stability_method, kalman_filter.STABILITY_FORCE_SYMMETRY)
    assert_equal(kf.conserve_memory, kalman_filter.MEMORY_STORE_ALL)
    # (the smoother object is so far not in Cython, so there is no
    # transferring)

    # Change the attributes in the model class
    mod.set_filter_method(100)
    mod.set_stability_method(101)
    mod.set_conserve_memory(102)
    mod.set_smoother_output(103)

    # Assert that the changes have occurred in the ssm class
    assert_equal(mod.ssm.filter_method, 100)
    assert_equal(mod.ssm.stability_method, 101)
    assert_equal(mod.ssm.conserve_memory, 102)
    assert_equal(mod.ssm.smoother_output, 103)

    # Assert that the changes have *not yet* occurred in the filter object
    assert_equal(kf.filter_method, kalman_filter.FILTER_CONVENTIONAL)
    assert_equal(kf.stability_method, kalman_filter.STABILITY_FORCE_SYMMETRY)
    assert_equal(kf.conserve_memory, kalman_filter.MEMORY_STORE_ALL)

    # Now, test the setting of the other two methods by resetting the
    # filter method to a valid value
    mod.set_filter_method(1)
    mod.ssm._initialize_filter()
    # Retrieve the new kalman filter object (a new object had to be created
    # due to the changing filter method)
    kf = mod.ssm._kalman_filter

    assert_equal(kf.filter_method, 1)
    assert_equal(kf.stability_method, 101)
    assert_equal(kf.conserve_memory, 102)


def test_fit_misc():
    true = results_sarimax.wpi1_stationary
    endog = np.diff(true["data"])[1:]

    mod = sarimax.SARIMAX(endog, order=(1, 0, 1), trend="c")

    # Test optim_hessian={'opg','oim','approx'}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res1 = mod.fit(
            method="ncg", disp=0, optim_hessian="opg", optim_complex_step=False
        )
        res2 = mod.fit(
            method="ncg", disp=0, optim_hessian="oim", optim_complex_step=False
        )
    # Check that the Hessians broadly result in the same optimum
    assert_allclose(res1.llf, res2.llf, rtol=1e-2)

    # Test return_params=True
    mod, _ = get_dummy_mod(fit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_params = mod.fit(disp=-1, return_params=True)

    # 5 digits necessary to accommodate 32-bit numpy/scipy with OpenBLAS 0.2.18
    assert_almost_equal(res_params, [0, 0], 5)


def _newton_model(name):
    data = macrodata.load_pandas().data
    if name == "sarimax_lgdp":
        # With the approximate diffuse initialization, the loglikelihood of this
        # model has a rounding error that is large compared with its curvature
        # near the optimum. Newton reaches the optimum but cannot meet its
        # tolerance then, so that the exact diffuse initialization is used.
        endog = np.log(data["realgdp"].to_numpy())
        return sarimax.SARIMAX(endog, order=(1, 1, 0), use_exact_diffuse=True)
    elif name.startswith("sarimax_infl"):
        return sarimax.SARIMAX(data["infl"].to_numpy(), order=(1, 0, 1), trend="c")
    else:  # "uc_infl"
        return structural.UnobservedComponents(data["infl"].to_numpy(), "llevel")


@pytest.mark.parametrize(
    "name, optim_hessian",
    [
        ("sarimax_lgdp", None),
        ("sarimax_infl", None),
        ("sarimax_infl", "oim"),
        ("sarimax_infl", "opg"),
        ("sarimax_infl_fixed", None),
        ("uc_infl", None),
    ],
)
def test_fit_newton_optimum(name, optim_hessian):
    # Newton evaluated the score and the Hessian without the flags of `fit`,
    # that is, as functions of the constrained parameters at the values of the
    # unconstrained ones, and it stopped before the optimum or failed. L-BFGS
    # with the complex-step score, which does not use the Hessian, is the
    # reference. Its default tolerances bound the agreement of the parameters
    # at about 1e-5.
    mod = _newton_model(name)
    fit_kwargs = {"disp": 0}
    if name.endswith("_fixed"):
        res = mod.fit_constrained(
            {"ar.L1": 0.6}, method="newton", optim_hessian=optim_hessian, **fit_kwargs
        )
        desired = mod.fit_constrained(
            {"ar.L1": 0.6}, method="lbfgs", optim_score="approx", **fit_kwargs
        )
    else:
        res = mod.fit(method="newton", optim_hessian=optim_hessian, **fit_kwargs)
        desired = mod.fit(method="lbfgs", optim_score="approx", **fit_kwargs)
    assert res.mle_retvals["converged"]
    assert desired.mle_retvals["converged"]
    assert_allclose(res.llf, desired.llf, rtol=1e-8)
    assert_allclose(res.params, desired.params, rtol=1e-4)


def test_fit_newton_derivatives_at_start():
    # Newton minimizes -loglike(params, transformed=False) / nobs over the
    # unconstrained parameters. Its score and Hessian at the start are the
    # derivatives of that objective, here computed by finite differences with
    # steps that are far above the rounding error of the loglikelihood.
    mod = _newton_model("sarimax_lgdp")
    start = mod.untransform_params(mod.start_params)
    res = mod.fit(method="newton", maxiter=0, disp=0, warn_convergence=False)
    assert_allclose(res.mle_settings["start_params"], start, rtol=1e-12)

    def objective(params):
        return -mod.loglike(params, transformed=False) / mod.nobs

    score = approx_fprime(start, objective, epsilon=1e-5 * np.abs(start), centered=True)
    hessian = approx_hess3(start, objective, epsilon=1e-3 * np.abs(start))
    assert_allclose(res.mle_retvals["score"], score, rtol=1e-6)
    assert_allclose(res.mle_retvals["Hessian"], hessian, rtol=1e-4)


def _ar2_model():
    # The transformation of the AR(2) parameters has a Jacobian that is not
    # diagonal, and the burned observations make nobs - loglikelihood_burn
    # differ from nobs.
    endog = macrodata.load_pandas().data["infl"].to_numpy()
    return sarimax.SARIMAX(endog, order=(2, 0, 1), trend="c", loglikelihood_burn=3)


@pytest.mark.parametrize("average", [True, False])
def test_hessian_untransformed(average):
    # The Hessian of the loglikelihood with respect to the unconstrained
    # parameters, at the start, where the score is not zero, so that the second
    # derivatives of the transformation matter. The finite differences agree to
    # about 1e-6 here.
    mod = _ar2_model()
    params = mod.untransform_params(mod.start_params)

    def loglike(params):
        return mod.loglike(params, transformed=False)

    epsilon = 1e-4 * np.maximum(np.abs(params), 0.1)
    desired = approx_hess3(params, loglike, epsilon=epsilon)
    if average:
        desired /= mod.nobs - mod.loglikelihood_burn
    actual = mod.hessian(params, transformed=False, average=average)
    assert_allclose(actual, desired, rtol=1e-5)


def test_hessian_untransformed_opg():
    # The outer product of the scores with respect to the unconstrained
    # parameters, which are here computed by differentiating loglikeobs
    # through the transformation
    mod = _ar2_model()
    params = mod.untransform_params(mod.start_params)
    score_obs = mod.score_obs(
        params, transformed=False, approx_complex_step=False, approx_centered=True
    )
    desired = -score_obs.T @ score_obs / (mod.nobs - mod.loglikelihood_burn)
    actual = mod.hessian(params, transformed=False, method="opg")
    assert_allclose(actual, desired, rtol=1e-7)


@pytest.mark.parametrize("method", ["approx", "oim", "opg"])
def test_hessian_average(method):
    # The Hessian of the loglikelihood is its average times the number of
    # observations that are not burned
    mod = _ar2_model()
    params = mod.start_params
    average = mod.hessian(params, method=method)
    total = mod.hessian(params, method=method, average=False)
    assert_allclose(total, average * (mod.nobs - mod.loglikelihood_burn), rtol=1e-14)


def test_hessian_positional_flags():
    # The four flags can be passed positionally, as before `average` was added.
    # `average` is keyword-only, so that a fifth positional flag is one too many.
    mod = _ar2_model()
    params = mod.start_params
    desired = mod.hessian(
        params,
        transformed=True,
        method="approx",
        approx_complex_step=None,
        approx_centered=False,
    )
    actual = mod.hessian(params, True, "approx", None, False)
    assert_allclose(actual, desired, rtol=1e-14)
    with pytest.raises(ValueError, match="zip"):
        mod.hessian(params, True, "approx", None, False, False)
    kind = inspect.signature(MLEModel.hessian).parameters["average"].kind
    assert kind is inspect.Parameter.KEYWORD_ONLY


def test_hessian_fit_flags_average():
    # fit passes the same dictionary of flags, with average False, to
    # loglike, score and hessian. Only hessian uses average.
    mod = _newton_model("sarimax_infl")
    res = mod.fit(method="newton", maxiter=0, disp=0, warn_convergence=False)
    (flags,) = res.mle_settings["fargs"]
    assert flags["average"] is False
    params = mod.untransform_params(mod.start_params)
    assert_allclose(
        mod.loglike(params, flags), mod.loglike(params, transformed=False), rtol=1e-14
    )
    assert_allclose(
        mod.score(params, flags), mod.score(params, transformed=False), rtol=1e-14
    )
    desired = mod.hessian(params, transformed=False, average=False)
    assert_allclose(mod.hessian(params, flags), desired, rtol=1e-14)
    # without an entry in the dictionary, the keyword is used
    actual = mod.hessian(params, {"transformed": False}, average=False)
    assert_allclose(actual, desired, rtol=1e-14)


@pytest.mark.parametrize("name", ["mlefit", "mle_retvals", "mle_settings"])
def test_fit_attrs_in_dir(name):
    # GH#9271: mlefit, mle_retvals and mle_settings were attached to the
    # results wrapper instead of the underlying results instance, so they
    # were reachable via attribute access but did not show up in dir(res)
    # (and therefore not in tab-completion).
    _, res = get_dummy_mod()

    # Reachable as before (no regression in attribute access)
    assert hasattr(res, name)
    # Now discoverable via dir() / tab-completion
    assert name in dir(res)
    # Because they now live on the underlying results instance
    assert name in dir(res._results)


@pytest.mark.parametrize("name", ["sarimax", "llevel"])
def test_fit_cg_optimum(name):
    # cg did not pass the flags of `fit` to the loglikelihood and the score,
    # so that it maximized the loglikelihood as a function of the constrained
    # parameters at the values of the unconstrained ones. It stopped far from
    # the optimum, and for the local level model with converged True. The
    # transformations of the parameters of these models have diagonal
    # Jacobians, so that the test does not depend on the chain rule of
    # MLEModel.score with transformed=False. Nelder-Mead, which does not use
    # the score, is the reference.
    infl = macrodata.load_pandas().data["infl"].to_numpy()
    if name == "sarimax":
        mod = sarimax.SARIMAX(infl, order=(1, 0, 1), trend="c")
    else:
        mod = structural.UnobservedComponents(infl, "llevel")
    res = mod.fit(method="cg", disp=0)
    desired = mod.fit(method="nm", maxiter=5000, xtol=1e-8, ftol=1e-10, disp=0)
    assert desired.mle_retvals["converged"]
    assert res.mle_retvals["converged"]
    # The loglikelihood is flat at the optimum, so that it agrees to second
    # order in the difference of the parameters. The parameters agree to the
    # accuracy of the solvers, about 1e-4: cg stops when the gradient of the
    # loglikelihood divided by nobs is below 1e-5. Without the flags, the
    # loglikelihood was lower by 6 and 38.
    assert_allclose(res.llf, desired.llf, rtol=1e-8)
    assert_allclose(res.params, desired.params, rtol=1e-3)


@pytest.mark.smoke
def test_score_misc():
    mod, res = get_dummy_mod()

    # Test that the score function works
    mod.score(res.params)


def test_score_hessian_invalid_method_raises():
    mod, res = get_dummy_mod()

    with pytest.raises(ValueError, match="method"):
        mod.score(res.params, method="invalid")
    with pytest.raises(ValueError, match="method"):
        mod.score_obs(res.params, method="invalid")
    with pytest.raises(ValueError, match="method"):
        mod.hessian(res.params, method="invalid")


def test_info_criteria_invalid_raises():
    mod, res = get_dummy_mod()

    with pytest.raises(ValueError, match="criteria"):
        res.info_criteria("invalid")
    with pytest.raises(ValueError, match="method"):
        res.info_criteria("aic", method="invalid")


def test_get_prediction_invalid_information_set_raises():
    mod, res = get_dummy_mod()

    with pytest.raises(ValueError, match="information_set"):
        res.get_prediction(information_set="invalid")


def test_get_smoothed_decomposition_invalid_raises():
    mod, res = get_dummy_mod()

    with pytest.raises(ValueError, match="decomposition_of"):
        res.get_smoothed_decomposition(decomposition_of="invalid")


def test_from_formula():
    with pytest.raises(NotImplementedError):
        MLEModel.from_formula(1, 2, 3)


def test_score_analytic_ar1():
    # Test the score against the analytic score for an AR(1) model with 2
    # observations
    # Let endog = [1, 0.5], params=[0, 1]
    mod = sarimax.SARIMAX([1, 0.5], order=(1, 0, 0))

    def partial_phi(phi, sigma2):
        return -0.5 * (phi**2 + 2 * phi * sigma2 - 1) / (sigma2 * (1 - phi**2))

    def partial_sigma2(phi, sigma2):
        return -0.5 * (2 * sigma2 + phi - 1.25) / (sigma2**2)

    params = np.r_[0.0, 2]

    # Compute the analytic score
    analytic_score = np.r_[
        partial_phi(params[0], params[1]), partial_sigma2(params[0], params[1])
    ]

    # Check each of the approximations, transformed parameters
    approx_cs = mod.score(params, transformed=True, approx_complex_step=True)
    assert_allclose(approx_cs, analytic_score)

    approx_fd = mod.score(params, transformed=True, approx_complex_step=False)
    assert_allclose(approx_fd, analytic_score, atol=1e-5)

    approx_fd_centered = mod.score(
        params, transformed=True, approx_complex_step=False, approx_centered=True
    )
    assert_allclose(approx_fd, analytic_score, atol=1e-5)

    harvey_cs = mod.score(
        params, transformed=True, method="harvey", approx_complex_step=True
    )
    assert_allclose(harvey_cs, analytic_score)
    harvey_fd = mod.score(
        params, transformed=True, method="harvey", approx_complex_step=False
    )
    assert_allclose(harvey_fd, analytic_score, atol=1e-5)
    harvey_fd_centered = mod.score(
        params,
        transformed=True,
        method="harvey",
        approx_complex_step=False,
        approx_centered=True,
    )
    assert_allclose(harvey_fd_centered, analytic_score, atol=1e-5)

    # Check the approximations for untransformed parameters. The analytic
    # check now comes from chain rule with the analytic derivative of the
    # transformation
    # if L* is the likelihood evaluated at untransformed parameters and
    # L is the likelihood evaluated at transformed parameters, then we have:
    # L*(u) = L(t(u))
    # and then
    # L'*(u) = L'(t(u)) * t'(u)
    def partial_transform_phi(phi):
        return -1.0 / (1 + phi**2) ** (3.0 / 2)

    def partial_transform_sigma2(sigma2):
        return 2.0 * sigma2

    uparams = mod.untransform_params(params)

    analytic_score = np.dot(
        np.diag(
            np.r_[
                partial_transform_phi(uparams[0]), partial_transform_sigma2(uparams[1])
            ]
        ),
        np.r_[partial_phi(params[0], params[1]), partial_sigma2(params[0], params[1])],
    )

    approx_cs = mod.score(uparams, transformed=False, approx_complex_step=True)
    assert_allclose(approx_cs, analytic_score)

    approx_fd = mod.score(uparams, transformed=False, approx_complex_step=False)
    assert_allclose(approx_fd, analytic_score, atol=1e-5)

    approx_fd_centered = mod.score(
        uparams, transformed=False, approx_complex_step=False, approx_centered=True
    )
    assert_allclose(approx_fd_centered, analytic_score, atol=1e-5)

    harvey_cs = mod.score(
        uparams, transformed=False, method="harvey", approx_complex_step=True
    )
    assert_allclose(harvey_cs, analytic_score)
    harvey_fd = mod.score(
        uparams, transformed=False, method="harvey", approx_complex_step=False
    )
    assert_allclose(harvey_fd, analytic_score, atol=1e-5)
    harvey_fd_centered = mod.score(
        uparams,
        transformed=False,
        method="harvey",
        approx_complex_step=False,
        approx_centered=True,
    )
    assert_allclose(harvey_fd_centered, analytic_score, atol=1e-5)

    # Check the Hessian: these approximations are not very good, particularly
    # when phi is close to 0
    params = np.r_[0.5, 1.0]

    def hessian(phi, sigma2):
        hessian = np.zeros((2, 2))
        hessian[0, 0] = (-(phi**2) - 1) / (phi**2 - 1) ** 2
        hessian[1, 0] = hessian[0, 1] = -1 / (2 * sigma2**2)
        hessian[1, 1] = (sigma2 + phi - 1.25) / sigma2**3
        return hessian

    analytic_hessian = hessian(params[0], params[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_allclose(
            mod._hessian_complex_step(params) * 2, analytic_hessian, atol=1e-1
        )
        assert_allclose(
            mod._hessian_finite_difference(params) * 2, analytic_hessian, atol=1e-1
        )


def _score_model(name):
    data = macrodata.load_pandas().data
    infl = data["infl"].to_numpy()
    levels = data[["realgdp", "realcons", "realinv"]].to_numpy()
    growth = 100 * np.diff(np.log(levels), axis=0)
    if name == "sarimax_arma21":
        return sarimax.SARIMAX(infl, order=(2, 0, 1), trend="c")
    elif name == "sarimax_seasonal":
        return sarimax.SARIMAX(infl, order=(1, 0, 0), seasonal_order=(2, 0, 0, 4))
    elif name == "var1":
        return varmax.VARMAX(growth[:, :2], order=(1, 0), trend="c")
    elif name == "var2":
        return varmax.VARMAX(growth[:, :2], order=(2, 0), trend="c")
    elif name == "dynamic_factor":
        return dynamic_factor.DynamicFactor(growth, k_factors=1, factor_order=2)
    else:  # "sarimax_ar1"
        return sarimax.SARIMAX(infl, order=(1, 0, 0))


def _centered_gradient(func, x):
    # Centered finite differences, with the step that balances the truncation
    # and the rounding errors
    step = np.finfo(float).eps ** (1 / 3) * np.maximum(np.abs(x), 0.1)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = step[i]
        grad[i] = (func(x + dx) - func(x - dx)) / (2 * step[i])
    return grad


_score_methods = [
    pytest.param("approx", True, False, id="approx-cs"),
    pytest.param("approx", False, False, id="approx-fd"),
    pytest.param("approx", False, True, id="approx-fd-centered"),
    pytest.param("harvey", True, False, id="harvey-cs"),
    pytest.param("harvey", False, False, id="harvey-fd"),
]


@pytest.mark.parametrize("method, approx_complex_step, approx_centered", _score_methods)
@pytest.mark.parametrize(
    "name",
    ["sarimax_arma21", "sarimax_seasonal", "var2", "dynamic_factor", "sarimax_ar1"],
)
def test_score_untransformed(name, method, approx_complex_step, approx_centered):
    # The score with respect to the unconstrained parameters u is J' g, where g
    # is the score with respect to the constrained parameters and J is the
    # Jacobian of the transformation. It was J g, which is wrong unless J is
    # symmetric. In the autoregressive parts of the first four models it is
    # not; in the AR(1) it is diagonal. The reference is the gradient of the
    # loglikelihood as a function of u, at a point away from the optimum.
    mod = _score_model(name)
    uparams = mod.untransform_params(mod.start_params) + 0.1
    actual = mod.score(
        uparams,
        transformed=False,
        method=method,
        approx_complex_step=approx_complex_step,
        approx_centered=approx_centered,
    )
    desired = _centered_gradient(
        lambda params: mod.loglike(params, transformed=False), uparams
    )
    # The errors of numerical derivatives scale with the largest element of
    # the gradient, not with each element
    assert_allclose(actual, desired, rtol=1e-5, atol=1e-4 * np.max(np.abs(desired)))


@pytest.mark.parametrize("method, approx_complex_step, approx_centered", _score_methods)
@pytest.mark.parametrize(
    "name, includes_fixed",
    [("sarimax_arma21", False), ("var1", False), ("var1", True)],
)
def test_score_untransformed_fixed_params(
    name, includes_fixed, method, approx_complex_step, approx_centered
):
    # Fixed parameters are set after the transformation, so that they do not
    # depend on the free parameters, unless `includes_fixed` is True, when all
    # of the parameters passed are transformed. In the VAR(1), the transformed
    # autoregressive coefficients depend on the error covariance, so that the
    # rows of J of the fixed coefficients must be left out.
    mod = _score_model(name)
    if name == "sarimax_arma21":
        fixed = ["ma.L1"]
    else:
        fixed = [param for param in mod.param_names if param.startswith("L1.")]
    start_params = dict(zip(mod.param_names, mod.start_params, strict=True))
    uparams = mod.untransform_params(mod.start_params) + 0.1
    if not includes_fixed:
        free = np.array(
            [i for i, param in enumerate(mod.param_names) if param not in fixed],
            dtype=int,
        )
        uparams = uparams[free]

    with mod.fix_params({param: start_params[param] for param in fixed}):
        actual = mod.score(
            uparams,
            transformed=False,
            includes_fixed=includes_fixed,
            method=method,
            approx_complex_step=approx_complex_step,
            approx_centered=approx_centered,
        )
        desired = _centered_gradient(
            lambda params: mod.loglike(
                params, transformed=False, includes_fixed=includes_fixed
            ),
            uparams,
        )
    # The errors of numerical derivatives scale with the largest element of
    # the gradient, not with each element
    assert_allclose(actual, desired, rtol=1e-5, atol=1e-4 * np.max(np.abs(desired)))


@pytest.mark.parametrize("method", ["bfgs", "lbfgs"])
def test_fit_untransformed_score_optimum(method):
    # With the wrong score, BFGS and L-BFGS stopped far from the optimum.
    # Nelder-Mead, which does not use the score, is the reference, with tight
    # tolerances. BFGS and L-BFGS stop when the score is below their default
    # tolerances, which leaves errors of about 1e-4 in the parameters. The
    # loglikelihood is flat at the optimum, so that its error is quadratic in
    # those and much smaller.
    mod = _score_model("sarimax_arma21")
    desired = mod.fit(
        method="nm", maxiter=20000, maxfun=40000, xtol=1e-10, ftol=1e-12, disp=0
    )
    res = mod.fit(method=method, optim_score="approx", disp=0)
    assert desired.mle_retvals["converged"]
    assert res.mle_retvals["converged"]
    assert_allclose(res.llf, desired.llf, rtol=1e-8)
    assert_allclose(res.params, desired.params, rtol=1e-3)


def test_cov_params():
    mod, res = get_dummy_mod()

    # Smoke test for each of the covariance types
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(res.params, disp=-1, cov_type="none")
        assert_equal(res.cov_kwds["description"], "Covariance matrix not calculated.")

        res = mod.fit(res.params, disp=-1, cov_type="approx")
        assert_equal(res.cov_type, "approx")
        assert_equal(
            res.cov_kwds["description"],
            "Covariance matrix calculated using numerical (complex-step) "
            "differentiation.",
        )

        res = mod.fit(res.params, disp=-1, cov_type="oim")
        assert_equal(res.cov_type, "oim")
        assert_equal(
            res.cov_kwds["description"],
            "Covariance matrix calculated using the observed information "
            "matrix (complex-step) described in Harvey (1989).",
        )

        res = mod.fit(res.params, disp=-1, cov_type="opg")
        assert_equal(res.cov_type, "opg")
        assert_equal(
            res.cov_kwds["description"],
            "Covariance matrix calculated using the outer product of "
            "gradients (complex-step).",
        )

        res = mod.fit(res.params, disp=-1, cov_type="robust")
        assert_equal(res.cov_type, "robust")
        assert_equal(
            res.cov_kwds["description"],
            "Quasi-maximum likelihood covariance matrix used for robustness "
            "to some misspecifications; calculated using the observed "
            "information matrix (complex-step) described in Harvey (1989).",
        )

        res = mod.fit(res.params, disp=-1, cov_type="robust_oim")
        assert_equal(res.cov_type, "robust_oim")
        assert_equal(
            res.cov_kwds["description"],
            "Quasi-maximum likelihood covariance matrix used for robustness "
            "to some misspecifications; calculated using the observed "
            "information matrix (complex-step) described in Harvey (1989).",
        )

        res = mod.fit(res.params, disp=-1, cov_type="robust_approx")
        assert_equal(res.cov_type, "robust_approx")
        assert_equal(
            res.cov_kwds["description"],
            "Quasi-maximum likelihood covariance matrix used for robustness "
            "to some misspecifications; calculated using numerical "
            "(complex-step) differentiation.",
        )

        with pytest.raises(NotImplementedError):
            mod.fit(res.params, disp=-1, cov_type="invalid_cov_type")


def test_transform():
    # The transforms in MLEModel are noops
    mod = MLEModel([1, 2], **kwargs)

    # Test direct transform, untransform
    assert_allclose(mod.transform_params([2, 3]), [2, 3])
    assert_allclose(mod.untransform_params([2, 3]), [2, 3])

    # Smoke test for transformation in `filter`, `update`, `loglike`,
    # `loglikeobs`
    mod.filter([], transformed=False)
    mod.update([], transformed=False)
    mod.loglike([], transformed=False)
    mod.loglikeobs([], transformed=False)

    # Note that mod is an SARIMAX instance, and the two parameters are
    # variances
    mod, _ = get_dummy_mod(fit=False)

    # Test direct transform, untransform
    assert_allclose(mod.transform_params([2, 3]), [4, 9])
    assert_allclose(mod.untransform_params([4, 9]), [2, 3])

    # Test transformation in `filter`
    res = mod.filter([2, 3], transformed=True)
    assert_allclose(res.params, [2, 3])

    res = mod.filter([2, 3], transformed=False)
    assert_allclose(res.params, [4, 9])


def test_filter():
    endog = np.array([1.0, 2.0])
    mod = MLEModel(endog, **kwargs)

    # Test return of ssm object
    res = mod.filter([], return_ssm=True)
    assert_equal(isinstance(res, kalman_filter.FilterResults), True)

    # Test return of full results object
    res = mod.filter([])
    assert_equal(isinstance(res, MLEResultsWrapper), True)
    assert_equal(res.cov_type, "opg")

    # Test return of full results object, specific covariance type
    res = mod.filter([], cov_type="oim")
    assert_equal(isinstance(res, MLEResultsWrapper), True)
    assert_equal(res.cov_type, "oim")


def test_params():
    mod = MLEModel([1, 2], **kwargs)

    # By default start_params raises NotImplementedError
    with pytest.raises(NotImplementedError):
        _ = mod.start_params
    # But param names are by default an empty array
    assert_equal(mod.param_names, [])

    # We can set them in the object if we want
    mod._start_params = [1]
    mod._param_names = ["a"]

    assert_equal(mod.start_params, [1])
    assert_equal(mod.param_names, ["a"])


def check_results(pandas):
    mod, res = get_dummy_mod(pandas=pandas)

    # Test fitted values
    assert_almost_equal(res.fittedvalues[2:], mod.endog[2:].squeeze())

    # Test residuals
    assert_almost_equal(res.resid[2:], np.zeros(mod.nobs - 2))

    # Test loglikelihood_burn
    assert_equal(res.loglikelihood_burn, 0)


def test_results():
    check_results(pandas=False)
    check_results(pandas=True)


def test_predict():
    dates = pd.date_range(start="1980-01-01", end="1981-01-01", freq="YS")
    endog = pd.Series([1, 2], index=dates)
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([])

    # Test that predict with start=None, end=None does prediction with full
    # dataset
    predict = res.predict()
    assert_equal(predict.shape, (mod.nobs,))
    assert_allclose(res.get_prediction().predicted_mean, predict)

    # Test a string value to the dynamic option
    assert_allclose(res.predict(dynamic="1981-01-01"), res.predict())

    # Test an invalid date string value to the dynamic option
    # with pytest.raises(ValueError, res.predict, dynamic='1982-01-01')

    # Test for passing a string to predict when dates are not set
    mod = MLEModel([1, 2], **kwargs)
    res = mod.filter([])
    with pytest.raises(KeyError):
        res.predict(dynamic="string")


def test_forecast():
    # Numpy
    mod = MLEModel([1, 2], **kwargs)
    res = mod.filter([])
    forecast = res.forecast(steps=10)
    assert_allclose(forecast, np.ones((10,)) * 2)
    assert_allclose(res.get_forecast(steps=10).predicted_mean, forecast)

    # Pandas
    index = pd.date_range("1960-01-01", periods=2, freq="MS")
    mod = MLEModel(pd.Series([1, 2], index=index), **kwargs)
    res = mod.filter([])
    assert_allclose(res.forecast(steps=10), np.ones((10,)) * 2)
    assert_allclose(res.forecast(steps="1960-12-01"), np.ones((10,)) * 2)
    assert_allclose(res.get_forecast(steps=10).predicted_mean, np.ones((10,)) * 2)


def test_summary():
    dates = pd.date_range(start="1980-01-01", end="1984-01-01", freq="YS")
    endog = pd.Series([1, 2, 3, 4, 5], index=dates)
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([])

    # Get the summary
    txt = str(res.summary())

    # Test res.summary when the model has dates
    assert_equal(re.search(r"Sample:\s+01-01-1980", txt) is not None, True)
    assert_equal(re.search(r"\s+- 01-01-1984", txt) is not None, True)

    # Test res.summary when `model_name` was not provided
    assert_equal(re.search(r"Model:\s+MLEModel", txt) is not None, True)

    # Smoke test that summary still works when diagnostic tests fail
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.filter_results._standardized_forecasts_error[:] = np.nan
        res.summary()
        res.filter_results._standardized_forecasts_error = 1
        res.summary()
        res.filter_results._standardized_forecasts_error = "a"
        res.summary()


def test_summary_after_remove_data():
    # summary() must still work after remove_data() has been called
    dates = pd.date_range(start="1980-01-01", end="1984-01-01", freq="YS")
    endog = pd.Series([1, 2, 3, 4, 5], index=dates)
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([])

    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)


def check_endog(endog, nobs=2, k_endog=1, **kwargs):
    # create the model
    mod = MLEModel(endog, **kwargs)
    # the data directly available in the model is the statsmodels version of
    # the data; it should be 2-dim, C-contiguous, long-shaped:
    # (nobs, k_endog) == (2, 1)
    assert_equal(mod.endog.ndim, 2)
    assert_equal(mod.endog.flags["C_CONTIGUOUS"], True)
    assert_equal(mod.endog.shape, (nobs, k_endog))
    # the data in the `ssm` object is the state space version of the data; it
    # should be 2-dim, F-contiguous, wide-shaped (k_endog, nobs) == (1, 2)
    # and it should share data with mod.endog
    assert_equal(mod.ssm.endog.ndim, 2)
    assert_equal(mod.ssm.endog.flags["F_CONTIGUOUS"], True)
    assert_equal(mod.ssm.endog.shape, (k_endog, nobs))
    if mod.ssm.endog.base.ndim > 1:
        assert mod.ssm.endog.base is mod.endog or not mod.endog.flags.writeable
    else:
        # Added to handle changes in numpy where shape cannot be assigned to
        # Even though array is no copy, the reshaped array is a new object
        np.testing.assert_equal(mod.ssm.endog.base, mod.endog.ravel())
    return mod


def test_basic_endog():
    # Test various types of basic python endog inputs (e.g., lists, scalars...)

    # Check cannot call with non-array_like
    # fails due to checks in statsmodels base classes
    with pytest.raises(ValueError):
        MLEModel(endog=1, k_states=1)
    with pytest.raises(ValueError):
        MLEModel(endog="a", k_states=1)
    with pytest.raises(ValueError):
        MLEModel(endog=True, k_states=1)

    # Check behavior with different types
    mod = MLEModel([1], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])

    mod = MLEModel([1.0], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])

    mod = MLEModel([True], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])

    mod = MLEModel(["a"], **kwargs)
    # raises error due to inability coerce string to numeric
    with pytest.raises(ValueError):
        mod.filter([])

    # Check that a different iterable tpyes give the expected result
    endog = [1.0, 2.0]
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    endog = [[1.0], [2.0]]
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    endog = (1.0, 2.0)
    mod = check_endog(endog, **kwargs)
    mod.filter([])


def test_numpy_endog():
    # Test various types of numpy endog inputs

    # Check behavior of the link maintained between passed `endog` and
    # `mod.endog` arrays
    endog = np.array([1.0, 2.0])
    mod = MLEModel(endog, **kwargs)
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.base is not endog, True)
    endog[0] = 2
    # there is no link to mod.endog
    assert_equal(mod.endog, np.r_[1, 2].reshape(2, 1))
    # there remains a link to mod.data.orig_endog
    assert_equal(mod.data.orig_endog, endog)

    # Check behavior with different memory layouts / shapes

    # Example  (failure): 0-dim array
    endog = np.array(1.0)
    # raises error due to len(endog) failing in statsmodels base classes
    with pytest.raises(TypeError):
        check_endog(endog, **kwargs)

    # Example : 1-dim array, both C- and F-contiguous, length 2
    endog = np.array([1.0, 2.0])
    assert_equal(endog.ndim, 1)
    assert_equal(endog.flags["C_CONTIGUOUS"], True)
    assert_equal(endog.flags["F_CONTIGUOUS"], True)
    assert_equal(endog.shape, (2,))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, C-contiguous, long-shaped: (nobs, k_endog)
    endog = np.array([1.0, 2.0]).reshape(2, 1)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags["C_CONTIGUOUS"], True)
    # On newer numpy (>= 0.10), this array is (rightly) both C and F contiguous
    # assert_equal(endog.flags['F_CONTIGUOUS'], False)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, C-contiguous, wide-shaped: (k_endog, nobs)
    endog = np.array([1.0, 2.0]).reshape(1, 2)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags["C_CONTIGUOUS"], True)
    # On newer numpy (>= 0.10), this array is (rightly) both C and F contiguous
    # assert_equal(endog.flags['F_CONTIGUOUS'], False)
    assert_equal(endog.shape, (1, 2))
    # raises error because arrays are always interpreted as
    # (nobs, k_endog), which means that k_endog=2 is incompatibile with shape
    # of design matrix (1, 1)
    with pytest.raises(ValueError):
        check_endog(endog, **kwargs)

    # Example : 2-dim array, F-contiguous, long-shaped (nobs, k_endog)
    endog = np.array([1.0, 2.0]).reshape(1, 2).transpose()
    assert_equal(endog.ndim, 2)
    # On newer numpy (>= 0.10), this array is (rightly) both C and F contiguous
    # assert_equal(endog.flags['C_CONTIGUOUS'], False)
    assert_equal(endog.flags["F_CONTIGUOUS"], True)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, F-contiguous, wide-shaped (k_endog, nobs)
    endog = np.array([1.0, 2.0]).reshape(2, 1).transpose()
    assert_equal(endog.ndim, 2)
    # On newer numpy (>= 0.10), this array is (rightly) both C and F contiguous
    # assert_equal(endog.flags['C_CONTIGUOUS'], False)
    assert_equal(endog.flags["F_CONTIGUOUS"], True)
    assert_equal(endog.shape, (1, 2))
    # raises error because arrays are always interpreted as
    # (nobs, k_endog), which means that k_endog=2 is incompatibile with shape
    # of design matrix (1, 1)
    with pytest.raises(ValueError):
        check_endog(endog, **kwargs)

    # Example  (failure): 3-dim array
    endog = np.array([1.0, 2.0]).reshape(2, 1, 1)
    # raises error due to direct ndim check in statsmodels base classes
    with pytest.raises(ValueError):
        check_endog(endog, **kwargs)

    # Example : np.array with 2 columns
    # Update kwargs for k_endog=2
    kwargs2 = {
        "k_states": 1,
        "design": [[1], [0.0]],
        "obs_cov": [[1, 0], [0, 1]],
        "transition": [[1]],
        "selection": [[1]],
        "state_cov": [[1]],
        "initialization": "approximate_diffuse",
    }
    endog = np.array([[1.0, 2.0], [3.0, 4.0]])
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])


def test_pandas_endog():
    # Test various types of pandas endog inputs (e.g., TimeSeries, etc.)

    # Example (failure): pandas.Series, no dates
    endog = pd.Series([1.0, 2.0])
    # raises error due to no dates
    warnings.simplefilter("always")
    # with pytest.raises(ValueError, check_endog, endog, **kwargs)

    # Example : pandas.Series
    dates = pd.date_range(start="1980-01-01", end="1981-01-01", freq="YS")
    endog = pd.Series([1.0, 2.0], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : pandas.Series, string datatype
    endog = pd.Series(["a", "b"], index=dates)
    # raises error due to direct type casting check in statsmodels base classes
    with pytest.raises(ValueError):
        check_endog(endog, **kwargs)

    # Example : pandas.Series
    endog = pd.Series([1.0, 2.0], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : pandas.DataFrame with 1 column
    endog = pd.DataFrame({"a": [1.0, 2.0]}, index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example (failure): pandas.DataFrame with 2 columns
    endog = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=dates)
    # raises error because 2-columns means k_endog=2, but the design matrix
    # set in **kwargs is shaped (1, 1)
    with pytest.raises(ValueError):
        check_endog(endog, **kwargs)

    # Check behavior of the link maintained between passed `endog` and
    # `mod.endog` arrays
    endog = pd.DataFrame({"a": [1.0, 2.0]}, index=dates)
    mod = check_endog(endog, **kwargs)
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.values.base is not endog, True)
    endog.iloc[0, 0] = 2
    # there is no link to mod.endog
    assert_equal(mod.endog, np.r_[1, 2].reshape(2, 1))
    # there remains a link to mod.data.orig_endog
    assert_allclose(mod.data.orig_endog, endog)

    # Example : pandas.DataFrame with 2 columns
    # Update kwargs for k_endog=2
    kwargs2 = {
        "k_states": 1,
        "design": [[1], [0.0]],
        "obs_cov": [[1, 0], [0, 1]],
        "transition": [[1]],
        "selection": [[1]],
        "state_cov": [[1]],
        "initialization": "approximate_diffuse",
    }
    endog = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=dates)
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])


def test_diagnostics():
    mod, res = get_dummy_mod()

    # Override the standardized forecasts errors to get more reasonable values
    # for the tests to run (not necessary, but prevents some annoying warnings)
    rs = np.random.RandomState(9991615)
    shape = res.filter_results._standardized_forecasts_error.shape
    res.filter_results._standardized_forecasts_error = rs.normal(size=shape)

    # Make sure method=None selects the appropriate test
    actual = res.test_normality(method=None)
    desired = res.test_normality(method="jarquebera")
    assert_allclose(actual, desired)

    with pytest.raises(ValueError, match="method"):
        res.test_normality(method="invalid")

    actual = res.test_heteroskedasticity(method=None)
    desired = res.test_heteroskedasticity(method="breakvar")
    assert_allclose(actual, desired)

    with pytest.raises(ValueError):
        res.test_heteroskedasticity(method=None, alternative="invalid")
    with pytest.raises(ValueError, match="method"):
        res.test_heteroskedasticity(method="invalid")

    actual = res.test_serial_correlation(method=None)
    desired = res.test_serial_correlation(method="ljungbox")
    assert_allclose(actual, desired)

    with pytest.raises(ValueError, match="method"):
        res.test_serial_correlation(method="invalid")

    # Smoke tests for other options
    res.test_heteroskedasticity(method=None, alternative="decreasing", use_f=False)
    res.test_serial_correlation(method="boxpierce")


def test_small_sample_serial_correlation_test():
    # Test the Ljung Box serial correlation test for small samples with df
    # adjustment using the Nile dataset. Ljung-Box statistic and p-value
    # are compared to R's Arima() and checkresiduals() functions in forecast
    # package:
    # library(forecast)
    # fit <- Arima(y, order=c(1,0,1), include.constant=FALSE)
    # checkresiduals(fit, lag=10)
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range("1871-01-01", "1970-01-01", freq="YS")
    mod = SARIMAX(
        endog=niledata["volume"], order=(1, 0, 1), trend="n", freq=niledata.index.freq
    )
    res = mod.fit()

    actual = res.test_serial_correlation(method="ljungbox", lags=10, df_adjust=True)[
        0, :, -1
    ]
    assert_allclose(actual, [14.116, 0.0788], atol=1e-3)


def test_diagnostics_nile_eviews():
    # Test the diagnostic tests using the Nile dataset. Results are from
    # "Fitting State Space Models with EViews" (Van den Bossche 2011,
    # Journal of Statistical Software).
    # For parameter values, see Figure 2
    # For Ljung-Box and Jarque-Bera statistics and p-values, see Figure 5
    # The Heteroskedasticity statistic is not provided in this paper.
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range("1871-01-01", "1970-01-01", freq="YS")

    mod = MLEModel(
        niledata["volume"],
        k_states=1,
        initialization="approximate_diffuse",
        initial_variance=1e15,
        loglikelihood_burn=1,
    )
    mod.ssm["design", 0, 0] = 1
    mod.ssm["obs_cov", 0, 0] = np.exp(9.600350)
    mod.ssm["transition", 0, 0] = 1
    mod.ssm["selection", 0, 0] = 1
    mod.ssm["state_cov", 0, 0] = np.exp(7.348705)
    res = mod.filter([])

    # Test Ljung-Box
    # Note: only 3 digits provided in the reference paper
    actual = res.test_serial_correlation(method="ljungbox", lags=10)[0, :, -1]
    assert_allclose(actual, [13.117, 0.217], atol=1e-3)

    # Test Jarque-Bera
    actual = res.test_normality(method="jarquebera")[0, :2]
    assert_allclose(actual, [0.041686, 0.979373], atol=1e-5)


def test_diagnostics_nile_durbinkoopman():
    # Test the diagnostic tests using the Nile dataset. Results are from
    # Durbin and Koopman (2012); parameter values reported on page 37; test
    # statistics on page 40
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range("1871-01-01", "1970-01-01", freq="YS")

    mod = MLEModel(
        niledata["volume"],
        k_states=1,
        initialization="approximate_diffuse",
        initial_variance=1e15,
        loglikelihood_burn=1,
    )
    mod.ssm["design", 0, 0] = 1
    mod.ssm["obs_cov", 0, 0] = 15099.0
    mod.ssm["transition", 0, 0] = 1
    mod.ssm["selection", 0, 0] = 1
    mod.ssm["state_cov", 0, 0] = 1469.1
    res = mod.filter([])

    # Test Ljung-Box
    # Note: only 3 digits provided in the reference paper
    actual = res.test_serial_correlation(method="ljungbox", lags=9)[0, 0, -1]
    assert_allclose(actual, [8.84], atol=1e-2)

    # Test Jarque-Bera
    # Note: The book reports 0.09 for Kurtosis, because it is reporting the
    # statistic less the mean of the Kurtosis distribution (which is 3).
    norm = res.test_normality(method="jarquebera")[0]
    actual = [norm[0], norm[2], norm[3]]
    assert_allclose(actual, [0.05, -0.03, 3.09], atol=1e-2)

    # Test Heteroskedasticity
    # Note: only 2 digits provided in the book
    actual = res.test_heteroskedasticity(method="breakvar")[0, 0]
    assert_allclose(actual, [0.61], atol=1e-2)


@pytest.mark.smoke
def test_prediction_results():
    # Just smoke tests for the PredictionResults class, which is copied from
    # elsewhere in statsmodels

    mod, res = get_dummy_mod()
    predict = res.get_prediction()
    predict.summary_frame()


def test_lutkepohl_information_criteria():
    # Setup dataset, use Lutkepohl data
    dta = pd.DataFrame(
        results_var_misc.lutkepohl_data,
        columns=["inv", "inc", "consump"],
        index=pd.date_range("1960-01-01", "1982-10-01", freq="QS"),
    )

    dta["dln_inv"] = np.log(dta["inv"]).diff()
    dta["dln_inc"] = np.log(dta["inc"]).diff()
    dta["dln_consump"] = np.log(dta["consump"]).diff()

    endog = dta.loc["1960-04-01":"1978-10-01", ["dln_inv", "dln_inc", "dln_consump"]]

    # AR model - SARIMAX
    # (use loglikelihood_burn=1 to mimic conditional MLE used by Stata's var
    # command).
    true = results_var_misc.lutkepohl_ar1_lustats
    mod = sarimax.SARIMAX(
        endog["dln_inv"], order=(1, 0, 0), trend="c", loglikelihood_burn=1
    )
    res = mod.filter(true["params"])
    assert_allclose(res.llf, true["loglike"])
    # Test the Lutkepohl ICs
    # Note: for the Lutkepohl ICs, Stata only counts the AR coefficients as
    # estimated parameters for the purposes of information criteria, whereas we
    # count all parameters including scale and constant, so we need to adjust
    # for that
    aic = res.info_criteria("aic", method="lutkepohl") - 2 * 2 / res.nobs_effective
    bic = (
        res.info_criteria("bic", method="lutkepohl")
        - 2 * np.log(res.nobs_effective) / res.nobs_effective
    )
    hqic = (
        res.info_criteria("hqic", method="lutkepohl")
        - 2 * 2 * np.log(np.log(res.nobs_effective)) / res.nobs_effective
    )
    assert_allclose(aic, true["aic"])
    assert_allclose(bic, true["bic"])
    assert_allclose(hqic, true["hqic"])

    # Test the non-Lutkepohl ICs
    # Note: for the non-Lutkepohl ICs, Stata does not count the scale as an
    # estimated parameter, but does count the constant term, for the
    # purposes of information criteria, whereas we count both, so we need to
    # adjust for that
    true = results_var_misc.lutkepohl_ar1
    aic = res.aic - 2
    bic = res.bic - np.log(res.nobs_effective)
    assert_allclose(aic, true["estat_aic"])
    assert_allclose(bic, true["estat_bic"])
    aic = res.info_criteria("aic") - 2
    bic = res.info_criteria("bic") - np.log(res.nobs_effective)
    assert_allclose(aic, true["estat_aic"])
    assert_allclose(bic, true["estat_bic"])

    # Note: could also test the "dfk" (degree of freedom corrections), but not
    # really necessary since they just rescale things a bit

    # VAR model - VARMAX
    # (use loglikelihood_burn=1 to mimic conditional MLE used by Stata's var
    # command).
    true = results_var_misc.lutkepohl_var1_lustats
    mod = varmax.VARMAX(
        endog,
        order=(1, 0),
        trend="n",
        error_cov_type="unstructured",
        loglikelihood_burn=1,
    )
    res = mod.filter(true["params"])
    assert_allclose(res.llf, true["loglike"])

    # Test the Lutkepohl ICs
    # Note: for the Lutkepohl ICs, Stata only counts the AR coefficients as
    # estimated parameters for the purposes of information criteria, whereas we
    # count all parameters including the elements of the covariance matrix, so
    # we need to adjust for that
    aic = res.info_criteria("aic", method="lutkepohl") - 2 * 6 / res.nobs_effective
    bic = (
        res.info_criteria("bic", method="lutkepohl")
        - 6 * np.log(res.nobs_effective) / res.nobs_effective
    )
    hqic = (
        res.info_criteria("hqic", method="lutkepohl")
        - 2 * 6 * np.log(np.log(res.nobs_effective)) / res.nobs_effective
    )
    assert_allclose(aic, true["aic"])
    assert_allclose(bic, true["bic"])
    assert_allclose(hqic, true["hqic"])

    # Test the non-Lutkepohl ICs
    # Note: for the non-Lutkepohl ICs, Stata does not count the elements of the
    # covariance matrix as estimated parameters for the purposes of information
    # criteria, whereas we count both, so we need to adjust for that
    true = results_var_misc.lutkepohl_var1
    aic = res.aic - 2 * 6
    bic = res.bic - 6 * np.log(res.nobs_effective)
    assert_allclose(aic, true["estat_aic"])
    assert_allclose(bic, true["estat_bic"])
    aic = res.info_criteria("aic") - 2 * 6
    bic = res.info_criteria("bic") - 6 * np.log(res.nobs_effective)
    assert_allclose(aic, true["estat_aic"])
    assert_allclose(bic, true["estat_bic"])


def test_append_extend_apply_invalid():
    # Test for invalid options to append, extend, and apply
    niledata = nile.data.load_pandas().data["volume"]
    niledata.index = pd.date_range("1871-01-01", "1970-01-01", freq="YS")

    endog1 = niledata.iloc[:20]
    endog2 = niledata.iloc[20:40]

    mod = sarimax.SARIMAX(endog1, order=(1, 0, 0), concentrate_scale=True)
    res1 = mod.smooth([0.5])

    with pytest.raises(ValueError):
        res1.append(endog2, fit_kwargs={"cov_type": "approx"})
    with pytest.raises(ValueError):
        res1.extend(endog2, fit_kwargs={"cov_type": "approx"})
    with pytest.raises(ValueError):
        res1.apply(endog2, fit_kwargs={"cov_type": "approx"})

    with pytest.raises(ValueError):
        res1.append(endog2, fit_kwargs={"cov_kwds": {}})
    with pytest.raises(ValueError):
        res1.extend(endog2, fit_kwargs={"cov_kwds": {}})
    with pytest.raises(ValueError):
        res1.apply(endog2, fit_kwargs={"cov_kwds": {}})

    # Test for exception when given a different frequency
    wrong_freq = niledata.iloc[20:40]
    wrong_freq.index = pd.date_range(
        start=niledata.index[0], periods=len(wrong_freq), freq="MS"
    )
    message = (
        "Given `endog` does not have an index that extends the index of"
        " the model. Expected index frequency is"
    )
    with pytest.raises(ValueError, match=message):
        res1.append(wrong_freq)
    with pytest.raises(ValueError, match=message):
        res1.extend(wrong_freq)
    message = (
        "Given `exog` does not have an index that extends the index of"
        " the model. Expected index frequency is"
    )
    with pytest.raises(ValueError, match=message):
        res1.append(endog2, exog=wrong_freq)
    message = "The indices for endog and exog are not aligned"
    with pytest.raises(ValueError, match=message):
        res1.extend(endog2, exog=wrong_freq)

    # Test for exception when given the same frequency but not right after the
    # end of model
    not_cts = niledata.iloc[21:41]
    message = (
        "Given `endog` does not have an index that extends the index of the model.$"
    )
    with pytest.raises(ValueError, match=message):
        res1.append(not_cts)
    with pytest.raises(ValueError, match=message):
        res1.extend(not_cts)
    message = (
        "Given `exog` does not have an index that extends the index of the model.$"
    )
    with pytest.raises(ValueError, match=message):
        res1.append(endog2, exog=not_cts)
    message = "The indices for endog and exog are not aligned"
    with pytest.raises(ValueError, match=message):
        res1.extend(endog2, exog=not_cts)

    # # Test for problems with non-date indexes
    endog3 = pd.Series(niledata.iloc[:20].values)
    endog4 = pd.Series(niledata.iloc[:40].values).iloc[20:]
    mod2 = sarimax.SARIMAX(endog3, order=(1, 0, 0), exog=endog3, concentrate_scale=True)
    res2 = mod2.smooth([0.2, 0.5])

    # Test for exception when given the same frequency but not right after the
    # end of model
    not_cts = pd.Series(niledata[:41].values)[21:]
    message = (
        "Given `endog` does not have an index that extends the index of the model.$"
    )
    with pytest.raises(ValueError, match=message):
        res2.append(not_cts)
    with pytest.raises(ValueError, match=message):
        res2.extend(not_cts)
    message = (
        "Given `exog` does not have an index that extends the index of the model.$"
    )
    with pytest.raises(ValueError, match=message):
        res2.append(endog4, exog=not_cts)
    message = "The indices for endog and exog are not aligned"
    with pytest.raises(ValueError, match=message):
        res2.extend(endog4, exog=not_cts)


def test_integer_params():
    # See GH#6335
    mod = sarimax.SARIMAX(
        [1, 1, 1], order=(1, 0, 0), exog=[2, 2, 2], concentrate_scale=True
    )
    res = mod.filter([1, 0])
    # The out-of-sample periods are all missing, so the scale is 0 / 0
    with warns_or_none(RuntimeWarning, match="invalid value encountered"):
        p = res.predict(end=5, dynamic=True, exog=[3, 3, 4])
    assert_equal(p.dtype, np.float64)


def check_states_index(states, ix, predicted_ix, cols):
    predicted_cov_ix = pd.MultiIndex.from_product([predicted_ix, cols]).swaplevel()
    filtered_cov_ix = pd.MultiIndex.from_product([ix, cols]).swaplevel()
    smoothed_cov_ix = pd.MultiIndex.from_product([ix, cols]).swaplevel()

    # Predicted
    assert states.predicted.index.equals(predicted_ix)
    assert states.predicted.columns.equals(cols)

    assert states.predicted_cov.index.equals(predicted_cov_ix)
    assert states.predicted.columns.equals(cols)

    # Filtered
    assert states.filtered.index.equals(ix)
    assert states.filtered.columns.equals(cols)

    assert states.filtered_cov.index.equals(filtered_cov_ix)
    assert states.filtered.columns.equals(cols)

    # Smoothed
    assert states.smoothed.index.equals(ix)
    assert states.smoothed.columns.equals(cols)

    assert states.smoothed_cov.index.equals(smoothed_cov_ix)
    assert states.smoothed.columns.equals(cols)


def test_states_index_periodindex():
    nobs = 10
    ix = pd.period_range(start="2000", periods=nobs, freq="M")
    endog = pd.Series(np.zeros(nobs), index=ix)

    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])

    predicted_ix = pd.period_range(start=ix[0], periods=nobs + 1, freq="M")
    cols = pd.Index(["state.0", "state.1"])

    check_states_index(res.states, ix, predicted_ix, cols)


def test_states_index_dateindex():
    nobs = 10
    ix = pd.date_range(start="2000", periods=nobs, freq=MONTH_END)
    endog = pd.Series(np.zeros(nobs), index=ix)

    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])

    predicted_ix = pd.date_range(start=ix[0], periods=nobs + 1, freq=MONTH_END)
    cols = pd.Index(["state.0", "state.1"])

    check_states_index(res.states, ix, predicted_ix, cols)


def test_states_index_int64index():
    nobs = 10
    ix = pd.Index(np.arange(10))
    endog = pd.Series(np.zeros(nobs), index=ix)

    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])

    predicted_ix = pd.Index(np.arange(11))
    cols = pd.Index(["state.0", "state.1"])

    check_states_index(res.states, ix, predicted_ix, cols)


def test_states_index_rangeindex():
    nobs = 10

    # Basic range index
    ix = pd.RangeIndex(10)
    endog = pd.Series(np.zeros(nobs), index=ix)

    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])

    predicted_ix = pd.RangeIndex(11)
    cols = pd.Index(["state.0", "state.1"])

    check_states_index(res.states, ix, predicted_ix, cols)

    # More complex range index
    ix = pd.RangeIndex(2, 32, 3)
    endog = pd.Series(np.zeros(nobs), index=ix)

    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])

    predicted_ix = pd.RangeIndex(2, 35, 3)
    cols = pd.Index(["state.0", "state.1"])

    check_states_index(res.states, ix, predicted_ix, cols)


def test_invalid_kwargs():
    endog = [0, 0, 1.0]
    # Make sure we can create basic SARIMAX
    sarimax.SARIMAX(endog)
    # Now check that it raises a warning if we add an invalid keyword argument
    with pytest.raises(TypeError, match="Unknown keyword arguments"):
        sarimax.SARIMAX(endog, invalid_kwarg=True)
    # (Note: once deprectation is completed in v0.15, switch to checking for
    # a TypeError, as below)
    # with pytest.raises(TypeError, sarimax.SARIMAX, endog, invalid_kwarg=True)


def test_set_inversion_method_and_initialization_property():
    rs = np.random.RandomState(2026)
    endog = rs.standard_normal(80).cumsum()
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    mod.set_inversion_method(invert_lu=True)
    assert mod.ssm.invert_lu is True

    mod.set_inversion_method(inversion_method=mod.ssm.inversion_method | 0)
    assert isinstance(mod.ssm.inversion_method, int)

    # initialization is a passthrough property to ssm.initialization
    assert mod.initialization is mod.ssm.initialization

    from statsmodels.tsa.statespace.initialization import Initialization
    new_init = Initialization(mod.k_states, "diffuse")
    mod.initialization = new_init
    assert mod.ssm.initialization is new_init
    assert mod.initialization is new_init


@pytest.mark.parametrize("scale", [1e9, 1e12])
@pytest.mark.parametrize("sign", [1, -1])
def test_loglike_unconstrained_stationary_boundary(scale, sign):
    # A line search of a derivative free optimizer can try unconstrained
    # parameters that the transformation maps to unit roots. The stationary
    # covariance of the initial state does not exist then, and the Lyapunov
    # equation is singular. This made the fit of the dynamic factor model of
    # the example notebook fail with a LinAlgError on some platforms.
    rng = np.random.default_rng(78240)
    endog = rng.standard_normal((100, 4))
    mod = dynamic_factor.DynamicFactor(
        endog, k_factors=1, factor_order=2, error_order=2
    )
    params = np.full(mod.k_params, sign * scale)
    assert mod.loglike(params, transformed=False) == -np.inf
    # the filter itself still raises, and the error is a LinAlgError as before
    with pytest.raises(np.linalg.LinAlgError, match="LU decomposition"):
        mod.ssm.loglike()

    # other initializations do not use the stationary covariance
    mod.ssm.initialize_approximate_diffuse()
    assert np.isfinite(mod.loglike(params, transformed=False))


def test_loglike_stationary_init_unit_root_ar1():
    # GH#5459: at a unit root the Lyapunov solver may not raise but return a
    # negative initial state variance (-4.99e291 here). The filter then gave
    # a loglikelihood of 0.0, above that of any stationary model.
    mod = sarimax.SARIMAX(np.zeros(10))
    assert mod.loglike([1.0, 1.0]) == -np.inf
    with pytest.raises(np.linalg.LinAlgError, match="not stationary to numerical"):
        mod.filter([1.0, 1.0])
    with pytest.raises(np.linalg.LinAlgError, match="not stationary to numerical"):
        mod.ssm.loglike()

    # An explosive root is not changed here: the Lyapunov equation has a
    # solution, 1 / (1 - 1.2**2) = -1 / 0.44, although it is not a covariance
    # matrix
    res = mod.filter([1.2, 1.0])
    desired = 1 / (1 - 1.2**2)
    assert_allclose(res.filter_results.initial_state_cov, [[desired]], rtol=1e-12)


@pytest.mark.parametrize("gap", [1e-4, 1e-8, 1e-12, 1e-15])
def test_stationary_init_near_unit_root_ar1(gap):
    # Close to the unit root the stationary covariance is still computed. The
    # variance of a stationary AR(1) process with unit innovation variance is
    # 1 / (1 - phi**2), which is exact for the double precision phi. The
    # relative error of the solver was at most 2.2e-16 for these gaps (scipy
    # 1.8.1 and 1.18.1), the tolerance leaves room for other LAPACK builds.
    phi = 1 - gap
    mod = sarimax.SARIMAX(np.zeros(10))
    res = mod.filter([phi, 1.0])
    desired = 1 / ((1 - phi) * (1 + phi))
    assert_allclose(res.filter_results.initial_state_cov, [[desired]], rtol=1e-12)


@pytest.mark.parametrize(
    "ar_params",
    [
        [0.5, 0.5],
        [1.0, 0.0],
        [2.0, -1.0],
        # a complex pair of roots on the unit circle
        [2 * np.cos(0.7), -1.0],
        # roots 2 and 0.5, which are not unit roots, but their product is 1
        [2.5, -1.0],
    ],
)
def test_loglike_stationary_init_unit_root_ar2(ar_params):
    # GH#5459: loglike returned 0.0 for these AR(2) parameters, with initial
    # state variances of about -3e14 to -2e15, instead of failing. The
    # Lyapunov equation is singular if the product of two eigenvalues of the
    # transition matrix is 1.
    rng = np.random.default_rng(5459)
    mod = sarimax.SARIMAX(rng.standard_normal(200), order=(2, 0, 0))
    assert mod.loglike(np.r_[ar_params, 1.0]) == -np.inf
    with pytest.raises(np.linalg.LinAlgError, match="not stationary to numerical"):
        mod.filter(np.r_[ar_params, 1.0])

    # the loglikelihood of a stationary model is unchanged, and matches the
    # one with the initial covariance from scipy as a known initialization
    params = [0.5, 0.3, 1.0]
    llf = mod.loglike(params)
    transition = np.array([[0.5, 1.0], [0.3, 0.0]])
    cov = solve_discrete_lyapunov(transition, np.diag([1.0, 0.0]))
    mod_known = sarimax.SARIMAX(mod.endog, order=(2, 0, 0))
    mod_known.ssm.initialize_known(np.zeros(2), cov)
    assert_allclose(llf, mod_known.loglike(params), rtol=1e-12)


@pytest.mark.parametrize("gap", [1e-2, 1e-4, 1e-6])
def test_stationary_init_near_unit_root_ar2(gap):
    # The covariance of the state (y_t, phi_2 y_{t-1}) of a stationary AR(2)
    # process with unit innovation variance, from the autocovariances
    # gamma_0 = (1 - phi_2) / ((1 + phi_2) ((1 - phi_2)**2 - phi_1**2)) and
    # gamma_1 = phi_1 gamma_0 / (1 - phi_2). With phi_2 = 0.5 - gap, gamma_0 is
    # about 0.33 / gap. The relative error of the solver was at most 5.1e-10
    # for these gaps (scipy 1.8.1 and 1.18.1), and it grows as the gap shrinks:
    # it is 5.3e-8 for a gap of 1e-8.
    phi1, phi2 = 0.5, 0.5 - gap
    gamma0 = (1 - phi2) / ((1 + phi2) * ((1 - phi2) ** 2 - phi1**2))
    gamma1 = phi1 * gamma0 / (1 - phi2)
    desired = np.array(
        [[gamma0, phi2 * gamma1], [phi2 * gamma1, phi2**2 * gamma0]]
    )
    mod = sarimax.SARIMAX(np.zeros(10), order=(2, 0, 0))
    res = mod.filter([phi1, phi2, 1.0])
    assert_allclose(res.filter_results.initial_state_cov, desired, rtol=1e-8)
    assert np.isfinite(mod.loglike([phi1, phi2, 1.0]))


def test_loglike_stationary_init_unit_root_block():
    # Only the states with a stationary initialization are checked: here the
    # first state is a diffuse random walk and the second is an AR(1)
    endog = np.random.default_rng(5459).standard_normal(50)
    mod = MLEModel(endog, k_states=2, k_posdef=2)
    mod["design"] = [1.0, 1.0]
    mod["selection"] = np.eye(2)
    mod["state_cov"] = np.eye(2)
    mod["obs_cov"] = [[1.0]]
    init = Initialization(2)
    init.set(0, "diffuse")
    init.set(1, "stationary")
    mod.ssm.initialization = init

    mod["transition"] = np.diag([1.0, 0.5])
    assert np.isfinite(mod.ssm.loglike())

    mod["transition"] = np.diag([1.0, 1.0])
    with pytest.raises(np.linalg.LinAlgError, match="not stationary to numerical"):
        mod.ssm.loglike()


@pytest.mark.parametrize("kind", [list, tuple])
@pytest.mark.parametrize(
    "kwds",
    [{}, {"method": "oim"}, {"method": "opg"}, {"transformed": False}],
    ids=["approx", "oim", "opg", "untransformed"],
)
def test_hessian_array_like_params(kind, kwds):
    # params is array_like. A list or tuple raised an AttributeError with the
    # default, complex-step differentiation.
    infl = macrodata.load_pandas().data["infl"].to_numpy()
    mod = sarimax.SARIMAX(infl, order=(1, 0, 0))
    params = np.array([0.5, 4.0])
    desired = mod.hessian(params, **kwds)
    actual = mod.hessian(kind(params.tolist()), **kwds)
    assert_allclose(actual, desired, rtol=1e-13)
    if not kwds:
        # Centered second differences of the loglikelihood agree to about
        # 1e-7 here
        epsilon = 1e-4 * np.maximum(np.abs(params), 0.1)
        approx = approx_hess3(params, mod.loglike, epsilon=epsilon)
        assert_allclose(actual, approx / mod.nobs, rtol=1e-5)


@pytest.mark.parametrize("kind", [list, tuple, np.array])
@pytest.mark.parametrize("approx_centered", [False, True])
def test_transform_jacobian_array_like(kind, approx_centered):
    # The transformations of SARIMAX are phi = -u / sqrt(1 + u**2) and
    # sigma2 = u**2, with derivatives -(1 + u**2)**-1.5 and 2 u. The
    # transformations of the two parameters are separate, so the
    # off-diagonal elements are exactly zero. A list or tuple raised an
    # AttributeError.
    mod = sarimax.SARIMAX(np.zeros(10), order=(1, 0, 0))
    unconstrained = [0.3, 2.0]
    desired = np.diag([-((1 + 0.3**2) ** -1.5), 4.0])
    actual = mod.transform_jacobian(kind(unconstrained), approx_centered=approx_centered)
    # Forward differences are accurate to about 1e-8, centered ones to 1e-10
    assert_allclose(actual, desired, rtol=1e-7 if approx_centered else 1e-5)


def test_append_unnamed_series():
    # A Series without a name raised "Columns must match to concatenate along
    # rows" for a model of a named Series. It is appended as the same Series
    # with the name of the model's endog
    rng = np.random.default_rng(20261123)
    index = pd.date_range("2000-01-01", periods=14, freq="MS")
    data = rng.standard_normal(14)
    endog = pd.Series(data[:11], index=index[:11], name="y")
    new = pd.Series(data[11:], index=index[11:])
    res = sarimax.SARIMAX(endog, order=(1, 0, 0)).smooth([0.5, 1.0])

    actual = res.append(new)
    desired = res.append(new.rename("y"))
    assert new.name is None
    assert actual.model.endog_names == "y"
    assert_allclose(actual.model.endog[:, 0], data, rtol=1e-15)
    assert_allclose(actual.llf, desired.llf, rtol=1e-12)
    assert_allclose(actual.smoothed_state, desired.smoothed_state, rtol=1e-12)
    forecast = actual.forecast(2)
    assert forecast.index.equals(desired.forecast(2).index)
    assert_allclose(forecast, desired.forecast(2), rtol=1e-12)


def _centered_jacobian(func, x):
    # Centered finite differences of a vector-valued function, with the step
    # that balances the truncation and the rounding errors
    step = np.finfo(float).eps ** (1 / 3) * np.maximum(np.abs(x), 0.1)
    columns = []
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = step[i]
        columns.append((func(x + dx) - func(x - dx)) / (2 * step[i]))
    return np.column_stack(columns)


@pytest.mark.parametrize(
    "method, approx_complex_step",
    [
        pytest.param("approx", None, id="approx-default"),
        pytest.param("approx", False, id="approx-fd"),
        pytest.param("harvey", None, id="harvey-default"),
        pytest.param("harvey", False, id="harvey-fd"),
    ],
)
@pytest.mark.parametrize(
    "name", ["sarimax_arma21", "var2", "dynamic_factor", "sarimax_ar1"]
)
def test_score_obs_untransformed(name, method, approx_complex_step):
    # The scores of the observations with respect to the unconstrained
    # parameters u. The "harvey" method returned the scores with respect to
    # the constrained parameters, evaluated at u. By default, complex-step
    # differentiation was applied through transform_params, which is wrong
    # for the VAR(2) and the dynamic factor model, whose transformations are
    # not complex differentiable. The reference is the derivative of the
    # loglikelihood of each observation as a function of u, at a point away
    # from the optimum, and their sum is the score.
    mod = _score_model(name)
    uparams = mod.untransform_params(mod.start_params) + 0.1
    actual = mod.score_obs(
        uparams,
        method=method,
        transformed=False,
        approx_complex_step=approx_complex_step,
    )
    desired = _centered_jacobian(
        lambda params: mod.loglikeobs(params, transformed=False), uparams
    )
    # The errors of numerical derivatives scale with the largest element, not
    # with each element
    atol = 1e-4 * np.max(np.abs(desired))
    assert_allclose(actual, desired, rtol=1e-5, atol=atol)
    score = mod.score(
        uparams,
        method=method,
        transformed=False,
        approx_complex_step=approx_complex_step,
    )
    assert_allclose(actual.sum(0), score, rtol=1e-5, atol=1e-4 * np.max(np.abs(score)))


@pytest.mark.parametrize(
    "method, approx_complex_step",
    [
        pytest.param("approx", None, id="approx-default"),
        pytest.param("harvey", None, id="harvey-default"),
        pytest.param("harvey", False, id="harvey-fd"),
    ],
)
@pytest.mark.parametrize("name", ["sarimax_arma21", "var2"])
def test_score_obs_untransformed_chain_rule(name, method, approx_complex_step):
    # The derivatives with respect to the constrained parameters are those of
    # transformed=True, by default complex-step derivatives, and the chain
    # rule gives the scores with respect to u to the accuracy of the Jacobian
    # of the transformation (centered differences, about 1e-10 here). For the
    # SARIMAX model, whose transformation is complex differentiable, the
    # default "approx" result is that of statsmodels 0.15.0, to about 1e-10.
    mod = _score_model(name)
    uparams = mod.untransform_params(mod.start_params) + 0.1
    params = mod.transform_params(uparams)
    jacobian = _centered_jacobian(mod.transform_params, uparams)
    desired = mod.score_obs(
        params, method=method, approx_complex_step=approx_complex_step is None
    )
    desired = desired @ jacobian
    actual = mod.score_obs(
        uparams,
        method=method,
        transformed=False,
        approx_complex_step=approx_complex_step,
    )
    assert_allclose(actual, desired, rtol=1e-8, atol=1e-8 * np.max(np.abs(desired)))


@pytest.mark.parametrize("approx_centered", [False, True])
def test_score_obs_untransformed_finite_difference(approx_centered):
    # With finite differences and the "approx" method, the loglikelihoods of
    # the observations are differentiated through the transformation, as
    # before, which does not need complex values
    mod = _score_model("var2")
    uparams = mod.untransform_params(mod.start_params) + 0.1
    desired = approx_fprime(
        uparams,
        mod.loglikeobs,
        kwargs={"transformed": False},
        centered=approx_centered,
    )
    actual = mod.score_obs(
        uparams,
        transformed=False,
        approx_complex_step=False,
        approx_centered=approx_centered,
    )
    assert_allclose(actual, desired, rtol=1e-10, atol=1e-10 * np.max(np.abs(desired)))


@pytest.mark.parametrize(
    "method, approx_complex_step",
    [
        pytest.param("approx", None, id="approx-default"),
        pytest.param("harvey", None, id="harvey-default"),
        pytest.param("harvey", False, id="harvey-fd"),
    ],
)
def test_score_obs_untransformed_includes_fixed(method, approx_complex_step):
    # With includes_fixed=True, all of the parameters passed are used and
    # transformed, also those that are fixed, so that the scores are those of
    # the model without fixed parameters
    mod = _score_model("sarimax_arma21")
    uparams = mod.untransform_params(mod.start_params) + 0.1
    kwds = {
        "method": method,
        "transformed": False,
        "approx_complex_step": approx_complex_step,
    }
    desired = mod.score_obs(uparams, **kwds)
    start_params = dict(zip(mod.param_names, mod.start_params, strict=True))
    with mod.fix_params({"ma.L1": start_params["ma.L1"]}):
        actual = mod.score_obs(uparams, includes_fixed=True, **kwds)
    assert_allclose(actual, desired, rtol=1e-10, atol=1e-10 * np.max(np.abs(desired)))


@pytest.mark.parametrize(
    "func, transformed",
    [("score", True), ("score_obs", True), ("score_obs", False)],
)
def test_score_harvey_approx_centered(func, transformed):
    # With finite differences, the "harvey" method ignored approx_centered
    # and used forward differences. The reference is the centered difference
    # of the loglikelihood (of each observation, for score_obs). For this
    # model the errors relative to the largest element are 3e-6 to 3e-5 with
    # forward differences and at most 4e-8 with centered ones.
    mod = _score_model("sarimax_seasonal")
    if transformed:
        params = mod.start_params * 1.05
    else:
        params = mod.untransform_params(mod.start_params) + 0.1
    if func == "score":
        desired = _centered_gradient(
            lambda x: mod.loglike(x, transformed=transformed), params
        )
    else:
        desired = _centered_jacobian(
            lambda x: mod.loglikeobs(x, transformed=transformed), params
        )
    actual = getattr(mod, func)(
        params,
        method="harvey",
        transformed=transformed,
        approx_complex_step=False,
        approx_centered=True,
    )
    tol = 3e-7
    assert_allclose(actual, desired, rtol=tol, atol=tol * np.max(np.abs(desired)))


FIXED_PARAMS_SCORE_METHODS = [
    pytest.param("approx", None, id="approx-default"),
    pytest.param("approx", False, id="approx-fd"),
    pytest.param("harvey", None, id="harvey-default"),
]


def _fixed_params_score_model():
    # ARMA(2, 1) with the MA coefficient fixed at its starting value
    mod = _score_model("sarimax_arma21")
    names = mod.param_names
    fixed = {"ma.L1": mod.start_params[names.index("ma.L1")]}
    free = [i for i, name in enumerate(names) if name not in fixed]
    return mod, fixed, free


@pytest.mark.parametrize("method, approx_complex_step", FIXED_PARAMS_SCORE_METHODS)
def test_score_obs_fixed_params(method, approx_complex_step):
    # With fixed parameters, score_obs had a column for every parameter, as
    # with includes_fixed=True, although score has one for each free
    # parameter. The scores are those of the model without fixed parameters,
    # at the same values, for the free parameters, and they sum to the score
    mod, fixed, free = _fixed_params_score_model()
    params = mod.start_params
    kwds = {"method": method, "approx_complex_step": approx_complex_step}
    desired = mod.score_obs(params, **kwds)[:, free]
    desired_opg = mod.opg_information_matrix(params)[np.ix_(free, free)]
    with mod.fix_params(fixed):
        actual = mod.score_obs(params[free], **kwds)
        score = mod.score(params[free], **kwds)
        opg = mod.opg_information_matrix(params[free])
        hessian = mod.hessian(params[free], method="opg")
    assert_allclose(actual, desired, rtol=1e-12)
    # The score differentiates the sum of the loglikelihoods instead
    assert_allclose(actual.sum(0), score, rtol=1e-6)
    assert_allclose(opg, desired_opg, rtol=1e-12)
    assert_allclose(hessian, -desired_opg, rtol=1e-12)


@pytest.mark.parametrize("method, approx_complex_step", FIXED_PARAMS_SCORE_METHODS)
def test_score_obs_fixed_params_untransformed(method, approx_complex_step):
    # With transformed=False, the fixed value (a constrained parameter) was
    # also transformed, as if it were unconstrained, so that the scores were
    # those at other parameters. The reference is the derivative of the
    # loglikelihood of each observation with respect to the unconstrained
    # free parameters, and their sum is the score
    mod, fixed, free = _fixed_params_score_model()
    uparams = mod.untransform_params(mod.start_params)[free] + 0.1
    kwds = {"method": method, "approx_complex_step": approx_complex_step}
    with mod.fix_params(fixed):
        actual = mod.score_obs(uparams, transformed=False, **kwds)
        score = mod.score(uparams, transformed=False, **kwds)
        desired = _centered_jacobian(
            lambda params: mod.loglikeobs(params, transformed=False), uparams
        )
        opg = mod.opg_information_matrix(
            uparams, transformed=False, approx_complex_step=False
        )
    # The errors of numerical derivatives scale with the largest element, not
    # with each element
    atol = 1e-4 * np.max(np.abs(desired))
    assert_allclose(actual, desired, rtol=1e-5, atol=atol)
    assert_allclose(actual.sum(0), score, rtol=1e-5)
    desired_opg = desired.T @ desired / mod.nobs
    assert_allclose(opg, desired_opg, rtol=1e-4)


# Closed-form references for an AR(1) model, used by the tests below. With
# the stationary initialization, the prediction errors and their variances
# are v_1 = y_1, F_1 = sigma2 / (1 - phi**2), and v_t = y_t - phi y_{t-1},
# F_t = sigma2 for t > 1. A short series keeps the first observation, whose
# terms differ from the others, important in every sum.
_AR1_ENDOG = np.array([0.8, -0.3, 1.1, 0.4, -0.9, 0.2])
_AR1_PARAMS = np.array([0.5, 1.5])


def _ar1_closed_form(endog, phi, sigma2):
    endog = np.asarray(endog, dtype=float)
    nobs = endog.shape[0]
    v = np.r_[endog[0], endog[1:] - phi * endog[:-1]]
    f = np.r_[sigma2 / (1 - phi**2), np.full(nobs - 1, sigma2)]
    # Derivatives of v_t and F_t with respect to (phi, sigma2)
    dv = np.zeros((nobs, 2))
    dv[1:, 0] = -endog[:-1]
    df = np.zeros((nobs, 2))
    df[0, 0] = 2 * phi * sigma2 / (1 - phi**2) ** 2
    df[0, 1] = 1 / (1 - phi**2)
    df[1:, 1] = 1.0
    llf_obs = -0.5 * np.log(2 * np.pi * f) - 0.5 * v**2 / f
    score_obs = (
        -0.5 * df / f[:, None] * (1 - v[:, None] ** 2 / f[:, None])
        - v[:, None] * dv / f[:, None]
    )
    # The information matrix of Harvey (1989), equation 3.4.69, which only
    # depends on the first derivatives of v_t and F_t
    oim = (
        0.5 * np.einsum("ti,tj->ij", df / f[:, None], df / f[:, None])
        + np.einsum("ti,tj->ij", dv / np.sqrt(f)[:, None], dv / np.sqrt(f)[:, None])
    ) / nobs
    # The second derivatives of the loglikelihood
    hessian = np.zeros((2, 2))
    y0 = endog[0]
    hessian[0, 0] = -(1 + phi**2) / (1 - phi**2) ** 2 + y0**2 / sigma2
    hessian[0, 1] = -phi * y0**2 / sigma2**2
    hessian[1, 1] = 0.5 / sigma2**2 - y0**2 * (1 - phi**2) / sigma2**3
    lag = endog[:-1]
    hessian[0, 0] += np.sum(-(lag**2) / sigma2)
    hessian[0, 1] += np.sum(-v[1:] * lag / sigma2**2)
    hessian[1, 1] += np.sum(0.5 / sigma2**2 - v[1:] ** 2 / sigma2**3)
    hessian[1, 0] = hessian[0, 1]
    return SimpleNamespace(
        llf_obs=llf_obs,
        llf=llf_obs.sum(),
        score_obs=score_obs,
        oim=oim,
        opg=score_obs.T @ score_obs / nobs,
        hessian=hessian,
    )


def _ar1_model():
    return sarimax.SARIMAX(_AR1_ENDOG, order=(1, 0, 0))


def test_ar1_closed_form_loglikelihood():
    # The reference itself against the Kalman filter
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    assert_allclose(mod.loglikeobs(_AR1_PARAMS), cf.llf_obs, rtol=1e-12)


def test_loglike_flag_positional_and_keyword_raises():
    # A flag that is given both positionally and by keyword raises
    mod = _ar1_model()
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    assert_allclose(mod.loglike(_AR1_PARAMS, True, False, False), cf.llf, rtol=1e-12)
    with pytest.raises(TypeError, match="multiple values for keyword argument"):
        mod.loglike(_AR1_PARAMS, True, False, False, transformed=True)


@pytest.mark.parametrize(
    "func",
    [
        lambda mod: mod.fit(start_params=[], optim_complex_step=True, disp=0),
        lambda mod: mod.score([], approx_complex_step=True),
        lambda mod: mod.score_obs([], approx_complex_step=True),
        lambda mod: mod.hessian([], approx_complex_step=True),
    ],
    ids=["fit", "score", "score_obs", "hessian"],
)
def test_complex_endog_complex_step_raises(func):
    # Complex-step derivatives need real data
    mod = MLEModel(np.array([1.0 + 0.5j, 2.0, 1.5]), **kwargs)
    with pytest.raises(ValueError, match="Cannot use complex step derivatives"):
        func(mod)


@pytest.mark.parametrize("method", ["filter", "smooth"])
def test_filter_smooth_complex_step(method):
    # Filtering at parameters with an imaginary perturbation h e_k gives the
    # derivative of the loglikelihood as its imaginary part divided by h. The
    # model has three variables and solves with the Cholesky factorization
    # by default, which is not complex-analytic, so that the complex step
    # must switch to the LU decomposition. The reference is the centered
    # finite difference gradient, whose error is about 1e-7 relative to the
    # largest element. The initialization is known, so that it does not
    # depend on the parameters.
    mod = _score_model("dynamic_factor")
    mod.ssm.initialize_known(np.zeros(mod.k_states), np.eye(mod.k_states))
    params = mod.start_params
    func = getattr(mod, method)
    step = 1e-20
    actual = np.zeros_like(params)
    for i in range(len(params)):
        params_cs = params.astype(complex)
        params_cs[i] += 1j * step
        res = func(params_cs, complex_step=True, return_ssm=True)
        actual[i] = res.llf.imag / step
        assert_allclose(res.llf.real, mod.loglike(params), rtol=1e-12)
    desired = _centered_gradient(mod.loglike, params)
    assert_allclose(actual, desired, rtol=1e-5, atol=1e-6 * np.max(np.abs(desired)))


@pytest.mark.parametrize(
    "approx_complex_step, approx_centered, rtol",
    [
        # The default uses complex-step derivatives, which are exact here
        (None, False, 1e-8),
        (True, False, 1e-8),
        # Finite differences are accurate to about the square root (forward)
        # and the two-thirds power (centered) of the machine precision
        (False, False, 1e-5),
        (False, True, 1e-7),
    ],
)
def test_observed_information_matrix_ar1(approx_complex_step, approx_centered, rtol):
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    kwds = {"approx_centered": approx_centered}
    if approx_complex_step is not None:
        kwds["approx_complex_step"] = approx_complex_step
    actual = mod.observed_information_matrix(_AR1_PARAMS, **kwds)
    # The cross term only comes from the first observation and is small; the
    # errors of the numerical derivatives scale with the largest element
    assert_allclose(actual, cf.oim, rtol=rtol, atol=rtol * np.max(np.abs(cf.oim)))
    if approx_complex_step is None:
        # The default for transformed parameters is the complex step
        desired = mod.observed_information_matrix(
            _AR1_PARAMS, approx_complex_step=True
        )
        assert_allclose(actual, desired, rtol=1e-15)


def test_observed_information_matrix_hessian_oim_ar1():
    # hessian(method="oim") is the negative of the information matrix
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    actual = mod.hessian(_AR1_PARAMS, method="oim", approx_complex_step=False)
    assert_allclose(actual, -cf.oim, rtol=1e-5, atol=1e-5 * np.max(np.abs(cf.oim)))


@pytest.mark.parametrize("approx_complex_step", [None, True, False])
def test_opg_information_matrix_ar1(approx_complex_step):
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    kwds = {}
    if approx_complex_step is not None:
        kwds["approx_complex_step"] = approx_complex_step
    actual = mod.opg_information_matrix(_AR1_PARAMS, **kwds)
    # Forward differences are accurate to about 1e-8 relative to the scores
    rtol = 1e-5 if approx_complex_step is False else 1e-10
    assert_allclose(actual, cf.opg, rtol=rtol)


@pytest.mark.parametrize(
    "func",
    [
        "observed_information_matrix",
        "opg_information_matrix",
        "score_obs",
    ],
)
def test_untransformed_complex_step_raises(func):
    # Complex-step derivatives are not available for unconstrained parameters
    # in these methods, since transform_params need not be complex-analytic
    mod = _ar1_model()
    params = mod.untransform_params(_AR1_PARAMS)
    with pytest.raises(ValueError, match="Cannot use complex-step approximations"):
        getattr(mod, func)(params, transformed=False, approx_complex_step=True)


@pytest.mark.parametrize("approx_complex_step, rtol", [(True, 1e-10), (False, 1e-5)])
def test_score_obs_harvey_ar1(approx_complex_step, rtol):
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    actual = mod.score_obs(
        _AR1_PARAMS, method="harvey", approx_complex_step=approx_complex_step
    )
    # The errors of the numerical derivatives scale with the largest score
    atol = rtol * np.max(np.abs(cf.score_obs))
    assert_allclose(actual, cf.score_obs, rtol=rtol, atol=atol)


def test_hessian_complex_step_ar1():
    # The default, second-order complex-step differentiation, is accurate to
    # the rounding error here
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    assert_allclose(mod.hessian(_AR1_PARAMS), cf.hessian / mod.nobs, rtol=1e-8)


@pytest.mark.parametrize("approx_centered", [False, True])
def test_hessian_finite_difference_ar1(approx_centered):
    # Second derivatives by finite differences of finite differences are only
    # accurate to about 2% here, so that they are checked to the precision
    # that catches a wrong element, scale or sign
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    with pytest.warns(PrecisionWarning, match="finite differences"):
        actual = mod.hessian(
            _AR1_PARAMS, approx_complex_step=False, approx_centered=approx_centered
        )
    assert_allclose(actual, cf.hessian / mod.nobs, rtol=0.1)


@pytest.mark.parametrize(
    "approx_centered, approx_type",
    [(False, "finite differences"), (True, "centered finite differences")],
)
def test_cov_params_approx_finite_difference(approx_centered, approx_type):
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    cov_kwds = {"approx_complex_step": False, "approx_centered": approx_centered}
    with pytest.warns(PrecisionWarning, match="finite differences"):
        res = mod.filter(_AR1_PARAMS, cov_type="approx", cov_kwds=cov_kwds)
    assert res.cov_kwds["description"] == (
        f"Covariance matrix calculated using numerical ({approx_type})"
        " differentiation."
    )
    # As above, the finite difference Hessian is only accurate to about 2%
    desired = np.linalg.inv(-cf.hessian)
    assert_allclose(res.cov_params_default, desired, rtol=0.1)


def test_aicc_ar1():
    # aicc = -2 llf + 2 k n / (n - k - 1), with k = 2 parameters
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    res = _ar1_model().filter(_AR1_PARAMS)
    nobs, k = len(_AR1_ENDOG), 2
    desired = -2 * cf.llf + 2 * k + 2 * k * (k + 1) / (nobs - k - 1)
    assert_allclose(res.aicc, desired, rtol=1e-12)


def test_lutkepohl_information_criteria_time_varying_state_cov_raises():
    endog = np.array([1.0, 2.0, 1.5])
    mod = MLEModel(
        endog,
        k_states=1,
        design=[[1.0]],
        transition=[[1.0]],
        selection=[[1.0]],
        state_cov=np.ones((1, 1, 3)),
        initialization="approximate_diffuse",
    )
    res = mod.filter([])
    # The standard criteria are available
    assert_allclose(res.info_criteria("aic"), -2 * res.llf, rtol=1e-12)
    with pytest.raises(ValueError, match="time-varying state covariance"):
        res.info_criteria("aic", method="lutkepohl")


def test_states_unsupported_index_warns():
    endog = pd.Series([1.0, 2.0, 1.5], index=[0.5, 1.5, 2.5])
    with pytest.warns(ValueWarning, match="unsupported index"):
        mod = MLEModel(endog, **kwargs)
    res = mod.filter([])
    with pytest.warns(ValueWarning, match="No supported index is available"):
        states = res.states
    assert_allclose(states.filtered, res.filtered_state.T, rtol=1e-15)
    assert_equal(states.filtered.index.to_numpy(), np.arange(3))


@pytest.mark.parametrize("extend_model", [None, True])
def test_get_forecast_date_steps_ar1(extend_model):
    # The h-step forecast is phi**h y_n, with variance
    # sigma2 (1 + phi**2 + ... + phi**(2 (h - 1)))
    index = pd.date_range("2000-01-01", periods=len(_AR1_ENDOG), freq="MS")
    endog = pd.Series(_AR1_ENDOG, index=index)
    phi, sigma2 = _AR1_PARAMS
    res = sarimax.SARIMAX(endog, order=(1, 0, 0)).filter(_AR1_PARAMS)
    horizon = np.arange(1, 4)
    desired_mean = phi**horizon * _AR1_ENDOG[-1]
    desired_var = sigma2 * np.cumsum(phi ** (2 * (horizon - 1)))
    for steps in [3, "2000-09-01"]:
        fcast = res.get_forecast(steps, extend_model=extend_model)
        assert_allclose(fcast.predicted_mean, desired_mean, rtol=1e-12)
        assert_allclose(fcast.var_pred_mean, desired_var, rtol=1e-12)
        assert fcast.predicted_mean.index.equals(
            pd.date_range("2000-07-01", periods=3, freq="MS")
        )


def _ar1_simulate_closed_form(initial_state, phi, state_shocks):
    # The AR(1) model of SARIMAX has no measurement error, so that the
    # observation is the state: y_0 = initial_state, y_{t+1} = phi y_t + eta_t
    out = [initial_state]
    for eta in state_shocks[:-1]:
        out.append(phi * out[-1] + eta)
    return np.array(out)


_SIM_SHOCKS = np.array([0.5, -1.0, 0.25, 2.0])


@pytest.fixture
def ar1_dated():
    index = pd.date_range("2000-01-01", periods=24, freq="MS")
    endog = pd.Series(np.random.default_rng(1234).standard_normal(24), index=index)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    params = np.array([0.6, 2.0])
    return mod, params


@pytest.mark.parametrize("anchor", [12, -12, "2001", "2001-01-01"])
@pytest.mark.parametrize("on_results", [False, True])
def test_simulate_anchor_ar1(ar1_dated, anchor, on_results):
    # The year "2001" covers 12 dates, so that its location is a slice
    mod, params = ar1_dated
    sim_kwds = {
        "anchor": anchor,
        "initial_state": [1.5],
        "state_shocks": _SIM_SHOCKS,
        "measurement_shocks": np.zeros(4),
    }
    if on_results:
        sim = mod.filter(params).simulate(4, **sim_kwds)
    else:
        sim = mod.simulate(params, 4, **sim_kwds)
    desired = _ar1_simulate_closed_form(1.5, params[0], _SIM_SHOCKS)
    assert_allclose(sim, desired, rtol=1e-12)
    assert sim.index[0] == pd.Timestamp("2001-01-01")


@pytest.mark.parametrize("anchor", [12, "2001"])
def test_results_simulate_anchor_initial_state_drawn_ar1(ar1_dated, anchor):
    # Without an initial state, the results draw it from the predicted state
    # at the anchor, which is N(phi y_11, sigma2) for the AR(1)
    mod, params = ar1_dated
    phi, sigma2 = params
    sim = mod.filter(params).simulate(
        4,
        anchor=anchor,
        state_shocks=_SIM_SHOCKS,
        measurement_shocks=np.zeros(4),
        rng=np.random.default_rng(4290),
    )
    mean = phi * mod.endog[11, 0]
    initial_state = np.random.default_rng(4290).multivariate_normal(
        [mean], [[sigma2]], size=1
    )[0, 0]
    desired = _ar1_simulate_closed_form(initial_state, phi, _SIM_SHOCKS)
    assert_allclose(sim, desired, rtol=1e-12)


def test_simulate_out_of_sample_extend_model_ar1(ar1_dated):
    mod, params = ar1_dated
    sim = mod.simulate(
        params,
        4,
        anchor="end",
        initial_state=[1.5],
        state_shocks=_SIM_SHOCKS,
        measurement_shocks=np.zeros(4),
        extend_model=True,
    )
    desired = _ar1_simulate_closed_form(1.5, params[0], _SIM_SHOCKS)
    assert_allclose(sim, desired, rtol=1e-12)
    assert sim.index[0] == pd.Timestamp("2002-01-01")


def test_simulate_initial_state_per_repetition_ar1(ar1_dated):
    # One initial state for each repetition, as columns
    mod, params = ar1_dated
    sim = mod.simulate(
        params,
        4,
        initial_state=[[1.5, -2.0]],
        repetitions=2,
        state_shocks=_SIM_SHOCKS,
        measurement_shocks=np.zeros(4),
    )
    for i, initial_state in enumerate([1.5, -2.0]):
        desired = _ar1_simulate_closed_form(initial_state, params[0], _SIM_SHOCKS)
        assert_allclose(sim.iloc[:, i], desired, rtol=1e-12)


@pytest.mark.parametrize("on_results", [False, True])
def test_simulate_anchor_invalid_raises(ar1_dated, on_results):
    mod, params = ar1_dated
    if on_results:
        func = mod.filter(params).simulate
    else:

        def func(*args, **kwds):
            return mod.simulate(params, *args, **kwds)

    with pytest.raises(ValueError, match="Cannot anchor simulation outside"):
        func(4, anchor=25, initial_state=[1.5])
    if not on_results:
        # The results object draws the initial state from the predicted state
        with pytest.raises(ValueError, match="must provide a value for"):
            func(4, anchor=10)


@pytest.fixture
def time_varying_ar1():
    # An AR(1) state with a time-varying coefficient phi_t, observed without
    # noise. The impulse response to a unit shock at the anchor a is
    # prod(phi_{a+1}, ..., phi_{a+h}) for h = 0, 1, ...
    index = pd.date_range("2000-01-01", periods=24, freq="MS")
    phis = np.linspace(0.1, 0.9, 24)
    mod = MLEModel(
        pd.Series(np.zeros(24), index=index),
        k_states=1,
        design=[[1.0]],
        transition=phis[None, None, :],
        selection=[[1.0]],
        state_cov=[[1.0]],
        obs_cov=[[0.0]],
        initialization="diffuse",
    )
    return mod, phis


@pytest.mark.parametrize("anchor", [12, -12, "2001", "2001-01-01"])
def test_impulse_responses_anchor_time_varying(time_varying_ar1, anchor):
    mod, phis = time_varying_ar1
    irfs = mod.impulse_responses([], steps=4, anchor=anchor)
    desired = np.r_[1.0, np.cumprod(phis[13:16])]
    assert_allclose(irfs, desired, rtol=1e-12)


@pytest.mark.parametrize("anchor", [24, "2002-01-01"])
def test_impulse_responses_anchor_out_of_sample_raises(time_varying_ar1, anchor):
    mod, _ = time_varying_ar1
    with pytest.raises(ValueError, match="Cannot anchor impulse responses outside"):
        mod.impulse_responses([], steps=4, anchor=anchor)


def test_impulse_responses_extend_model_time_varying_regression():
    # A regression with a random walk coefficient beta_t: y_t = x_t beta_t +
    # e_t. The response of y_{a+1+h} to a unit shock to beta at the anchor a
    # is x_{a+1+h}, so that responses beyond the sample need new exog values,
    # which the extended model takes
    rng = np.random.default_rng(5)
    exog = rng.standard_normal(23)
    endog = 0.7 * exog[:20] + rng.standard_normal(20)
    mod = sarimax.SARIMAX(
        endog,
        exog=exog[:20],
        order=(0, 0, 0),
        time_varying_regression=True,
        mle_regression=False,
    )
    res = mod.smooth([0.5, 1.0])
    for extend_model in [None, True]:
        irfs = res.impulse_responses(
            steps=3, anchor="end", exog=exog[20:], extend_model=extend_model
        )
        assert_allclose(irfs, exog[20:], rtol=1e-12)


def test_validate_out_of_sample_exog():
    # A forecast of a model with exog needs exog values, and exog values for
    # an in-sample prediction are ignored with a warning
    rng = np.random.default_rng(2207)
    exog = rng.standard_normal(30)
    endog = rng.standard_normal((30, 2))
    res = sarimax.SARIMAX(endog[:, 0], exog=exog, order=(1, 0, 0)).smooth(
        [0.5, 0.5, 1.0]
    )
    with pytest.raises(ValueError, match="require additional exogenous values"):
        res.forecast(2)
    mod = varmax.VARMAX(endog, exog=exog, order=(1, 0), trend="n")
    res = mod.smooth(mod.start_params)
    desired = res.get_prediction(start=25, end=29).predicted_mean
    with pytest.warns(ValueWarning, match="`exog` argument ignored"):
        pred = res.get_prediction(start=25, end=29, exog=exog[:2])
    assert_allclose(pred.predicted_mean, desired, rtol=1e-15)


def test_fit_untransformed_start_params_and_flags():
    # Newton with no iterations returns the start parameters, so that the
    # parameters, the score and the Hessian are evaluated at known values
    mod = _ar1_model()
    uparams = mod.untransform_params(_AR1_PARAMS)
    res = mod.fit(
        start_params=uparams,
        transformed=False,
        method="newton",
        maxiter=0,
        disp=0,
        warn_convergence=False,
    )
    assert_allclose(res.mle_settings["start_params"], uparams, rtol=1e-15)
    assert_allclose(res.params, _AR1_PARAMS, rtol=1e-14)

    # Additional flags are passed on to the score: here a centered finite
    # difference score, which differs from the forward one by about 1e-7
    flags = {"approx_centered": True}
    with pytest.warns(PrecisionWarning, match="finite differences"):
        res = mod.fit(
            start_params=_AR1_PARAMS,
            method="newton",
            maxiter=0,
            disp=0,
            warn_convergence=False,
            optim_complex_step=False,
            flags=flags,
        )
    (fargs_flags,) = res.mle_settings["fargs"]
    assert fargs_flags["approx_centered"] is True
    score = mod.score(
        uparams, transformed=False, approx_complex_step=False, approx_centered=True
    )
    assert_allclose(res.mle_retvals["score"], -score / mod.nobs, rtol=1e-12)


class _CustomResults(MLEResults):
    pass


class _CustomResultsWrapper(MLEResultsWrapper):
    pass


def test_filter_results_class_or_wrapper_class():
    # Either class can be given alone; the other is the model's default
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    mod = _ar1_model()
    default = mod.filter(_AR1_PARAMS)
    res = mod.filter(_AR1_PARAMS, results_class=_CustomResults)
    assert type(res) is type(default)
    assert type(res._results) is _CustomResults
    assert_allclose(res.llf, cf.llf, rtol=1e-12)
    res = mod.filter(_AR1_PARAMS, results_wrapper_class=_CustomResultsWrapper)
    assert type(res) is _CustomResultsWrapper
    assert type(res._results) is type(default._results)
    assert_allclose(res.llf, cf.llf, rtol=1e-12)


def test_filter_results_without_smoother():
    # With results_class=FilterResults, the state space model only filters.
    # extend then filters the new data from the predicted state at the end
    # of the sample, so that its loglikelihood is the remainder of that of
    # the whole sample. news needs the smoother.
    index = pd.date_range("2000-01-01", periods=len(_AR1_ENDOG), freq="MS")
    endog = pd.Series(_AR1_ENDOG, index=index)
    mod = sarimax.SARIMAX(
        endog.iloc[:4], order=(1, 0, 0), results_class=kalman_filter.FilterResults
    )
    res = mod.filter(_AR1_PARAMS)
    assert res.smoother_results is None
    res_extend = res.extend(endog.iloc[4:])
    assert res_extend.smoother_results is None
    cf = _ar1_closed_form(_AR1_ENDOG, *_AR1_PARAMS)
    assert_allclose(res_extend.llf, cf.llf_obs[4:].sum(), rtol=1e-12)
    with pytest.raises(ValueError, match="without Kalman smoother results"):
        res.news(res_extend, impact_date=5)


def test_append_refit_fixed_params():
    # With the scale concentrated out and the autoregressive coefficient
    # fixed, refitting estimates nothing, and the scale is the closed form
    # sum(v_t**2 / f_t) / n of the appended sample with f_t = F_t / sigma2
    endog = np.r_[_AR1_ENDOG, 0.6, -0.1]
    phi = 0.5
    mod = sarimax.SARIMAX(endog[:6], order=(1, 0, 0), concentrate_scale=True)
    res = mod.fit_constrained({"ar.L1": phi}, disp=0)
    res_append = res.append(endog[6:], refit=True, fit_kwargs={"disp": 0})
    assert res_append.fixed_params == ["ar.L1"]
    assert_allclose(res_append.params, [phi], rtol=1e-15)
    v = np.r_[endog[0] * np.sqrt(1 - phi**2), endog[1:] - phi * endog[:-1]]
    assert_allclose(res_append.scale, np.sum(v**2) / len(endog), rtol=1e-12)


def test_append_refit():
    # Refitting after append is the same fit as on the whole dataset, from
    # the parameters of the first fit
    endog = np.random.default_rng(4425).standard_normal(40)
    res = sarimax.SARIMAX(endog[:30], order=(1, 0, 0)).fit(disp=0)
    res_append = res.append(endog[30:], refit=True, fit_kwargs={"disp": 0})
    desired = sarimax.SARIMAX(endog, order=(1, 0, 0)).fit(
        start_params=res.params, disp=0
    )
    assert_equal(res_append.nobs, 40)
    assert_allclose(res_append.params, desired.params, rtol=1e-12)
    assert_allclose(res_append.llf, desired.llf, rtol=1e-12)


def test_get_prediction_low_memory_filtered():
    # Without the filtered states (low memory), filtered predictions are NaN,
    # while the one-step-ahead predictions phi y_{t-1} are available
    phi = _AR1_PARAMS[0]
    res = _ar1_model().filter(_AR1_PARAMS, low_memory=True)
    pred = res.get_prediction(start=1, end=5, information_set="filtered")
    assert np.all(np.isnan(pred.predicted_mean))
    assert np.all(np.isnan(pred.var_pred_mean))
    pred = res.get_prediction(start=1, end=5, information_set="predicted")
    assert_allclose(pred.predicted_mean, phi * _AR1_ENDOG[:-1], rtol=1e-12)


@pytest.fixture
def ar1_vintages():
    # The previous vintage has n observations; the updated vintage revises
    # the last of them by 1.5 and adds one
    nobs = 12
    index = pd.date_range("2000-01-01", periods=nobs + 1, freq="MS")
    endog = np.random.default_rng(42).standard_normal(nobs + 1)
    revised = endog.copy()
    revised[nobs - 1] += 1.5
    previous = pd.Series(endog[:nobs], index=index[:nobs], name="y")
    updated = pd.Series(revised, index=index, name="y")
    params = [0.6, 2.0]
    res_previous = sarimax.SARIMAX(previous, order=(1, 0, 0)).smooth(params)
    res_updated = sarimax.SARIMAX(updated, order=(1, 0, 0)).smooth(params)
    return res_previous, res_updated, previous, updated


@pytest.mark.parametrize("revisions_details_start", [11, "2000-12", "2000-12-01"])
@pytest.mark.parametrize("state_index", [None, [0]])
def test_news_revisions_details_start_ar1(
    ar1_vintages, revisions_details_start, state_index
):
    # The forecast of y_{n+1} is phi y_n given the updated data and
    # phi**2 y_{n-1} given the previous data. The revision of y_{n-1}
    # changes the latter by phi**2 1.5, and the news is
    # phi (y_n - phi y_{n-1}), with the revised y_{n-1}.
    res_previous, res_updated, previous, updated = ar1_vintages
    nobs = len(previous)
    phi = 0.6
    news = res_previous.news(
        res_updated,
        impact_date=nobs + 1,
        revisions_details_start=revisions_details_start,
        state_index=state_index,
    )
    total = phi * updated.iloc[nobs] - phi**2 * previous.iloc[nobs - 1]
    revision = phi**2 * 1.5
    update = phi * (updated.iloc[nobs] - phi * updated.iloc[nobs - 1])
    assert_allclose(news.total_impacts.to_numpy(), [[total]], rtol=1e-12)
    # The revision of observation 11 is shown in detail, so that the grouped
    # impact is zero up to rounding
    assert_allclose(
        news.revision_detailed_impacts.to_numpy(), [[revision]], rtol=1e-12
    )
    assert_allclose(
        news.revision_grouped_impacts.to_numpy(), [[0.0]], atol=1e-12 * revision
    )
    assert_allclose(news.update_impacts.to_numpy(), [[update]], rtol=1e-12)


def test_news_return_raw_ar1(ar1_vintages):
    res_previous, res_updated, previous, updated = ar1_vintages
    nobs = len(previous)
    phi = 0.6
    raw = res_previous.news(res_updated, impact_date=nobs + 1, return_raw=True)
    assert not isinstance(raw, NewsResults)
    assert_allclose(
        raw.post_impacted_forecasts, [[phi * updated.iloc[nobs]]], rtol=1e-12
    )
    assert_allclose(
        raw.prev_impacted_forecasts, [[phi**2 * previous.iloc[nobs - 1]]], rtol=1e-12
    )


@pytest.mark.parametrize("state_index", [[-1], [1], [0, 1]])
def test_news_state_index_invalid_raises(ar1_vintages, state_index):
    res_previous, res_updated, _, _ = ar1_vintages
    match = "negative indexes" if min(state_index) < 0 else "too large"
    with pytest.raises(ValueError, match=match):
        res_previous.news(res_updated, impact_date=13, state_index=state_index)


_VARX_COEF_AR = np.array([[0.5, 0.1], [-0.2, 0.3]])
_VARX_COEF_EXOG = np.array([0.4, -0.7])
_VARX_PARAMS = np.r_[_VARX_COEF_AR.ravel(), _VARX_COEF_EXOG, 1.0, 0.2, 1.0]


def _varx1_forecast(endog_last, exog_future):
    # Forecasts of a VARX(1), y_t = A y_{t-1} + B x_t + e_t, given the last
    # observation and the future exog
    out = []
    for value in exog_future:
        endog_last = _VARX_COEF_AR @ endog_last + _VARX_COEF_EXOG * value
        out.append(endog_last)
    return np.array(out)


@pytest.fixture
def varx1_data():
    nobs = 15
    rng = np.random.default_rng(4486)
    exog = rng.standard_normal(nobs + 3)
    endog = rng.standard_normal((nobs + 1, 2))

    def fitted(n):
        mod = varmax.VARMAX(endog[:n], exog=exog[:n], order=(1, 0), trend="n")
        return mod.smooth(_VARX_PARAMS)

    return fitted, endog, exog, nobs


@pytest.mark.parametrize("as_list", [False, True])
def test_news_comparison_with_longer_exog(varx1_data, as_list):
    # The exog of the comparison dataset has one more period than its endog,
    # so that the dataset is extended by missing values
    fitted, endog, exog, nobs = varx1_data
    res = fitted(nobs)
    comparison = endog.tolist() if as_list else endog
    news = res.news(comparison, exog=exog[: nobs + 2], impact_date=nobs + 1)
    post = _varx1_forecast(endog[nobs], exog[nobs + 1 : nobs + 2])[-1]
    prev = _varx1_forecast(endog[nobs - 1], exog[nobs : nobs + 2])[-1]
    assert_allclose(news.post_impacted_forecasts.to_numpy(), [post], rtol=1e-12)
    assert_allclose(news.prev_impacted_forecasts.to_numpy(), [prev], rtol=1e-12)
    assert_allclose(news.total_impacts.to_numpy(), [post - prev], rtol=1e-12)


@pytest.mark.parametrize("ncols", [1, 3])
def test_news_comparison_with_longer_exog_invalid_shape_raises(varx1_data, ncols):
    fitted, endog, exog, nobs = varx1_data
    res = fitted(nobs)
    comparison = endog[:, 0] if ncols == 1 else np.c_[endog, endog[:, 0]]
    with pytest.raises(ValueError, match="Invalid shape for `comparison`"):
        res.news(comparison, exog=exog[: nobs + 2], impact_date=nobs + 1)


def test_news_time_varying_extends_updated_multivariate(varx1_data):
    # A VARX(1) is time-varying through exog in the state intercept. The
    # impact date is beyond the updated sample, so that the updated results
    # are extended with the given exog.
    fitted, endog, exog, nobs = varx1_data
    res_previous = fitted(nobs)
    res_updated = fitted(nobs + 1)
    assert not res_updated.filter_results.time_invariant
    news = res_previous.news(
        res_updated, impact_date=nobs + 2, exog=exog[nobs + 1 : nobs + 3]
    )
    post = _varx1_forecast(endog[nobs], exog[nobs + 1 : nobs + 3])[-1]
    prev = _varx1_forecast(endog[nobs - 1], exog[nobs : nobs + 3])[-1]
    assert_allclose(news.post_impacted_forecasts.to_numpy(), [post], rtol=1e-10)
    assert_allclose(news.prev_impacted_forecasts.to_numpy(), [prev], rtol=1e-10)


@pytest.fixture
def tvp_vintages():
    # A regression with a random walk coefficient beta_t, which makes the
    # model time-varying. The forecast of y_d beyond the sample is
    # x_d E[beta_{n+1} | y_1, ..., y_n], the last predicted state.
    nobs = 20
    rng = np.random.default_rng(4743)
    exog = rng.standard_normal(nobs + 3)
    endog = 0.7 * exog[: nobs + 1] + rng.standard_normal(nobs + 1)

    def model(n):
        return sarimax.SARIMAX(
            endog[:n],
            exog=exog[:n],
            order=(0, 0, 0),
            time_varying_regression=True,
            mle_regression=False,
        )

    params = [0.5, 1.0]
    res_previous = model(nobs).smooth(params)
    res_updated = model(nobs + 1).smooth(params)
    return res_previous, res_updated, endog, exog


@pytest.mark.parametrize("from_updated", [False, True])
def test_news_time_varying_extends_updated(tvp_vintages, from_updated):
    # The impact date is beyond the updated sample, so that the updated
    # results are extended with the given exog
    res_previous, res_updated, _, exog = tvp_vintages
    nobs = res_previous.nobs
    impact_date = nobs + 2
    if from_updated:
        news = res_updated.news(
            res_previous, impact_date=impact_date, exog=exog[nobs + 1 :]
        )
    else:
        news = res_previous.news(
            res_updated, impact_date=impact_date, exog=exog[nobs + 1 :]
        )
    post = exog[impact_date] * res_updated.predicted_state[0, -1]
    prev = exog[impact_date] * res_previous.predicted_state[0, -1]
    assert_allclose(news.post_impacted_forecasts.to_numpy(), [[post]], rtol=1e-10)
    assert_allclose(news.prev_impacted_forecasts.to_numpy(), [[prev]], rtol=1e-10)
    assert_allclose(news.total_impacts.to_numpy(), [[post - prev]], rtol=1e-10)


def test_news_time_varying_dataset_beyond_end_raises(tvp_vintages):
    # With a dataset as the comparison and its exog, the updated results
    # cannot be extended
    res_previous, res_updated, endog, exog = tvp_vintages
    nobs = res_previous.nobs
    with pytest.raises(ValueError, match="previous dataset as the `comparison`"):
        res_updated.news(endog[:nobs], exog=exog[:nobs], impact_date=nobs + 2)


@pytest.mark.matplotlib
@pytest.mark.parametrize("use_dates", [False, True])
def test_plot_diagnostics_variable_name(close_figures, use_dates):
    # A variable given by name, a truncated name in the title, the dates or
    # the integer positions of a dataset without dates, and options for
    # plot_acf
    rng = np.random.default_rng(5389)
    endog = rng.standard_normal((40, 2))
    if use_dates:
        index = pd.date_range("2000-01-01", periods=40, freq="MS")
        columns = ["b", "a_very_long_variable_name"]
        endog = pd.DataFrame(endog, index=index, columns=columns)
        variable, i = "a_very_long_variable_name", 1
        title = 'Standardized residual for "a_very_..."'
    else:
        variable, i = "y1", 0
        title = 'Standardized residual for "y1"'
    mod = varmax.VARMAX(endog, order=(1, 0), trend="n")
    res = mod.smooth(mod.start_params)
    fig = res.plot_diagnostics(
        variable=variable, truncate_endog_names=10, acf_kwargs={"zero": False}
    )
    ax = fig.axes[0]
    assert ax.get_title() == title
    line = ax.lines[0]
    if not use_dates:
        assert_equal(np.asarray(line.get_xdata()), np.arange(res.nobs))
    assert_allclose(
        line.get_ydata(), res.filter_results.standardized_forecasts_error[i], rtol=1e-15
    )
    # Without lag zero, the correlogram starts at lag 1
    markers = [line for line in fig.axes[3].lines if line.get_marker() == "o"]
    assert_equal(markers[0].get_xdata(), np.arange(1, 11))


def test_summary_display_max_endog():
    # More variables than display_max_endog: the diagnostic tests are shown in
    # a table with one row per variable. The references are scipy and the
    # statsmodels tests, applied to the standardized residuals.
    from scipy import stats

    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test

    # Two independent local level models, without parameters
    rng = np.random.default_rng(5651)
    endog = pd.DataFrame(
        rng.standard_normal((50, 2)).cumsum(axis=0),
        columns=["first_series", "second_series"],
    )
    mod = MLEModel(
        endog,
        k_states=2,
        design=np.eye(2),
        obs_cov=np.eye(2),
        transition=np.eye(2),
        selection=np.eye(2),
        state_cov=np.eye(2),
        initialization="diffuse",
    )
    res = mod.filter([])
    summary = res.summary(
        start=3,
        truncate_endog_names=5,
        display_max_endog=1,
        extra_top_left=[("Extra left:", ["one"])],
        extra_top_right=[("Extra right:", ["two"])],
    )
    text = str(summary)
    assert re.search(r'Dep. Variable:\s+"first\.\.\.", and 1 more', text) is not None
    assert re.search(r"Sample:\s+3\s", text) is not None
    assert re.search(r"Extra left:\s+one", text) is not None
    assert re.search(r"Extra right:\s+two", text) is not None
    # No parameters, so that the diagnostics table follows the first one. The
    # column labels have two lines.
    table = summary.tables[1]
    assert table.title == "Residual diagnostics:"
    header = [[cell.data for cell in row] for row in table[:2]]
    assert header == [
        ["Residual of", "LjungBox", "Prob(Q)", "Het.(H)", "Prob(H)", "Jarque"]
        + ["Prob(JB)", "Skew", "Kurtosis"],
        ["Dep. variable", "(L1) (Q)", "", "", "", "Bera(JB)", "", "", ""],
    ]
    d = res.nobs_diffuse
    for i, name in enumerate(["first...", "secon..."]):
        resid = res.filter_results.standardized_forecasts_error[i, d:]
        lb = acorr_ljungbox(resid, lags=[1])
        het = breakvar_heteroskedasticity_test(
            resid, subset_length=int(np.round((res.nobs - d) / 3))
        )
        jb = stats.jarque_bera(resid)
        values = [
            lb["lb_stat"].iloc[0],
            lb["lb_pvalue"].iloc[0],
            het.statistic,
            het.pvalue,
            jb.statistic,
            jb.pvalue,
            stats.skew(resid),
            stats.kurtosis(resid, fisher=False),
        ]
        desired = [name] + [f"{value:.2f}" for value in values]
        assert [cell.data for cell in table[i + 2]] == desired


def test_summary_singular_cov_params_fixed_params():
    # A regressor of zeros does not change the likelihood, so that the outer
    # product of gradients is singular. The condition number in the warning
    # is that of the covariance matrix of the free parameters (inf, or huge
    # after rounding), and not NaN, that of the matrix with NaN for the fixed
    # parameter.
    rng = np.random.default_rng(5695)
    endog = rng.standard_normal(30)
    mod = sarimax.SARIMAX(endog, exog=np.zeros(30), order=(1, 0, 0))
    with mod.fix_params({"ar.L1": 0.5}):
        res = mod.filter([0.3, 1.0])
    # The standard error of the coefficient of zeros is zero
    with warns_or_none(RuntimeWarning, match="divide by zero"):
        text = str(res.summary())
    pattern = (
        r"Covariance matrix is singular or near-singular, with condition number"
        r"\s+(\S+)\.\s+Standard errors"
    )
    match = re.search(pattern, text)
    assert match is not None
    assert float(match.group(1)) > 1e10


@pytest.mark.parametrize("method", ["forecast", "get_forecast"])
@pytest.mark.parametrize(
    "nobs, steps, dtype",
    [(11, 3, np.int64), (11, 15, np.int64), (11, 15, np.int32), (300, 3, np.uint8)],
)
def test_forecast_numpy_integer_steps(method, nobs, steps, dtype):
    # A NumPy integer `steps` is the number of steps, as an int is. It was
    # used as the index of the last forecast, so that 3 raised and 15 gave 5
    # forecasts for 11 observations. With 300 observations, the number of
    # observations does not fit in a uint8. The forecasts of a zero-mean
    # AR(1) are phi**h y_T.
    endog = np.random.default_rng(2468).standard_normal(nobs)
    phi = 0.5
    res = sarimax.SARIMAX(endog, order=(1, 0, 0)).smooth([phi, 1.0])
    desired = phi ** np.arange(1, steps + 1) * endog[-1]
    actual = getattr(res, method)(dtype(steps))
    if method == "get_forecast":
        actual = actual.predicted_mean
    assert_allclose(actual, desired, rtol=1e-12)


@pytest.mark.parametrize("flags", [{}, {"approx_centered": True}])
def test_fit_flags_not_changed(flags):
    # fit added its own entries to the dictionary of flags of the caller. It
    # passes a copy with those entries to the optimizer.
    infl = macrodata.load_pandas().data["infl"].to_numpy()
    mod = sarimax.SARIMAX(infl, order=(1, 0, 0))
    original = dict(flags)
    res = mod.fit(disp=0, flags=flags)
    assert flags == original
    (used,) = res.mle_settings["fargs"]
    assert used is not flags
    assert used["transformed"] is False
    assert used.get("approx_centered") is original.get("approx_centered")
    # the flags do not change the estimates here (the score is computed by
    # complex-step differentiation, so approx_centered has no effect)
    desired = mod.fit(disp=0)
    assert_allclose(res.params, desired.params, rtol=1e-10)


@pytest.mark.parametrize("method", ["forecast", "get_forecast"])
@pytest.mark.parametrize(
    "steps",
    [3.0, 15.0, np.array(3), np.array(15)],
    ids=["3.0", "15.0", "0d-3", "0d-15"],
)
def test_forecast_int_like_steps(method, steps):
    # An integral float or a 0-d integer array `steps` is the number of
    # steps, as an int is. It was used as the index of the last forecast,
    # which raised for 3 and gave 5 forecasts for 15 (a 0-d array) or raised
    # (a float). The forecasts of a zero-mean AR(1) are phi**h y_T.
    endog = np.random.default_rng(2468).standard_normal(11)
    phi = 0.5
    res = sarimax.SARIMAX(endog, order=(1, 0, 0)).smooth([phi, 1.0])
    desired = phi ** np.arange(1, int(steps) + 1) * endog[-1]
    actual = getattr(res, method)(steps)
    if method == "get_forecast":
        actual = actual.predicted_mean
    assert_allclose(actual, desired, rtol=1e-12)


@pytest.mark.parametrize("method", ["forecast", "get_forecast"])
@pytest.mark.parametrize("steps", [True, np.True_], ids=["True", "np.True_"])
def test_forecast_bool_steps(method, steps):
    # A bool is not a number of steps: True was one step. It is the location
    # of the last forecast, which raises.
    endog = np.random.default_rng(2468).standard_normal(11)
    res = sarimax.SARIMAX(endog, order=(1, 0, 0)).smooth([0.5, 1.0])
    with pytest.raises(KeyError):
        getattr(res, method)(steps)
