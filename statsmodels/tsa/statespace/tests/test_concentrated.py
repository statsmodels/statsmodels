"""
Tests for concentrating the scale out of the loglikelihood function

Note: the univariate cases is well tested in test_sarimax.py

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.tools.tools import Bunch
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_CONSERVE,
    MEMORY_NO_FORECAST_COV,
    MEMORY_NO_LIKELIHOOD,
)
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

from .results import results_varmax


def get_sarimax_models(endog, filter_univariate=False, **kwargs):
    kwargs.setdefault("tolerance", 0)
    # Construct a concentrated version of the given SARIMAX model, and get
    # the estimate of the scale
    mod_conc = sarimax.SARIMAX(endog, **kwargs)
    mod_conc.ssm.filter_concentrated = True
    mod_conc.ssm.filter_univariate = filter_univariate
    params_conc = mod_conc.start_params
    params_conc[-1] = 1
    res_conc = mod_conc.smooth(params_conc)
    scale = res_conc.scale

    # Construct the non-concentrated version
    mod_orig = sarimax.SARIMAX(endog, **kwargs)
    mod_orig.ssm.filter_univariate = filter_univariate
    params_orig = params_conc.copy()
    k_vars = 1 + kwargs.get("measurement_error", False)
    params_orig[-k_vars:] = scale * params_conc[-k_vars:]
    res_orig = mod_orig.smooth(params_orig)

    return Bunch(
        mod_conc=mod_conc,
        params_conc=params_conc,
        mod_orig=mod_orig,
        params_orig=params_orig,
        res_conc=res_conc,
        res_orig=res_orig,
        scale=scale,
    )


def test_concentrated_loglike_sarimax():
    # Note: we will not use the "concentrate_scale" option to SARIMAX for this
    # test, which is a lower-level test of the Kalman filter using the SARIMAX
    # model as an example
    nobs = 30
    rs = np.random.RandomState(28953)
    endog = rs.normal(size=nobs)
    kwargs = {}

    # Typical model
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs, out.res_orig.llf_obs)
    assert_allclose(
        out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig)
    )
    assert_allclose(
        out.mod_conc.loglikeobs(out.params_conc),
        out.mod_orig.loglikeobs(out.params_orig),
    )

    # Add missing entries
    endog[2:10] = np.nan
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs, out.res_orig.llf_obs)
    assert_allclose(
        out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig)
    )
    assert_allclose(
        out.mod_conc.loglikeobs(out.params_conc),
        out.mod_orig.loglikeobs(out.params_orig),
    )

    # Add seasonal differencing
    # Note: now, due to differences in approximate diffuse initialization,
    # we may have differences in the first 2 observations (but notice that
    # this does not affect the computed joint log-likelihood because those
    # observations are not included there)
    kwargs["seasonal_order"] = (1, 1, 1, 2)
    out = get_sarimax_models(endog, **kwargs)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs[2:], out.res_orig.llf_obs[2:])
    assert_allclose(
        out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig)
    )
    assert_allclose(
        out.mod_conc.loglikeobs(out.params_conc)[2:],
        out.mod_orig.loglikeobs(out.params_orig)[2:],
    )

    # Add loglikelihood burn, trend, and exog
    kwargs["loglikelihood_burn"] = 5
    kwargs["trend"] = "c"
    kwargs["exog"] = np.arange(nobs)
    out = get_sarimax_models(endog, **kwargs)
    assert_allclose(out.res_conc.llf, out.res_orig.llf)
    assert_allclose(out.res_conc.llf_obs[2:], out.res_orig.llf_obs[2:])
    assert_allclose(
        out.mod_conc.loglike(out.params_conc), out.mod_orig.loglike(out.params_orig)
    )
    assert_allclose(
        out.mod_conc.loglikeobs(out.params_conc)[2:],
        out.mod_orig.loglikeobs(out.params_orig)[2:],
    )


def test_concentrated_predict_sarimax():
    # Note: we will not use the "concentrate_scale" option to SARIMAX for this
    # test, which is a lower-level test of the Kalman filter using the SARIMAX
    # model as an example
    nobs = 30
    rs = np.random.RandomState(28953)
    endog = rs.normal(size=nobs)

    # Typical model
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.predict(), out.res_orig.predict())
    assert_allclose(out.res_conc.forecast(5), out.res_orig.forecast(5))
    assert_allclose(
        out.res_conc.predict(start=0, end=45, dynamic=10),
        out.res_orig.predict(start=0, end=45, dynamic=10),
    )


def test_fixed_scale_sarimax():
    # Test that the fixed_scale context manager works
    nobs = 30
    rs = np.random.RandomState(28953)
    endog = rs.normal(size=nobs)
    kwargs = {
        "seasonal_order": (1, 1, 1, 2),
        "trend": "ct",
        "exog": np.sin(np.arange(nobs)),
    }

    # Construct a concentrated version of the given SARIMAX model
    mod_conc = sarimax.SARIMAX(endog, concentrate_scale=True, **kwargs)

    # Construct the non-concentrated version
    mod_orig = sarimax.SARIMAX(endog, **kwargs)

    # Modify the scale parameter
    params = mod_orig.start_params
    params[-1] *= 1.2

    # Test that these are not equal in the default computation
    with pytest.raises(AssertionError):
        assert_allclose(mod_conc.loglike(params[:-1]), mod_orig.loglike(params))

    # Now test that the llf is equal when we use the fixed_scale decorator
    with mod_conc.ssm.fixed_scale(params[-1]):
        res1 = mod_conc.smooth(params[:-1])
        llf1 = mod_conc.loglike(params[:-1])
        llf_obs1 = mod_conc.loglikeobs(params[:-1])

    res2 = mod_orig.smooth(params)
    llf2 = mod_orig.loglike(params)
    llf_obs2 = mod_orig.loglikeobs(params)

    assert_allclose(res1.llf, res2.llf)
    assert_allclose(res1.llf_obs[2:], res2.llf_obs[2:])
    assert_allclose(llf1, llf2)
    assert_allclose(llf_obs1, llf_obs2)


def check_concentrated_scale(filter_univariate=False, missing=False, **kwargs):
    # Test that concentrating the scale out of the likelihood function works
    index = pd.date_range("1960-01-01", "1982-10-01", freq="QS")
    dta = pd.DataFrame(
        results_varmax.lutkepohl_data, columns=["inv", "inc", "consump"], index=index
    )
    dta["dln_inv"] = np.log(dta["inv"]).diff()
    dta["dln_inc"] = np.log(dta["inc"]).diff()
    dta["dln_consump"] = np.log(dta["consump"]).diff()

    endog = dta.loc["1960-04-01":"1978-10-01", ["dln_inv", "dln_inc"]]

    # Optionally add some missing observations
    if missing:
        endog.iloc[0, 0] = np.nan
        endog.iloc[3:5, :] = np.nan
        endog.iloc[8, 1] = np.nan

    # Sometimes we can have slight differences if the Kalman filters
    # converge at different observations, so disable convergence.
    kwargs.update({"tolerance": 0})

    mod_orig = varmax.VARMAX(endog, **kwargs)
    mod_conc = varmax.VARMAX(endog, **kwargs)
    mod_conc.ssm.filter_concentrated = True

    mod_orig.ssm.filter_univariate = filter_univariate
    mod_conc.ssm.filter_univariate = filter_univariate

    # Since VARMAX does not explicitly allow concentrating out the scale, for
    # now we will simulate it by setting the first variance to be 1.
    # Note that start_scale will not be the scale used for the non-concentrated
    # model, because we need to use the MLE scale estimated by the
    # concentrated model.
    conc_params = mod_conc.start_params
    start_scale = conc_params[mod_conc._params_state_cov][0]
    if kwargs.get("error_cov_type", "unstructured") == "diagonal":
        conc_params[mod_conc._params_state_cov] /= start_scale
    else:
        conc_params[mod_conc._params_state_cov] /= start_scale**0.5

    # Concentrated model smoothing
    res_conc = mod_conc.smooth(conc_params)
    scale = res_conc.scale

    # Map the concentrated parameters to the non-concentrated model
    orig_params = conc_params.copy()
    if kwargs.get("error_cov_type", "unstructured") == "diagonal":
        orig_params[mod_orig._params_state_cov] *= scale
    else:
        orig_params[mod_orig._params_state_cov] *= scale**0.5

    # Measurement error variances also get rescaled
    orig_params[mod_orig._params_obs_cov] *= scale

    # Non-oncentrated model smoothing
    res_orig = mod_orig.smooth(orig_params)

    # Test loglike
    # Need to reduce the tolerance when we have measurement error.
    assert_allclose(res_conc.llf, res_orig.llf)

    # Test state space representation matrices
    for name in mod_conc.ssm.shapes:
        if name == "obs":
            continue
        assert_allclose(
            getattr(res_conc.filter_results, name),
            getattr(res_orig.filter_results, name),
        )

    # Test filter / smoother output
    scale = res_conc.scale
    d = res_conc.loglikelihood_burn

    filter_attr = [
        "predicted_state",
        "filtered_state",
        "forecasts",
        "forecasts_error",
        "kalman_gain",
    ]

    for name in filter_attr:
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-7)

    # Note: do not want to compare the elements from any diffuse
    # initialization for things like covariances, so only compare for
    # periods past the loglikelihood_burn period
    filter_attr_burn = [
        "standardized_forecasts_error",
        "predicted_state_cov",
        "filtered_state_cov",
        "tmp1",
        "tmp2",
        "tmp3",
        "tmp4",
    ]

    for name in filter_attr_burn:
        actual = getattr(res_conc.filter_results, name)[..., d:]
        desired = getattr(res_orig.filter_results, name)[..., d:]
        assert_allclose(actual, desired, atol=1e-7)

    smoothed_attr = [
        "smoothed_state",
        "smoothed_state_cov",
        "smoothed_state_autocov",
        "smoothed_state_disturbance",
        "smoothed_state_disturbance_cov",
        "smoothed_measurement_disturbance",
        "smoothed_measurement_disturbance_cov",
        "scaled_smoothed_estimator",
        "scaled_smoothed_estimator_cov",
        "smoothing_error",
        "smoothed_forecasts",
        "smoothed_forecasts_error",
        "smoothed_forecasts_error_cov",
    ]

    for name in smoothed_attr:
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired, atol=1e-7)

    # Test prediction output
    nobs = mod_conc.nobs
    pred_conc = res_conc.get_prediction(start=10, end=nobs + 50, dynamic=40)
    pred_orig = res_conc.get_prediction(start=10, end=nobs + 50, dynamic=40)

    assert_allclose(pred_conc.predicted_mean, pred_orig.predicted_mean)
    assert_allclose(pred_conc.se_mean, pred_orig.se_mean)


def test_concentrated_scale_conventional():
    check_concentrated_scale(filter_univariate=False)
    check_concentrated_scale(filter_univariate=False, measurement_error=True)
    check_concentrated_scale(filter_univariate=False, error_cov_type="diagonal")
    check_concentrated_scale(filter_univariate=False, missing=True)
    check_concentrated_scale(
        filter_univariate=False, missing=True, loglikelihood_burn=10
    )


def test_concentrated_scale_univariate():
    check_concentrated_scale(filter_univariate=True)
    check_concentrated_scale(filter_univariate=True, measurement_error=True)
    check_concentrated_scale(filter_univariate=True, error_cov_type="diagonal")
    check_concentrated_scale(filter_univariate=True, missing=True)
    check_concentrated_scale(
        filter_univariate=True, missing=True, loglikelihood_burn=10
    )


@pytest.mark.parametrize("univariate", [False, True])
@pytest.mark.parametrize("burn", [0, 1])
@pytest.mark.parametrize("memory", [0, MEMORY_NO_FORECAST_COV,
                                  MEMORY_NO_LIKELIHOOD, MEMORY_CONSERVE])
def test_concentrated_exact_diffuse_closed_form(univariate, burn, memory):
    # Independent scalar diffuse recursion: the first innovation is diffuse
    # (F_inf=1); the second has F=7/4 and v=-1/2. At t=1 the two finite
    # innovations have F=97/56, v=-9/14 and F=623/388, v=79/97.
    # Their standardized squared errors sum to 58/89, so scale=29/89.
    # Include log(2*pi) for the diffuse innovation, as in the existing
    # exact-diffuse test convention (Durbin and Koopman, section 7.2.5).
    mod = KalmanSmoother(2, 1)
    mod.bind(np.asfortranarray([[1., 0.], [0., 1.]]))
    mod["design"] = [[1.], [0.5]]
    mod["obs_cov"] = np.diag([1., 1.5])
    mod["transition"] = [[0.75]]
    mod["selection"] = [[1.]]
    mod["state_cov"] = [[0.25]]
    mod.initialize_diffuse()
    mod.filter_concentrated = True
    mod.filter_univariate = univariate
    mod.loglikelihood_burn = burn
    expected = -4.004213273294209 if burn == 0 else -2.227986760129886
    assert_allclose(mod.loglike(conserve_memory=memory), expected,
                    rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("memory", [0, MEMORY_NO_FORECAST_COV,
                                   MEMORY_NO_LIKELIHOOD, MEMORY_CONSERVE])
def test_concentrated_exact_diffuse_singular_innovation(memory):
    # A deterministic zero observation contributes no likelihood. Burning
    # one leading row ends inside the diffuse period, and must not count its
    # zero finite forecast variance as an ordinary innovation.
    endog = np.array([[np.nan, np.nan, 0.], [np.nan, np.nan, 0.],
                      [1., 0., 0.], [0., 1., 0.], [2., -1., 0.]])
    mod = KalmanSmoother(3, 1, tolerance=0)
    mod.bind(np.asfortranarray(endog.T))
    mod["design"] = [[1.], [0.5], [0.]]
    mod["obs_cov"] = np.diag([1., 1.5, 0.])
    mod["transition"] = [[0.75]]
    mod["selection"] = [[1.]]
    mod["state_cov"] = [[0.25]]
    mod.initialize_diffuse()
    mod.filter_concentrated = True
    mod.filter_univariate = True
    mod.loglikelihood_burn = 1
    scale = mod.smooth().scale
    with mod.fixed_scale(scale):
        expected = mod.loglike(conserve_memory=0)
    assert_allclose(mod.loglike(conserve_memory=memory), expected,
                    rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("burn", [0, 2])
@pytest.mark.parametrize("memory", [0, MEMORY_NO_LIKELIHOOD, MEMORY_CONSERVE])
def test_concentrated_stationary_loglike_memory(burn, memory):
    # Preserve the non-diffuse concentrated likelihood with and without
    # individual likelihood storage, including after likelihood burn.
    mod = KalmanSmoother(1, 1, tolerance=0)
    mod.bind(np.array([[1., 0., -1., 2., 0., 1.]]))
    mod["design"] = [[1.]]
    mod["obs_cov"] = [[1.]]
    mod["transition"] = [[0.75]]
    mod["selection"] = [[1.]]
    mod["state_cov"] = [[0.25]]
    mod.initialize_stationary()
    mod.filter_concentrated = True
    mod.loglikelihood_burn = burn
    scale = mod.smooth().scale
    with mod.fixed_scale(scale):
        expected = mod.loglike(conserve_memory=0)
    assert_allclose(mod.loglike(conserve_memory=memory), expected,
                    rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("univariate", [False, True])
@pytest.mark.parametrize("burn", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("memory", [0, MEMORY_NO_FORECAST_COV,
                                  MEMORY_NO_LIKELIHOOD, MEMORY_CONSERVE])
def test_concentrated_exact_diffuse_loglike(univariate, burn, memory):
    # Leading missing data make burn=1/2 fall inside the diffuse period.
    # At the first nonmissing period, one innovation is diffuse and the
    # other is finite. Compare with a nonconcentrated filter at the SAME
    # fitted scale; this does not redefine the scale estimator.
    endog = np.array([[np.nan, np.nan], [np.nan, np.nan], [1., 0.],
                      [0., 1.], [np.nan, 2.], [3., np.nan],
                      [np.nan, np.nan], [1., -1.], [2., 0.]])
    mod = KalmanSmoother(2, 1, tolerance=0)
    mod.bind(np.asfortranarray(endog.T))
    mod["design"] = [[1.], [0.5]]
    mod["obs_cov"] = np.diag([1., 1.5])
    mod["transition"] = [[0.75]]
    mod["selection"] = [[1.]]
    mod["state_cov"] = [[0.25]]
    mod.initialize_diffuse()
    mod.filter_concentrated = True
    mod.filter_univariate = univariate
    mod.loglikelihood_burn = burn
    result = mod.smooth()
    observed = mod.loglikeobs()
    with mod.fixed_scale(result.scale):
        expected = mod.loglike(conserve_memory=0)
    assert_allclose(result.llf, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(observed.sum(), expected, rtol=1e-12, atol=1e-12)
    assert_allclose(mod.loglike(conserve_memory=memory), expected,
                    rtol=1e-12, atol=1e-12)
