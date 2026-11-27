r"""
Tests for forecasting-related features not tested elsewhere
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.tsa.statespace import sarimax, varmax


@pytest.mark.parametrize("data", ["list", "numpy", "range", "date", "period"])
def test_append_multistep(data):
    # Test that `MLEResults.append` works when called repeatedly
    endog = [1., 0.5, 1.5, 0.9, 0.2, 0.34]
    if data == "numpy":
        endog = np.array(endog)
    elif data == "range":
        endog = pd.Series(endog)
    elif data == "date":
        index = pd.date_range(start="2000-01-01", periods=6, freq="MS")
        endog = pd.Series(endog, index=index)
    elif data == "period":
        index = pd.period_range(start="2000-01", periods=6, freq="M")
        endog = pd.Series(endog, index=index)

    # Base model fitting
    mod = sarimax.SARIMAX(endog[:2], order=(1, 0, 0))
    res = mod.smooth([0.5, 1.0])
    assert_allclose(res.model.endog[:, 0], [1., 0.5])
    assert_allclose(res.forecast(1), 0.25)

    # First append
    res1 = res.append(endog[2:3])
    assert_allclose(res1.model.endog[:, 0], [1., 0.5, 1.5])
    assert_allclose(res1.forecast(1), 0.75)

    # Second append
    res2 = res1.append(endog[3:5])
    assert_allclose(res2.model.endog[:, 0], [1., 0.5, 1.5, 0.9, 0.2])
    assert_allclose(res2.forecast(1), 0.1)

    # Third append
    res3 = res2.append(endog[5:6])
    print(res3.model.endog)
    assert_allclose(res3.model.endog[:, 0], [1., 0.5, 1.5, 0.9, 0.2, 0.34])
    assert_allclose(res3.forecast(1), 0.17)


@pytest.mark.parametrize("model", ["sarimax", "varmax"])
def test_extend_kwargs_not_modified(model):
    # The defaults of the extension model (the trend offset, and for SARIMAX
    # validate_specification) are not written into the caller's
    # extend_kwargs, so reusing the dict with another model gives the
    # forecasts of that model
    from statsmodels.tsa.statespace.varmax import VARMAX

    rng = np.random.default_rng(1234)
    exog = rng.standard_normal((43, 1))
    if model == "sarimax":
        endog = 0.1 * np.arange(1, 44) + 0.5 * exog[:, 0]
        endog += rng.standard_normal(43)

        def results(nobs):
            mod = sarimax.SARIMAX(
                endog[:nobs], exog=exog[:nobs], order=(1, 0, 0), trend="t"
            )
            return mod.smooth([0.1, 0.5, 0.3, 1.0])

    else:
        endog = 0.1 * np.arange(1, 44)[:, None] + 0.5 * exog
        endog = endog + rng.standard_normal((43, 2))

        def results(nobs):
            mod = VARMAX(endog[:nobs], order=(1, 0), trend="ct", exog=exog[:nobs])
            return mod.smooth(mod.start_params)

    extend_kwargs = {}
    results(40).get_forecast(3, exog=exog[40:], extend_kwargs=extend_kwargs)
    assert extend_kwargs == {}
    res = results(30)
    actual = res.forecast(3, exog=exog[30:33], extend_kwargs=extend_kwargs)
    assert extend_kwargs == {}
    assert_allclose(actual, res.forecast(3, exog=exog[30:33]), rtol=1e-12)


@pytest.mark.parametrize("timing_init_filtered", [False, True])
def test_extend_append_apply_filtered_timing(timing_init_filtered):
    # extend initializes the new model with the predicted state of the first
    # new period. With the filtered timing convention, the new model treated
    # it as the filtered state of the period before, so the extension was one
    # period late. extend equals the last periods of the model of the whole
    # sample; append and apply refilter their whole sample.
    endog = np.array([1.0, 0.5, -0.3, 0.8, 1.2, 0.1, -0.4, 0.6, 0.9, 0.2, 0.4, 0.3])
    params = [0.5, 1.0, 1.0]
    # tolerance=0 so that the filter does not converge: with the filtered
    # timing, a converged filter leaves its last predicted state covariance,
    # from which the extension starts, at zero
    kwargs = dict(
        order=(1, 0, 0),
        measurement_error=True,
        tolerance=0,
        timing_init_filtered=timing_init_filtered,
    )
    res = sarimax.SARIMAX(endog[:10], **kwargs).smooth(params)
    full = sarimax.SARIMAX(endog, **kwargs).smooth(params)

    ext = res.extend(endog[10:])
    assert_allclose(ext.fittedvalues, full.fittedvalues[10:], rtol=1e-12)
    assert_allclose(ext.llf_obs, full.llf_obs[10:], rtol=1e-12)
    assert_allclose(ext.filtered_state, full.filtered_state[:, 10:], rtol=1e-12)
    assert_allclose(ext.smoothed_state, full.smoothed_state[:, 10:], rtol=1e-12)
    assert_allclose(
        ext.smoothed_state_cov, full.smoothed_state_cov[..., 10:], rtol=1e-12
    )

    app = res.append(endog[10:])
    assert_allclose(app.fittedvalues, full.fittedvalues, rtol=1e-12)
    assert_allclose(app.smoothed_state, full.smoothed_state, rtol=1e-12)
    app = res.append(endog[10:], copy_initialization=True)
    assert_allclose(app.fittedvalues, full.fittedvalues, rtol=1e-12)
    assert_allclose(app.smoothed_state, full.smoothed_state, rtol=1e-12)

    new = np.array([0.3, -0.2, 0.5, 0.1])
    fresh = sarimax.SARIMAX(new, **kwargs).smooth(params)
    applied = res.apply(new)
    assert_allclose(applied.fittedvalues, fresh.fittedvalues, rtol=1e-12)
    assert_allclose(applied.smoothed_state, fresh.smoothed_state, rtol=1e-12)


@pytest.mark.parametrize("timing_init_filtered", [False, True])
def test_varmax_extend_filtered_timing(timing_init_filtered):
    # VARMAXResults.extend does not call MLEResults.extend
    rng = np.random.default_rng(20261124)
    endog = rng.standard_normal((15, 2))
    params = [0.1, -0.2, 0.5, 0.1, -0.2, 0.3, 1.0, 0.2, 0.8, 0.5, 0.4]
    kwargs = dict(
        order=(1, 0),
        trend="c",
        measurement_error=True,
        tolerance=0,
        timing_init_filtered=timing_init_filtered,
    )
    res = varmax.VARMAX(endog[:12], **kwargs).smooth(params)
    full = varmax.VARMAX(endog, **kwargs).smooth(params)

    ext = res.extend(endog[12:])
    assert_allclose(ext.fittedvalues, full.fittedvalues[12:], rtol=1e-12)
    assert_allclose(ext.llf_obs, full.llf_obs[12:], rtol=1e-12)
    assert_allclose(ext.smoothed_state, full.smoothed_state[:, 12:], rtol=1e-12)
