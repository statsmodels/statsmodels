"""
Tests for ARMA innovations algorithm
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.innovations import _arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX


def test_brockwell_davis_ex533():
    # See Brockwell and Davis (2009) - Time Series Theory and Methods
    # Example 5.3.3: ARMA(1, 1) process, p.g. 177

    ar_params = np.array([0.2])
    ma_params = np.array([0.4])
    sigma2 = 8.92

    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    # First, get the autocovariance of the process
    arma_process_acovf = arma_acovf(ar, ma, nobs=10, sigma2=sigma2)
    unconditional_variance = (
        sigma2 * (1 + 2 * ar_params[0] * ma_params[0] + ma_params[0]**2) /
        (1 - ar_params[0]**2))
    assert_allclose(arma_process_acovf[0], unconditional_variance)

    # Next, get the autocovariance of the transformed process
    # Note: as required by {{prefix}}arma_transformed_acovf, we first divide
    # through by sigma^2
    arma_process_acovf /= sigma2
    unconditional_variance /= sigma2
    acovf = np.array(_arma_innovations.darma_transformed_acovf(
        ar, ma, arma_process_acovf))
    assert_allclose(np.diagonal(acovf)[1:], 1 + ma_params[0]**2)
    ix = np.diag_indices_from(acovf)
    ix_lower = (ix[0][:-1] + 1, ix[1][:-1])
    ix_upper = (ix[0][:-1], ix[1][:-1] + 1)
    assert_allclose(acovf[ix][0], unconditional_variance)
    assert_allclose(acovf[ix][1:], 1 + ma_params[0]**2)
    assert_allclose(acovf[ix_lower], ma_params[0])
    assert_allclose(acovf[ix_upper], ma_params[0])

    out = _arma_innovations.darma_innovations_algo(ar_params, ma_params, acovf)
    theta = np.array(out[0])
    v = np.array(out[1])

    # Test v (see eq. 5.3.13)
    desired_v = np.zeros(10)
    desired_v[0] = unconditional_variance
    for i in range(1, 10):
        desired_v[i] = 1 + (1 - 1 / desired_v[i - 1]) * ma_params[0]**2
    assert_allclose(v, desired_v)

    # Test theta (see eq. 5.3.13)
    desired_theta = np.zeros(10)
    for i in range(1, 10):
        desired_theta[i] = ma_params[0] / desired_v[i - 1]
    assert_allclose(theta[:, 0], desired_theta)
    assert_allclose(theta[:, 1:], 0, atol=1e-12)

    # Test against Table 5.3.1
    endog = np.array([
        -1.1, 0.514, 0.116, -0.845, 0.872, -0.467, -0.977, -1.699, -1.228,
        -1.093])
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params,
                                                   theta)

    # Note: Table 5.3.1 has \hat X_n+1 = -0.5340 for n = 1, but this seems to
    # be a typo, since equation 5.3.12 gives the form of the prediction
    # equation as \hat X_n+1 = \phi X_n + \theta_n1 (X_n - \hat X_n)
    # Then for n = 1 we have:
    # \hat X_n+1 = 0.2 (-1.1) + (0.2909) (-1.1 - 0) = -0.5399
    # And for n = 2 if we use what we have computed, then we get:
    # \hat X_n+1 = 0.2 (0.514) + (0.3833) (0.514 - (-0.54)) = 0.5068
    # as desired, whereas if we used the book's number for n=1 we would get:
    # \hat X_n+1 = 0.2 (0.514) + (0.3833) (0.514 - (-0.534)) = 0.5045
    # which is not what Table 5.3.1 shows.
    desired_hat = np.array([
        0, -0.540, 0.5068, -0.1321, -0.4539, 0.7046, -0.5620, -0.3614,
        -0.8748, -0.3869])
    desired_u = endog - desired_hat
    assert_allclose(u, desired_u, atol=1e-4)


def test_brockwell_davis_ex534():
    # See Brockwell and Davis (2009) - Time Series Theory and Methods
    # Example 5.3.4: ARMA(1, 1) process, p.g. 178

    ar_params = np.array([1, -0.24])
    ma_params = np.array([0.4, 0.2, 0.1])
    sigma2 = 1

    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    # First, get the autocovariance of the process
    arma_process_acovf = arma_acovf(ar, ma, nobs=10, sigma2=sigma2)
    assert_allclose(arma_process_acovf[:3],
                    [7.17133, 6.44139, 5.06027], atol=1e-5)

    # Next, get the autocovariance of the transformed process
    acovf = np.array(_arma_innovations.darma_transformed_acovf(
        ar, ma, arma_process_acovf))
    ix = np.diag_indices_from(acovf)
    ix_lower1 = (ix[0][:-1] + 1, ix[1][:-1])
    ix_lower2 = (ix[0][:-2] + 2, ix[1][:-2])
    ix_lower3 = (ix[0][:-3] + 3, ix[1][:-3])
    ix_lower4 = (ix[0][:-4] + 4, ix[1][:-4])

    # Test along diagonals and off diagonals (see 5.3.14)
    assert_allclose(acovf[ix][:3], 7.17133, atol=1e-5)
    assert_allclose(acovf[ix][3:], 1.21)

    desired = [6.44139, 6.44139, 0.816]
    assert_allclose(acovf[ix_lower1][:3], desired, atol=1e-5)
    assert_allclose(acovf[ix_lower1][3:], 0.5, atol=1e-5)

    assert_allclose(acovf[ix_lower2][0], 5.06027, atol=1e-5)
    assert_allclose(acovf[ix_lower2][1:3], 0.34, atol=1e-5)
    assert_allclose(acovf[ix_lower2][3:], 0.24, atol=1e-5)

    assert_allclose(acovf[ix_lower3], 0.1, atol=1e-5)

    assert_allclose(acovf[ix_lower4], 0, atol=1e-5)

    # Test innovations algorithm output
    out = _arma_innovations.darma_innovations_algo(ar_params, ma_params, acovf)
    theta = np.array(out[0])
    v = np.array(out[1])

    # Test v (see Table 5.3.2)
    desired_v = [7.1713, 1.3856, 1.0057, 1.0019, 1.0016, 1.0005, 1.0000,
                 1.0000, 1.0000, 1.0000]
    assert_allclose(v, desired_v, atol=1e-4)

    # Test theta (see Table 5.3.2)
    desired_theta = np.array([
        [0, 0.8982, 1.3685, 0.4008, 0.3998, 0.3992, 0.4000, 0.4000, 0.4000,
         0.4000],
        [0, 0, 0.7056, 0.1806, 0.2020, 0.1995, 0.1997, 0.2000, 0.2000, 0.2000],
        [0, 0, 0, 0.0139, 0.0722, 0.0994, 0.0998, 0.0998, 0.0999, 0.1]]).T
    assert_allclose(theta[:, :3], desired_theta, atol=1e-4)
    assert_allclose(theta[:, 3:], 0, atol=1e-12)

    # Test innovations filter output
    endog = np.array([1.704, 0.527, 1.041, 0.942, 0.555, -1.002, -0.585, 0.010,
                      -0.638, 0.525])
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params,
                                                   theta)

    desired_hat = np.array([
        0, 1.5305, -0.1710, 1.2428, 0.7443, 0.3138, -1.7293, -0.1688,
        0.3193, -0.8731])
    desired_u = endog - desired_hat
    assert_allclose(u, desired_u, atol=1e-4)


@pytest.mark.parametrize("ar_params,ma_params,sigma2", [
    (np.array([]), np.array([]), 1),
    (np.array([0.]), np.array([0.]), 1),
    (np.array([0.9]), np.array([]), 1),
    (np.array([]), np.array([0.9]), 1),
    (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123),
    (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123),
])
def test_innovations_algo_filter_kalman_filter(ar_params, ma_params, sigma2):
    # Test the innovations algorithm and filter against the Kalman filter
    # for exact likelihood evaluation of an ARMA process
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    endog = np.random.normal(size=10)

    # Innovations algorithm approach
    arma_process_acovf = arma_acovf(ar, ma, nobs=len(endog), sigma2=sigma2)
    acovf = np.array(_arma_innovations.darma_transformed_acovf(
                     ar, ma, arma_process_acovf / sigma2))
    theta, r = _arma_innovations.darma_innovations_algo(ar_params, ma_params,
                                                        acovf)
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params,
                                                   theta)

    v = np.array(r) * sigma2
    u = np.array(u)

    llf_obs = -0.5 * u**2 / v - 0.5 * np.log(2 * np.pi * v)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    assert_allclose(u, res.forecasts_error[0])
    # assert_allclose(theta[1:, 0], res.filter_results.kalman_gain[0, 0, :-1])
    assert_allclose(llf_obs, res.llf_obs)

    # Get llf_obs directly
    llf_obs2 = _arma_innovations.darma_loglikeobs(endog, ar_params, ma_params,
                                                  sigma2)

    assert_allclose(llf_obs2, res.llf_obs)
