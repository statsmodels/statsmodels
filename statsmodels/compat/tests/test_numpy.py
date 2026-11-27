from statsmodels.compat.numpy import inplace_reshape

import numpy as np
import pytest

from statsmodels.tsa.statespace.kalman_filter import KalmanFilter


def test_inplace_reshape_leaves_input_unchanged():
    # On NumPy < 2.1 the shape of the caller's array was changed
    arr = np.arange(6.0)
    out = inplace_reshape(arr, (6, 1))
    assert out.shape == (6, 1)
    assert arr.shape == (6,)
    assert np.shares_memory(out, arr)


def test_inplace_reshape_raises_without_copy():
    # The transpose of a C-ordered matrix cannot be flattened without a copy
    arr = np.arange(12.0).reshape(3, 4).T
    with pytest.raises((ValueError, AttributeError)):
        inplace_reshape(arr, (12,))
    assert arr.shape == (4, 3)


@pytest.mark.parametrize("k_endog, nobs", [(1, 10), (2, 1)])
def test_bind_leaves_endog_unchanged(k_endog, nobs):
    # Binding a 1-dimensional array (nobs observations of one series, or one
    # observation of k_endog series) reshaped the caller's array to 2
    # dimensions on NumPy < 2.1, also in clone
    mod = KalmanFilter(k_endog=k_endog, k_states=1)
    endog = np.arange(float(nobs * k_endog))
    mod.bind(endog)
    assert endog.shape == (nobs * k_endog,)
    assert mod.nobs == nobs
    assert mod.endog.shape == (k_endog, nobs)
