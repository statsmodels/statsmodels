from statsmodels.compat.scipy import _mquantiles, _mquantiles_numpy, _next_regular

import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

PROBS = np.array([0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1])

# (alphap, betap) and the equivalent numpy.quantile method (Hyndman & Fan
# types 4 - 9)
HF_METHODS = [
    ((0.0, 1.0), "interpolated_inverted_cdf"),
    ((0.5, 0.5), "hazen"),
    ((0.0, 0.0), "weibull"),
    ((1.0, 1.0), "linear"),
    ((1 / 3, 1 / 3), "median_unbiased"),
    ((3 / 8, 3 / 8), "normal_unbiased"),
]


@pytest.fixture(params=[_mquantiles, _mquantiles_numpy], ids=["dispatch", "numpy"])
def mquantiles(request):
    return request.param


def test_next_regular():
    hams = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 8,
        8: 8,
        14: 15,
        15: 15,
        16: 16,
        17: 18,
        1021: 1024,
        1536: 1536,
        51200000: 51200000,
        510183360: 510183360,
        510183360 + 1: 512000000,
        511000000: 512000000,
        854296875: 854296875,
        854296875 + 1: 859963392,
        196608000000: 196608000000,
        196608000000 + 1: 196830000000,
        8789062500000: 8789062500000,
        8789062500000 + 1: 8796093022208,
        206391214080000: 206391214080000,
        206391214080000 + 1: 206624260800000,
        470184984576000: 470184984576000,
        470184984576000 + 1: 470715894135000,
        7222041363087360: 7222041363087360,
        7222041363087360 + 1: 7230196133913600,
        # power of 5    5**23
        11920928955078125: 11920928955078125,
        11920928955078125 - 1: 11920928955078125,
        # power of 3    3**34
        16677181699666569: 16677181699666569,
        16677181699666569 - 1: 16677181699666569,
        # power of 2   2**54
        18014398509481984: 18014398509481984,
        18014398509481984 - 1: 18014398509481984,
        # above this, int(ceil(n)) == int(ceil(n+1))
        19200000000000000: 19200000000000000,
        19200000000000000 + 1: 19221679687500000,
        288230376151711744: 288230376151711744,
        288230376151711744 + 1: 288325195312500000,
        288325195312500000 - 1: 288325195312500000,
        288325195312500000: 288325195312500000,
        288325195312500000 + 1: 288555831593533440,
        # power of 3    3**83
        3**83 - 1: 3**83,
        3**83: 3**83,
        # power of 2     2**135
        2**135 - 1: 2**135,
        2**135: 2**135,
        # power of 5      5**57
        5**57 - 1: 5**57,
        5**57: 5**57,
        # http://www.drdobbs.com/228700538
        # 2**96 * 3**1 * 5**13
        2**96 * 3**1 * 5**13 - 1: 2**96 * 3**1 * 5**13,
        2**96 * 3**1 * 5**13: 2**96 * 3**1 * 5**13,
        2**96 * 3**1 * 5**13 + 1: 2**43 * 3**11 * 5**29,
        # 2**36 * 3**69 * 5**7
        2**36 * 3**69 * 5**7 - 1: 2**36 * 3**69 * 5**7,
        2**36 * 3**69 * 5**7: 2**36 * 3**69 * 5**7,
        2**36 * 3**69 * 5**7 + 1: 2**90 * 3**32 * 5**9,
        # 2**37 * 3**44 * 5**42
        2**37 * 3**44 * 5**42 - 1: 2**37 * 3**44 * 5**42,
        2**37 * 3**44 * 5**42: 2**37 * 3**44 * 5**42,
        2**37 * 3**44 * 5**42 + 1: 2**20 * 3**106 * 5**7,
    }

    for x, y in hams.items():
        assert_equal(_next_regular(x), y)


def test_mquantiles_reference(mquantiles):
    # Reference values from the scipy.stats.mstats.mquantiles docstring
    a = [6.0, 47.0, 49.0, 15.0, 42.0, 41.0, 7.0, 39.0, 43.0, 40.0, 36.0]
    res = mquantiles(a, [0.25, 0.5, 0.75])
    assert isinstance(res, np.ndarray)
    assert not isinstance(res, np.ma.MaskedArray)
    assert_allclose(res, [19.2, 40.0, 42.8])


def test_mquantiles_default_formula(mquantiles):
    # Direct implementation of the definition in mquantiles
    data = np.random.default_rng(20260925).standard_normal(37)
    x = np.sort(data)
    n = x.shape[0]
    alphap = betap = 0.4
    m = alphap + PROBS * (1 - alphap - betap)
    aleph = n * PROBS + m
    k = np.floor(np.clip(aleph, 1, n - 1)).astype(int)
    gamma = np.clip(aleph - k, 0, 1)
    expected = (1 - gamma) * x[k - 1] + gamma * x[k]
    assert_allclose(mquantiles(data, PROBS), expected, rtol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 10, 37])
@pytest.mark.parametrize("params, method", HF_METHODS)
def test_mquantiles_hyndman_fan(mquantiles, params, method, n):
    x = np.random.default_rng(n).standard_normal(n)
    alphap, betap = params
    res = mquantiles(x, PROBS, alphap=alphap, betap=betap)
    expected = np.quantile(x, PROBS, method=method)
    assert_allclose(res, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 37])
@pytest.mark.parametrize(
    "alphap, betap", [(0.4, 0.4), (0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.2, 0.7)]
)
def test_mquantiles_numpy_vs_scipy(n, alphap, betap):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scipy.stats.mstats import mquantiles
    except ImportError:
        pytest.skip("scipy.stats.mstats is not available")
    x = np.random.default_rng(n).standard_normal(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = np.asarray(mquantiles(x, PROBS, alphap=alphap, betap=betap))
    res = _mquantiles_numpy(x, PROBS, alphap=alphap, betap=betap)
    assert_allclose(res, expected, rtol=1e-12, atol=1e-14)


def test_mquantiles_single_observation(mquantiles):
    res = mquantiles([3.0], PROBS)
    assert_equal(res, np.full(PROBS.shape, 3.0))


def test_mquantiles_scalar_p(mquantiles):
    data = np.random.default_rng(0).standard_normal(37)
    res = mquantiles(data, 0.5)
    assert np.ndim(res) == 0
    assert_allclose(res, np.median(data))


def test_mquantiles_extremes(mquantiles):
    data = np.random.default_rng(0).standard_normal(37)
    res = mquantiles(data, [0.0, 1.0])
    assert_equal(res, [data.min(), data.max()])


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_mquantiles_axis(mquantiles, axis):
    x = np.random.default_rng(0).standard_normal((20, 3))
    p = [0.1, 0.25, 0.5, 0.75, 0.9]
    cols = np.moveaxis(x, axis, 0)
    expected = np.column_stack(
        [_mquantiles_numpy(cols[:, i], p) for i in range(cols.shape[1])]
    )
    res = mquantiles(x, p, axis=axis)
    assert res.shape == (5, cols.shape[1])
    assert_allclose(res, expected, rtol=1e-12)

    res = mquantiles(x, 0.25, axis=axis)
    assert res.shape == (cols.shape[1],)
    assert_allclose(res, expected[1], rtol=1e-12)


def test_mquantiles_axis_none(mquantiles):
    x = np.random.default_rng(0).standard_normal((20, 3))
    expected = _mquantiles_numpy(x.ravel(), PROBS)
    assert_allclose(mquantiles(x, PROBS), expected, rtol=1e-12)
    assert_allclose(mquantiles(x, PROBS, axis=None), expected, rtol=1e-12)
