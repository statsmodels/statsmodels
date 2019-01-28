import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.doe import latin


def test_lhs():
    np.random.seed(123456)

    corners = np.array([[0, 2], [10, 5]])

    sample = latin.latin_hypercube(dim=2, n_samples=5, bounds=corners)
    out = np.array([[5.746, 3.219], [5.479, 3.261], [9.246, 4.798],
                    [9.097, 4.495], [9.753, 4.074]])
    assert_almost_equal(sample, out, decimal=1)

    sample = latin.latin_hypercube(dim=2, n_samples=5, centered=True)
    out = np.array([[0.3, 0.9], [0.7, 0.7], [0.1, 0.9],
                    [0.5, 0.5], [0.1, 0.7]])
    assert_almost_equal(sample, out, decimal=1)


def test_orthogonal_lhs():
    np.random.seed(123456)

    corners = np.array([[0, 2], [10, 5]])

    sample = latin.orthogonal_latin_hypercube(2, 5, bounds=corners)
    out = np.array([[3.933, 2.670], [7.794, 4.031], [4.520, 2.129],
                    [0.253, 4.976], [8.753, 3.249]])
    assert_almost_equal(sample, out, decimal=1)

    # Checking independency of the random numbers generated
    n_samples = 500
    sample = latin.orthogonal_latin_hypercube(dim=2, n_samples=n_samples)
    min_b = 50  # number of bins
    bins = np.linspace(0, 1, min(min_b, n_samples) + 1)
    hist = np.histogram(sample[:, 0], bins=bins)
    out = np.array([n_samples / min_b] * min_b)
    assert_equal(hist[0], out)

    hist = np.histogram(sample[:, 1], bins=bins)
    assert_equal(hist[0], out)


@pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
def test_optimal_design():
    np.random.seed(123456)

    start_design = latin.orthogonal_latin_hypercube(2, 5)
    sample = latin.optimal_design(dim=2, n_samples=5,
                                  start_design=start_design)
    out = np.array([[0.025, 0.223], [0.779, 0.677], [0.452, 0.043],
                    [0.393, 0.992], [0.875, 0.416]])
    assert_almost_equal(sample, out, decimal=1)

    corners = np.array([[0, 2], [10, 5]])
    sample = latin.optimal_design(2, 5, bounds=corners)
    out = np.array([[5.189, 4.604], [3.553, 2.344], [6.275, 3.947],
                    [0.457, 3.554], [9.705, 2.636]])
    assert_almost_equal(sample, out, decimal=1)

    sample = latin.optimal_design(2, 5, niter=2)
    out = np.array([[0.681, 0.231], [0.007, 0.719], [0.372, 0.101],
                    [0.550, 0.456], [0.868, 0.845]])
    assert_almost_equal(sample, out, decimal=1)

    sample = latin.optimal_design(2, 5, bounds=corners, force=True)
    out = np.array([[8.610, 4.303], [5.318, 3.498], [7.323, 2.288],
                    [1.135, 2.657], [3.561, 4.938]])
    assert_almost_equal(sample, out, decimal=1)

    sample = latin.optimal_design(2, 5, bounds=corners, optimization=False)
    out = np.array([[1.052, 4.218], [2.477, 2.987], [7.616, 4.527],
                    [9.134, 3.393], [4.064, 2.430]])
    assert_almost_equal(sample, out, decimal=1)
