import numpy as np
import numpy.testing as npt
from statsmodels.doe import lhs


def test_lhs():
    np.random.seed(123456)

    corners = np.array([[0, 2], [10, 5]])

    sample = lhs.lhs(dim=2, n_sample=5, bounds=corners)
    out = np.array([[5.746, 3.219], [5.479, 3.261], [9.246, 4.798],
                    [9.097, 4.495], [9.753, 4.074]])
    npt.assert_almost_equal(sample, out, decimal=1)

    sample = lhs.lhs(dim=2, n_sample=5, centered=True)
    out = np.array([[0.3, 0.9], [0.7, 0.7], [0.1, 0.9],
                    [0.5, 0.5], [0.1, 0.7]])
    npt.assert_almost_equal(sample, out, decimal=1)


def test_orthogonal_lhs():
    np.random.seed(123456)

    corners = np.array([[0, 2], [10, 5]])

    sample = lhs.olhs(dim=2, n_sample=5, bounds=corners)
    out = np.array([[3.933, 2.670], [7.794, 4.031], [4.520, 2.129],
                    [0.253, 4.976], [8.753, 3.249]])
    npt.assert_almost_equal(sample, out, decimal=1)


def test_optimal_design():
    np.random.seed(123456)

    corners = np.array([[0, 2], [10, 5]])

    sample = lhs.optimal_design(dim=2, n_sample=5, bounds=corners)
    out = np.array([[4.520, 2.670], [0.253, 4.031], [7.794, 2.129],
                    [3.933, 4.976], [8.753, 3.249]])
    npt.assert_almost_equal(sample, out, decimal=1)

    sample = lhs.optimal_design(dim=2, n_sample=5)
    out = np.array([[0.355, 0.868], [0.627, 0.114], [0.045, 0.649],
                    [0.518, 0.518], [0.970, 0.212]])
    npt.assert_almost_equal(sample, out, decimal=1)

    sample = lhs.optimal_design(dim=2, n_sample=5, bounds=corners, force=True)
    out = np.array([[8.687, 2.695], [0.075, 4.159], [3.723, 2.305],
                    [5.507, 3.368], [6.810, 4.535]])
    npt.assert_almost_equal(sample, out, decimal=1)
