from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import numpy as np
from numpy.testing.utils import assert_array_compare
from scipy import stats

from ...tools.testing import assert_equal
from .. import kde
from .. import bootstrap


def assert_array_less_equal(a1, a2, *args, **kwargs):
    return assert_array_compare(operator.__le__, a1, a2, *args, **kwargs)


def adjust_bw1(fitted, fct):
    return 0.1*fitted.exog.std(ddof=1)/(fitted.npts**0.2)


def adjust_bw2(fitted, fct):
    return fitted.exog.std(ddof=1)/(fitted.npts**0.2)


class TestPDFBootstrap(object):
    @classmethod
    def setup_class(cls):
        dst = stats.norm(0, 1)
        cls.data = dst.rvs(200)
        cls.ks = kde.KDE(cls.data).fit()
        cls.grid, cls.values = cls.ks.grid()
        cls.eval_points = np.r_[-2:2:64j]
        cls.point_values = cls.ks(cls.eval_points)
        cls.nb_samples = 64

    def test_grid_between(self):
        """
        Estimated grid values should be within the range found in bootstrapping
        """
        grid, est = bootstrap.bootstrap_grid(self.ks, self.nb_samples,
                                             CIs=(.95, .99))
        assert_equal(est.shape, self.values.shape + (2, 2))
        assert_array_less_equal(est[:, 0, 0], self.values)
        assert_array_less_equal(est[:, 1, 0], self.values)
        assert_array_less_equal(self.values, est[:, 0, 1])
        assert_array_less_equal(self.values, est[:, 1, 1])

    def test_full_grid_between(self):
        """
        Estimated grid values should be within the range found in bootstrapping
        """
        grid, est = bootstrap.bootstrap_grid(self.ks, self.nb_samples)
        assert_equal(est.shape, self.values.shape + (self.nb_samples,))
        assert_array_less_equal(est.min(axis=-1), self.values)
        assert_array_less_equal(self.values, est.max(axis=-1))

    def test_points_between(self):
        """
        Estimated point values should be within the range found in
        bootstrapping
        """
        est = bootstrap.bootstrap(self.ks, self.eval_points, self.nb_samples,
                                  CIs=(.95, .99))
        assert_equal(est.shape, self.eval_points.shape + (2, 2))
        assert_array_less_equal(est[:, 0, 0], self.point_values)
        assert_array_less_equal(est[:, 1, 0], self.point_values)
        assert_array_less_equal(self.point_values, est[:, 0, 1])
        assert_array_less_equal(self.point_values, est[:, 1, 1])

    def test_full_points_between(self):
        """
        Estimated point values should be within the range found in
        bootstrapping
        """
        est = bootstrap.bootstrap(self.ks, self.eval_points, self.nb_samples)
        assert_equal(est.shape, self.eval_points.shape + (self.nb_samples,))
        assert_array_less_equal(est.min(axis=-1), self.point_values)
        assert_array_less_equal(self.point_values, est.max(axis=-1))

    def test_grid_adjust_bw(self):
        """
        The grid bootstrapping with smaller bandwidth should be mostly larger
        """
        _, est1 = bootstrap.bootstrap_grid(self.ks, self.nb_samples,
                                           CIs=(.95,), adjust_bw=adjust_bw1)
        _, est2 = bootstrap.bootstrap_grid(self.ks, self.nb_samples,
                                           CIs=(.95,), adjust_bw=adjust_bw2)
        larger1 = sum(est1[:, 0, 0] <= est2[:, 0, 0]) / est1.shape[0]
        assert larger1 > 0.5
        larger2 = sum(est2[:, 0, 1] <= est1[:, 0, 1]) / est1.shape[0]
        assert larger2 > 0.5

    def test_points_adjust_bw(self):
        """
        The point bootstrapping with smaller bandwidth should be mostly larger
        """
        est1 = bootstrap.bootstrap(self.ks, self.eval_points, self.nb_samples,
                                   CIs=(.95,), adjust_bw=adjust_bw1)
        est2 = bootstrap.bootstrap(self.ks, self.eval_points, self.nb_samples,
                                   CIs=(.95,), adjust_bw=adjust_bw2)
        larger1 = sum(est1[:, 0, 0] <= est2[:, 0, 0]) / est1.shape[0]
        assert larger1 > 0.5
        larger2 = sum(est2[:, 0, 1] <= est1[:, 0, 1]) / est1.shape[0]
        assert larger2 > 0.5
