from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product

from scipy import stats
import numpy as np

from ...tools.testing import (assert_equal, assert_allclose, assert_raises)
from .. import fast_linbin as linbin


class TestContinuousBinning1D(object):
    @classmethod
    def setup_class(cls):
        dst = stats.norm(0, 1)
        cls.data = dst.rvs(2000)
        cls.weights = stats.uniform(1, 5).rvs(2000)
        cls.bin_type = 'crb'
        cls.bounds = [-3, 3]
        cls.sizes = (64, 128, 159)

    def validity(self, fct, M, bin_type, weighted):
        data = self.data
        bounds = self.bounds
        weights = 1.
        size = data.shape[0]
        if bin_type == 'b':
            sel = (data >= bounds[0]) & (data < bounds[1])
            if weighted:
                weights = self.weights
                size = weights[sel].sum()
            else:
                size = sel.sum()
        elif weighted:
            weights = self.weights
            size = weights.sum()
        mesh, bins = fct(data, bounds, M, weights, bin_type)
        assert_equal(mesh.shape, (M,))
        assert mesh.grid[0][0] >= bounds[0]
        assert mesh.grid[0][-1] <= bounds[1]
        assert_allclose(bins.sum(), size, rtol=1e-8)

    def test_validity(self):
        for fct, s, t in product([linbin.fast_bin, linbin.fast_linbin],
                                 self.sizes, self.bin_type):
            yield self.validity, fct, s, t, True
            yield self.validity, fct, s, t, False

    def bad_data(self, fct):
        bounds = self.bounds
        assert_raises(ValueError, fct, [[[1]]], bounds, 4, 1., 'b')

    def bad_bin_type1(self, fct):
        bounds = self.bounds
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'X')

    def bad_bin_type2(self, fct):
        bounds = self.bounds
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'bb')

    def bad_bin_type3(self, fct):
        bounds = self.bounds
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., '')

    def bad_weights(self, fct):
        bounds = self.bounds
        assert_raises(ValueError, fct, self.data, bounds, 4, [1, 2, 3], 'b')

    def bad_bounds1(self, fct):
        assert_raises(ValueError, fct, self.data, [[1, 2], [3, 4]], 4, 1., 'b')

    def bad_bounds2(self, fct):
        assert_raises(ValueError, fct, self.data, [1, 2, 3], 4, 1., 'b')

    def bad_bounds3(self, fct):
        assert_raises(ValueError, fct, self.data, 'bad', 4, 1., 'b')

    def bad_out1(self, fct):
        out = np.empty((1,), dtype=float)
        assert_raises(ValueError, fct, self.data, self.bounds, 4, 1., 'b', out)

    def bad_out2(self, fct):
        out = np.empty(self.weights.shape, dtype=int)
        assert_raises(ValueError, fct, self.data, self.bounds, 4, 1., 'b', out)

    def test_bad_args(self):
        for fct in [linbin.fast_bin, linbin.fast_linbin]:
            yield self.bad_data, fct
            yield self.bad_bin_type1, fct
            yield self.bad_bin_type2, fct
            yield self.bad_bin_type3, fct
            yield self.bad_weights, fct
            yield self.bad_bounds1, fct
            yield self.bad_bounds2, fct
            yield self.bad_bounds3, fct
            yield self.bad_out1, fct
            yield self.bad_out2, fct


class TestContinuousBinningnD(object):
    @classmethod
    def setup_class(cls):
        dst = stats.norm(0, 1)
        cls.data = dst.rvs(4*2000).reshape(2000, 4)
        cls.weights = stats.uniform(1, 5).rvs(2000)
        cls.bin_type = 'crb'
        cls.bounds = [[-3, 3]]
        cls.sizes = (16, 32, 21)

    def validity(self, fct, d, M, bin_type, weighted):
        data = self.data[:, :d]
        weights = 1.
        size = data.shape[0]
        bounds = self.bounds*d
        if bin_type == 'b':
            sel = np.all([
                (data[:, i] >= bounds[i][0]) & (data[:, i] < bounds[i][1])
                for i in range(d)
            ], axis=0)
            if weighted:
                weights = self.weights
                size = weights[sel].sum()
            else:
                size = sel.sum()
        elif weighted:
            weights = self.weights
            size = weights.sum()
        bin_type = bin_type*d
        mesh, bins = fct(data, bounds, M, weights, bin_type)
        assert_equal(mesh.shape, M)
        for d in range(len(bounds)):
            assert mesh.grid[d][0] >= bounds[d][0]
            assert mesh.grid[d][-1] <= bounds[d][1]
        assert_allclose(bins.sum(), size, rtol=1e-8)

    def test_validity(self):
        for fct, s, t, d in product(
                [linbin.fast_bin_nd, linbin.fast_linbin_nd], self.sizes,
                self.bin_type, [2, 3, 4]):
            s = (s,)*d
            yield self.validity, fct, d, s, t, True
            yield self.validity, fct, d, s, t, False

    def bad_data(self, fct):
        bounds = self.bounds * 4
        assert_raises(ValueError, fct, [[[1]]], bounds, 4, 1., 'bbbb')

    def bad_bin_type1(self, fct):
        bounds = self.bounds * 4
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'X')

    def bad_bin_type2(self, fct):
        bounds = self.bounds * 4
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'bbbx')

    def bad_bin_type3(self, fct):
        bounds = self.bounds * 4
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'bbb')

    def bad_weights(self, fct):
        bounds = self.bounds * 4
        assert_raises(ValueError, fct, self.data, bounds, 4, [1, 2, 3], 'bbbb')

    def bad_bounds1(self, fct):
        bounds = self.bounds * 3
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'bbbb')

    def bad_bounds2(self, fct):
        assert_raises(ValueError, fct, self.data, [1, 2, 3], 4, 1., 'bbbb')

    def bad_bounds3(self, fct):
        assert_raises(ValueError, fct, self.data, 'bad', 4, 1., 'bbbb')

    def bad_out1(self, fct):
        bounds = self.bounds * 4
        out = np.empty((1,), dtype=float)
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'bbbb', out)

    def bad_out2(self, fct):
        bounds = self.bounds * 4
        out = np.empty(self.weights.shape, dtype=int)
        assert_raises(ValueError, fct, self.data, bounds, 4, 1., 'bbbb', out)

    def bad_args(self, test, *args):
        f = getattr(self, test)
        f(*args)

    def test_bad_args(self):
        for fct in [linbin.fast_bin_nd, linbin.fast_linbin_nd]:
            yield self.bad_args, "bad_data", fct
            yield self.bad_args, "bad_bin_type1", fct
            yield self.bad_args, "bad_bin_type2", fct
            yield self.bad_args, "bad_bin_type3", fct
            yield self.bad_args, "bad_weights", fct
            yield self.bad_args, "bad_bounds1", fct
            yield self.bad_args, "bad_bounds2", fct
            yield self.bad_args, "bad_bounds3", fct
            yield self.bad_args, "bad_out1", fct
            yield self.bad_args, "bad_out2", fct


class TestDiscreteBinning(object):
    @classmethod
    def setup_class(cls):
        dst = stats.poisson(12)
        cls.data = dst.rvs(2000).reshape(1000, 2)
        cls.weights = stats.uniform(1, 5).rvs(1000)
        cls.bin_type = 'dd'
        cls.real_upper = cls.data.max(axis=0)
        cls.test_upper = [12, 12]
        cls.sizes = (16, 32, 21)

    def validity_1d(self, fct, M, bounds, weighted):
        data = self.data[:, 0]
        sel = data <= bounds[1]
        if weighted:
            weights = self.weights
            size = weights[sel].sum()
        else:
            weights = 1.
            size = sel.sum()
        mesh, bins = fct(data, bounds, M, weights, 'd')
        assert_equal(mesh.grid[0][0], 0)
        assert_equal(mesh.grid[0][-1], bounds[1])
        assert_equal(len(mesh.grid[0]), bounds[1]+1)
        assert_allclose(bins.sum(), size, rtol=1e-8)

    def test_validity_1d(self):
        for fct, s in product([linbin.fast_bin, linbin.fast_linbin],
                              self.sizes):
            yield self.validity_1d, fct, s, [0, self.real_upper[0]], True
            yield self.validity_1d, fct, s, [0, self.real_upper[0]], False
            yield self.validity_1d, fct, s, [0, self.test_upper[0]], True
            yield self.validity_1d, fct, s, [0, self.test_upper[0]], False
