from __future__ import division, absolute_import, print_function

from .. import _fast_linbin as linbin

from nose.plugins.attrib import attr
from nose.tools import raises
from scipy import stats
from itertools import product
import numpy as np

@attr("kernel_methods")
class TestContinuousBinning1D(object):
    @classmethod
    def setUpClass(cls):
        dst = stats.norm(0, 1)
        cls.data = dst.rvs(2000)
        cls.weights = stats.uniform(1, 5).rvs(2000)
        cls.bin_types = 'CRB'
        cls.bounds = [-3, 3]
        cls.sizes = (64, 128, 159)

    def validity(self, fct, M, bin_type, weighted):
        data = self.data
        bounds = self.bounds
        weights = 1.
        size = data.shape[0]
        if bin_type == 'B':
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
        assert mesh.shape == (M,), "Bad mesh shape"
        assert mesh.grid[0][0] >= bounds[0], "Incorrect lower bound"
        assert mesh.grid[0][-1] <= bounds[1], "Incorrect upper bound"
        assert abs(bins.sum() - size) < 1e-8, "Incorrect sum of binned values"

    def test_validity(self):
        for fct, s, t in product([linbin.fast_bin, linbin.fast_linbin],
                                 self.sizes, self.bin_types):
            yield self.validity, fct, s, t, True
            yield self.validity, fct, s, t, False

    @raises(ValueError)
    def bad_data(self, fct):
        bounds = self.bounds
        fct([[[1]]], bounds, 256, 1., 'B')

    @raises(ValueError)
    def bad_bin_type1(self, fct):
        bounds = self.bounds
        fct(self.data, bounds, 256, 1., 'X')

    @raises(ValueError)
    def bad_bin_type2(self, fct):
        bounds = self.bounds
        fct(self.data, bounds, 256, 1., 'BB')

    @raises(ValueError)
    def bad_bin_type3(self, fct):
        bounds = self.bounds
        fct(self.data, bounds, 256, 1., '')

    @raises(ValueError)
    def bad_weights(self, fct):
        bounds = self.bounds
        fct(self.data, bounds, 256, [1, 2, 3], 'B')

    @raises(ValueError)
    def bad_bounds1(self, fct):
        fct(self.data, [[1, 2], [3, 4]], 256, 1., 'B')

    @raises(ValueError)
    def bad_bounds2(self, fct):
        fct(self.data, [1, 2, 3], 256, 1., 'B')

    @raises(ValueError)
    def bad_bounds3(self, fct):
        fct(self.data, 'bad', 256, 1., 'B')

    @raises(ValueError)
    def bad_out1(self, fct):
        out = np.empty((1,), dtype=float)
        fct(self.data, self.bounds, 256, 1., 'B', out)

    @raises(ValueError)
    def bad_out2(self, fct):
        out = np.empty(self.weights.shape, dtype=int)
        fct(self.data, self.bounds, 256, 1., 'B', out)

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

@attr("kernel_methods")
class TestContinuousBinningnD(object):
    @classmethod
    def setUpClass(cls):
        dst = stats.norm(0, 1)
        cls.data = dst.rvs(4*2000).reshape(2000, 4)
        cls.weights = stats.uniform(1, 5).rvs(2000)
        cls.bin_types = 'CRB'
        cls.bounds = [[-3, 3]]
        cls.sizes = (16, 32, 21)

    def validity(self, fct, d, M, bin_type, weighted):
        data = self.data[:, :d]
        weights = 1.
        size = data.shape[0]
        bounds = self.bounds*d
        if bin_type == 'B':
            sel = np.all([(data[:, i] >= bounds[i][0]) & (data[:, i] < bounds[i][1]) for i in range(d)], axis=0)
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
        assert mesh.shape == M, "Bad mesh shape"
        for d in range(len(bounds)):
            assert mesh.grid[d][0] >= bounds[d][0], "Incorrect lower bound"
            assert mesh.grid[d][-1] <= bounds[d][1], "Incorrect upper bound"
        assert abs(bins.sum() - size) < 1e-8, "Incorrect sum of binned values"

    def test_validity(self):
        for fct, s, t, d in product([linbin.fast_bin_nd, linbin.fast_linbin_nd],
                                    self.sizes, self.bin_types, [2, 3, 4]):
            s = (s,)*d
            yield self.validity, fct, d, s, t, True
            yield self.validity, fct, d, s, t, False

    @raises(ValueError)
    def bad_data(self, fct):
        bounds = self.bounds * 4
        fct([[[1]]], bounds, 256, 1., 'BBBB')

    @raises(ValueError)
    def bad_bin_type1(self, fct):
        bounds = self.bounds * 4
        fct(self.data, bounds, 256, 1., 'X')

    @raises(ValueError)
    def bad_bin_type2(self, fct):
        bounds = self.bounds * 4
        fct(self.data, bounds, 256, 1., 'BBBX')

    @raises(ValueError)
    def bad_bin_type3(self, fct):
        bounds = self.bounds * 4
        fct(self.data, bounds, 256, 1., 'BBB')

    @raises(ValueError)
    def bad_weights(self, fct):
        bounds = self.bounds * 4
        fct(self.data, bounds, 256, [1, 2, 3], 'BBBB')

    @raises(ValueError)
    def bad_bounds1(self, fct):
        bounds = self.bounds * 3
        fct(self.data, bounds, 256, 1., 'BBBB')

    @raises(ValueError)
    def bad_bounds2(self, fct):
        fct(self.data, [1, 2, 3], 256, 1., 'BBBB')

    @raises(ValueError)
    def bad_bounds3(self, fct):
        fct(self.data, 'bad', 256, 1., 'BBBB')

    @raises(ValueError)
    def bad_out1(self, fct):
        bounds = self.bounds * 4
        out = np.empty((1,), dtype=float)
        fct(self.data, bounds, 256, 1., 'BBBB', out)

    @raises(ValueError)
    def bad_out2(self, fct):
        bounds = self.bounds * 4
        out = np.empty(self.weights.shape, dtype=int)
        fct(self.data, bounds, 256, 1., 'BBBB', out)

    def test_bad_args(self):
        for fct in [linbin.fast_bin_nd, linbin.fast_linbin_nd]:
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
