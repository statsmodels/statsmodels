from .. import fast_linbin as linbin

import numpy.testing as npt
import pytest
from scipy import stats
from itertools import product
import numpy as np

binfcts = [linbin.fast_bin, linbin.fast_linbin]
binfcts_nd = [linbin.fast_bin_nd, linbin.fast_linbin_nd]

class TestContinuousBinning1D(object):
    @classmethod
    def setup_class(cls):
        dst = stats.norm(0, 1)
        cls.data = dst.rvs(2000)
        cls.weights = stats.uniform(1, 5).rvs(2000)
        cls.bounds = [-3, 3]


    @pytest.mark.parametrize("fct", binfcts)
    @pytest.mark.parametrize("M", (64, 128, 159))
    @pytest.mark.parametrize("bin_type", 'CRB')
    @pytest.mark.parametrize("weighted", [True, False])
    def test_validity(self, fct, M, bin_type, weighted):
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
        npt.assert_equal(mesh.shape, (M,))
        assert mesh.grid[0][0] >= bounds[0]
        assert mesh.grid[0][-1] <= bounds[1]
        npt.assert_allclose(bins.sum(), size, rtol=1e-8)

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_data(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds
            fct([[[1]]], bounds, 4, 1., 'B')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_bin_type1(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds
            fct(self.data, bounds, 4, 1., 'X')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_bin_type2(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds
            fct(self.data, bounds, 4, 1., 'BB')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_bin_type3(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds
            fct(self.data, bounds, 4, 1., '')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_weights(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds
            fct(self.data, bounds, 4, [1, 2, 3], 'B')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_bounds1(self, fct):
        with pytest.raises(ValueError):
            fct(self.data, [[1, 2], [3, 4]], 4, 1., 'B')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_bounds2(self, fct):
        with pytest.raises(ValueError):
            fct(self.data, [1, 2, 3], 4, 1., 'B')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_bounds3(self, fct):
        with pytest.raises(ValueError):
            fct(self.data, 'bad', 4, 1., 'B')

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_out1(self, fct):
        with pytest.raises(ValueError):
            out = np.empty((1,), dtype=float)
            fct(self.data, self.bounds, 4, 1., 'B', out)

    @pytest.mark.parametrize("fct", binfcts)
    def test_bad_out2(self, fct):
        with pytest.raises(ValueError):
            out = np.empty(self.weights.shape, dtype=int)
            fct(self.data, self.bounds, 4, 1., 'B', out)

class TestContinuousBinningnD(object):
    @classmethod
    def setup_class(cls):
        dst = stats.norm(0, 1)
        cls.data = dst.rvs(4*2000).reshape(2000, 4)
        cls.weights = stats.uniform(1, 5).rvs(2000)
        cls.bounds = [[-3, 3]]

    @pytest.mark.parametrize("fct", binfcts_nd)
    @pytest.mark.parametrize("d", [2, 3, 4])
    @pytest.mark.parametrize("M", (16, 32, 21))
    @pytest.mark.parametrize("bin_type", 'CRB')
    @pytest.mark.parametrize("weighted", [True, False])
    def test_validity(self, fct, d, M, bin_type, weighted):
        M = (M,)*d
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
        npt.assert_equal(mesh.shape, M)
        for d in range(len(bounds)):
            assert mesh.grid[d][0] >= bounds[d][0]
            assert mesh.grid[d][-1] <= bounds[d][1]
        npt.assert_allclose(bins.sum(), size, rtol=1e-8)

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_data(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 4
            fct([[[1]]], bounds, 4, 1., 'BBBB')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_bin_type1(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 4
            fct(self.data, bounds, 4, 1., 'X')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_bin_type2(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 4
            fct(self.data, bounds, 4, 1., 'BBBX')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_bin_type3(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 4
            fct(self.data, bounds, 4, 1., 'BBB')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_weights(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 4
            fct(self.data, bounds, 4, [1, 2, 3], 'BBBB')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_bounds1(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 3
            fct(self.data, bounds, 4, 1., 'BBBB')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_bounds2(self, fct):
        with pytest.raises(ValueError):
            fct(self.data, [1, 2, 3], 4, 1., 'BBBB')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_bounds3(self, fct):
        with pytest.raises(ValueError):
            fct(self.data, 'bad', 4, 1., 'BBBB')

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_out1(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 4
            out = np.empty((1,), dtype=float)
            fct(self.data, bounds, 4, 1., 'BBBB', out)

    @pytest.mark.parametrize("fct", binfcts_nd)
    def test_bad_out2(self, fct):
        with pytest.raises(ValueError):
            bounds = self.bounds * 4
            out = np.empty(self.weights.shape, dtype=int)
            fct(self.data, bounds, 4, 1., 'BBBB', out)

class TestDiscreteBinning(object):
    @classmethod
    def setup_class(cls):
        dst = stats.poisson(12)
        cls.data = dst.rvs(2000).reshape(1000, 2)
        cls.weights = stats.uniform(1, 5).rvs(1000)
        cls.bin_type = 'DD'
        cls.real_upper = cls.data.max(axis=0)
        cls.test_upper = [12, 12]

    @pytest.mark.parametrize("fct", binfcts)
    @pytest.mark.parametrize("M", (16, 32, 21))
    @pytest.mark.parametrize("use_real_bounds", [True, False])
    @pytest.mark.parametrize("weighted", [True, False])
    def test_validity_1d(self, fct, M, use_real_bounds, weighted):
        if use_real_bounds:
            bounds = [0, self.real_upper[0]]
        else:
            bounds = [0, self.test_upper[0]]
        data = self.data[:, 0]
        sel = data <= bounds[1]
        if weighted:
            weights = self.weights
            size = weights[sel].sum()
        else:
            weights = 1.
            size = sel.sum()
        mesh, bins = fct(data, bounds, M, weights, 'D')
        assert mesh.grid[0][0] == 0
        assert mesh.grid[0][-1] == bounds[1]
        assert len(mesh.grid[0]) == bounds[1]+1
        npt.assert_allclose(bins.sum(), size, rtol=1e-8)

