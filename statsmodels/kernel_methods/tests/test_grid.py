from __future__ import division, absolute_import, print_function

from ..kde_utils import Grid, GridInterpolator
import numpy as np
from ...compat.python import zip
from scipy.interpolate import interp2d
from nose.plugins.attrib import attr

@attr('kernel_methods')
class TestBasics(object):
    @classmethod
    def setUpClass(cls):
        cls.sparse_grid = np.ogrid[0:11, 1:100:100j, -5.5:5.5:12j, 0:124]
        cls.sparse_grid[-1] = (cls.sparse_grid[-1] + 0.5) * 2 * np.pi / 124
        cls.axes_def = [g.squeeze() for g in cls.sparse_grid]
        cls.full_grid_c = np.array(np.meshgrid(*cls.sparse_grid, indexing='ij'))
        cls.full_grid_f = np.concatenate([g[..., None] for g in np.meshgrid(*cls.sparse_grid, indexing='ij')], axis=-1)
        cls.bin_types = 'DBRC'
        cls.ndim = 4
        cls.shape = cls.full_grid_c.shape[1:]
        cls.bounds = np.asarray([[-.5, 10.5],
                                 [0.5, 100.5],
                                 [-6, 6],
                                 [0, 2 * np.pi]])
        cls.edges = [np.r_[-0.5:10.5:12j],
                     np.r_[0.5:100.5:101j],
                     np.r_[-6:6:13j],
                     np.r_[0:2 * np.pi:125j]]
        cls.reference = Grid(cls.axes_def, bounds=cls.bounds, bin_types=cls.bin_types,
                             edges=cls.edges)

    def checkIsSame(self, g):
        assert self.reference.almost_equal(g)

    def test_to_sparse(self):
        assert all(np.all(g1 == g2) for (g1, g2) in zip(self.reference.sparse(), self.sparse_grid))

    def test_to_full_c(self):
        assert np.all(self.reference.full('C') == self.full_grid_c)

    def test_to_full_f(self):
        assert np.all(self.reference.full('F') == self.full_grid_f)

    def test_from_axes(self):
        g = Grid(self.axes_def, bin_types=self.bin_types)
        self.checkIsSame(g)

    def test_from_sparse(self):
        g = Grid.fromSparse(self.sparse_grid, bin_types=self.bin_types)
        self.checkIsSame(g)

    def test_from_full_C(self):
        g = Grid.fromFull(self.full_grid_c, order='C', bin_types=self.bin_types)
        self.checkIsSame(g)

    def test_from_full_F(self):
        g = Grid.fromFull(self.full_grid_f, order='F', bin_types=self.bin_types)
        self.checkIsSame(g)

@attr('kernel_methods')
class TestInterpolation(object):
    @classmethod
    def setUpClass(cls):
        ax1 = np.r_[0:2 * np.pi:124j]
        ax1 = (ax1 + (ax1[1] - ax1[0]) / 2)[:-1]
        cls.grid1 = Grid([ax1])
        cls.val1 = np.cos(ax1)
        ax2 = np.r_[-90:90:257j]
        ax2 = (ax2 + (ax2[1] - ax2[0]) / 2)[:-1]
        cls.grid2 = Grid([ax1, ax2])
        sg = cls.grid2.sparse()
        cls.val2 = np.cos(sg[0]) + np.sin(sg[1] * np.pi / 180)
        ax3 = np.r_[0:20]
        cls.grid3 = Grid([ax1, ax3], bin_types='BD')
        sg = cls.grid2.sparse()
        cls.val3 = np.cos(sg[0]) + sg[1]

    def test_1d_bounded(self):
        self.grid1.bin_types = 'B'
        interp = GridInterpolator(self.grid1, self.val1)
        test_values = np.random.rand(256) * 3 * np.pi - np.pi / 2
        interp_test = interp(test_values)
        interp_comp = np.interp(test_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    def test_1d_cyclic(self):
        self.grid1.bin_types = 'C'
        interp = GridInterpolator(self.grid1, self.val1)
        test_values = np.random.rand(256) * 3 * np.pi - np.pi / 2
        # Compute equivalent values
        real_values = test_values % (2 * np.pi)
        interp_test = interp(test_values)
        interp_comp = np.interp(real_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    def test_1d_reflection(self):
        self.grid1.bin_types = 'R'
        interp = GridInterpolator(self.grid1, self.val1)
        test_values = np.random.rand(256) * 3 * np.pi - np.pi / 2
        # Compute equivalent values
        real_values = test_values % (4 * np.pi)
        real_values[real_values > 2 * np.pi] = 4 * np.pi - real_values[real_values > 2 * np.pi]

        interp_test = interp(test_values)
        interp_comp = np.interp(real_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    @staticmethod
    def np_interpolate_2d(ax1, ax2, values, test_values):
        interp = interp2d(ax1, ax2, values.T)
        res = np.empty_like(test_values[:, 0])
        for i in range(test_values.shape[0]):
            res[i] = interp(test_values[i, 0], test_values[i, 1])
        return res

    def test_2d_bounded(self):
        grid = self.grid2
        grid.bin_types = 'B'
        interp = GridInterpolator(grid, self.val2)
        N = 1024
        test_values = np.c_[np.random.rand(N) * 3 * np.pi - np.pi / 2,
                            np.random.rand(N) * 200 - 100]
        real_values = test_values.copy()
        min_val0 = grid.grid[0][0]
        max_val0 = grid.grid[0][-1]
        min_val1 = grid.grid[1][0]
        max_val1 = grid.grid[1][-1]
        real_values[real_values[:, 0] < min_val0, 0] = min_val0
        real_values[real_values[:, 0] > max_val0, 0] = max_val0
        real_values[real_values[:, 1] < min_val1, 1] = min_val1
        real_values[real_values[:, 1] > max_val1, 1] = max_val1

        interp_test = interp(test_values)
        interp_comp = self.np_interpolate_2d(grid.grid[0], grid.grid[1], self.val2, real_values)
        np.testing.assert_allclose(interp_test, interp_comp)

    def test_2d_cyclic(self):
        grid = self.grid2
        grid.bin_types = 'C'
        interp = GridInterpolator(grid, self.val2)
        N = 1024
        test_values = np.c_[np.random.rand(N) * 3 * np.pi - np.pi / 2,
                            np.random.rand(N) * 200 - 100]
        min_val0 = grid.grid[0][0]
        max_val0 = grid.grid[0][-1] + grid.start_interval[0]
        delta_val0 = max_val0 - min_val0
        min_val1 = grid.grid[1][0]
        max_val1 = grid.grid[1][-1] + grid.start_interval[1]
        delta_val1 = max_val1 - min_val1

        real_values = test_values.copy()
        real_values[:, 0] = (real_values[:, 0] - min_val0) % delta_val0 + min_val0
        real_values[:, 1] = (real_values[:, 1] - min_val1) % delta_val1 + min_val1

        ax1 = np.concatenate([grid.grid[0], [grid.grid[0][-1] + grid.start_interval[0]]])
        ax2 = np.concatenate([grid.grid[1], [grid.grid[1][-1] + grid.start_interval[1]]])
        val2 = self.val2
        val2 = np.concatenate([val2, val2[:1, :]], axis=0)
        val2 = np.concatenate([val2, val2[:, :1]], axis=1)
        interp_test = interp(test_values)
        interp_comp = self.np_interpolate_2d(ax1, ax2, val2, real_values)
        np.testing.assert_allclose(interp_test, interp_comp)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
