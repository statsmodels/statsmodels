import pytest
from ..kde_utils import Grid
import numpy as np
import numpy.testing as npt
from .kde_datasets import DataSets, createKDE
from .. import bandwidths, kde, kde_1d, kde_nc

all_methods_data = DataSets.multivariate()


@pytest.mark.parametrize('data', all_methods_data)
class TestMultivariate(object):
    def test_method_works(self, data):
        k = createKDE(data)
        est = k.fit()
        bt = est.bin_type
        bounds = [None, None]
        if est.methods[0].bin_type == 'D':
            bounds[0] = [est.lower[0], est.upper[0]]
        else:
            if data.method[0].bound_low:
                low = data.lower[0]
            else:
                low = est.exog[:, 0].min() - 5 * est.bandwidth[0]
            if data.method[0].bound_high:
                high = data.upper[0]
            else:
                high = est.exog[:, 0].max() + 5 * est.bandwidth[0]
            bounds[0] = [low, high]
        if est.methods[1].bin_type == 'D':
            bounds[1] = [est.lower[1], est.upper[1]]
        else:
            if data.method[1].bound_low:
                low = data.lower[1]
            else:
                low = est.exog[:, 1].min() - 5 * est.bandwidth[1]
            if data.method[1].bound_high:
                high = data.upper[1]
            else:
                high = est.exog[:, 1].max() + 5 * est.bandwidth[1]
            bounds[1] = [low, high]
        grid = Grid.fromBounds(bounds, bin_type=bt, shape=128, dtype=float)
        values = est(grid.linear()).reshape(grid.shape)
        tot = grid.integrate(values)
        # Note: the precision is quite bad as we use small number of values!
        acc = 100 * max(m.normed_accuracy for m in data.method)
        npt.assert_allclose(tot, 1., rtol=acc)
        del k.weights
        del k.adjust
        est = k.fit()
        npt.assert_equal(est.total_weights, est.npts)
        npt.assert_equal(est.adjust, 1.)

    def test_grid_method_works(self, data):
        k = createKDE(data)
        est = k.fit()
        mesh, values = est.grid(512)
        tot = mesh.integrate(values)
        acc = max(m.grid_accuracy for m in data.method)
        npt.assert_allclose(tot, 1., rtol=acc)


class MyOrdered(kde_nc.Ordered):
    pass


class MyUnordered(kde_nc.Unordered):
    pass


class TestMultivariateExtra(object):
    @staticmethod
    @pytest.fixture(scope='class')
    def data():
        return next(d for d in all_methods_data if d.weights is not None)

    def test_set_wrong_axis_type(self, data):
        k = kde.KDE(data.exog, axis_type='COCO')
        with pytest.raises(ValueError):
            k.fit()

    def test_fallback_continuous(self, data):
        k = kde.KDE(data.exog[:, [1]], axis_type='C')
        est = k.fit()
        assert isinstance(est, type(k.method.continuous_method))

    def test_fallback_unordered(self, data):
        k = kde.KDE(data.exog[:, [1]].round().astype(int), axis_type='U')
        est = k.fit()
        assert isinstance(est, type(k.method.unordered_method))

    def test_fallback_ordered(self, data):
        k = kde.KDE(data.exog[:, [1]].round().astype(int), axis_type='O')
        est = k.fit()
        assert isinstance(est, type(k.method.ordered_method))

    def test_set_incorrect_property(self, data):
        with pytest.raises(ValueError):
            kde.Multivariate(bad_attr=1)

    def test_set_bandwidth(self, data):
        k = kde.KDE(data.exog, bandwidth=[2, 2])
        est = k.fit()
        npt.assert_equal(est.bandwidth, [2, 2])

    def test_compute_bandwidth(self, data):
        k = kde.KDE(
            data.exog,
            method=kde.Multivariate(
                methods=[kde_1d.Unbounded1D(),
                         kde_1d.Unbounded1D()]))
        k.bandwidth = [bandwidths.scotts, 2.]
        est = k.fit()
        assert est.bandwidth[1] == 2.

    def test_compute_bandwidth_bad_size(self, data):
        k = kde.KDE(
            data.exog,
            method=kde.Multivariate(
                methods=[kde_1d.Unbounded1D(),
                         kde_1d.Unbounded1D()]))
        k.bandwidth = [1, 2, 3]
        with pytest.raises(ValueError):
            k.fit()

    def test_continuous_method(self, data):
        k = kde.KDE(data.exog, axis_type='CC')
        k.method.continuous_method = kde_1d.Reflection1D()
        assert isinstance(k.method.continuous_method, kde_1d.Reflection1D)
        est = k.fit()
        assert isinstance(est.methods[0], kde_1d.Reflection1D)
        assert isinstance(est.methods[1], kde_1d.Reflection1D)

    def test_ordered_method(self, data):
        k = kde.KDE(data.exog, axis_type='OO')
        k.method.ordered_method = MyOrdered()
        assert isinstance(k.method.ordered_method, MyOrdered)
        est = k.fit()
        assert isinstance(est.methods[0], MyOrdered)
        assert isinstance(est.methods[1], MyOrdered)

    def test_unordered_method(self, data):
        k = kde.KDE(data.exog, axis_type='UU')
        k.method.unordered_method = MyUnordered()
        assert isinstance(k.method.unordered_method, MyUnordered)
        est = k.fit()
        assert isinstance(est.methods[0], MyUnordered)
        assert isinstance(est.methods[1], MyUnordered)

    def test_adjust(self, data):
        k = kde.KDE(data.exog, axis_type='C')
        est = k.fit()
        est.adjust = 2.
        est.adjust = np.r_[1:2:1j * est.npts]

    def test_adjust_wrong_shape(self, data):
        k = kde.KDE(data.exog, axis_type='CC')
        est = k.fit()
        with pytest.raises(ValueError):
            est.adjust = np.ones((2, 2))

    def test_set_bandwidth_after_fit(self, data):
        k = kde.KDE(data.exog, axis_type='C')
        est = k.fit()
        est.bandwidth = [1, 2]
        npt.assert_equal(est.bandwidth, [1, 2])

    def test_to_bin_exog_if_possible(self, data):
        k = kde.KDE(data.exog, axis_type='C')
        est = k.fit()
        assert est.to_bin is est.exog


#    def test_to_bin_transformed(self, data):
#        exog = abs(data.exog) + 0.1
#        k = kde.KDE(data.exog, axis_type='C')
#        k.continuous_method = kde_1d.Transform1D(kde_1d.LogTransform)
#        est = k.fit()
#        npt.assert_allclose(est.to_bin, np.log(exog), 1e-5, 1e-5)
