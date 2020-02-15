import pytest
from .kde_datasets import DataSets, createKDE
import numpy as np
import numpy.testing as npt
from .. import kde_methods, kde

all_methods_data = DataSets.normnd(2)


@pytest.mark.parametrize('data', all_methods_data)
class TestKDE2D(object):
    def test_method_works(self, data):
        """
        Enure the sum of the kernel over its domain is unity
        """
        k = createKDE(data)
        est = k.fit()
        val = est([0, 0])
        assert val >= 0
        if k.weights is None and k.adjust is None:
            return
        del k.weights
        del k.adjust
        est = k.fit()
        npt.assert_equal(est.total_weights, k.npts)
        npt.assert_equal(est.adjust, 1.)

    def test_grid_method_works(self, data):
        """
        Enure the sum of the kernel over its domain is unity
        """
        k = createKDE(data)
        est = k.fit()
        mesh, vals = est.grid(N=32)
        tot = mesh.integrate(vals)
        acc = max(data.method.normed_accuracy, data.method.grid_accuracy)
        npt.assert_allclose(tot, 1, rtol=acc, atol=acc)
        if k.weights is None and k.adjust is None:
            return
        del k.weights
        del k.adjust
        est = k.fit()
        npt.assert_equal(est.total_weights, k.npts)
        npt.assert_equal(est.adjust, 1.)

class TestKDE2DExtra(object):

    methods = [kde_methods.KDEnDMethod, kde_methods.Cyclic]

    @staticmethod
    @pytest.fixture
    def data():
        return next(d for d in DataSets.normnd(2, [64]) if d.weights is not None)

    @pytest.mark.parametrize('method', methods)
    def test_grid_matrix_bandwidth(self, data, method):
        k = kde.KDE(data.exog, bandwidth = np.array([[.1, .2], [.2, .15]]), method=method)
        est = k.fit()
        grid, pdf = est.grid(16, cut=3)
        assert grid.shape == (16, 16)

    @pytest.mark.parametrize('method', methods)
    def test_default_grid_size(self, data, method):
        k = kde.KDE(data.exog, method=method)
        est = k.fit()
        assert est.grid_size() == 2**5

    @pytest.mark.parametrize('method', methods)
    def test_grid_scalar_bandwidth(self, data, method):
        k = kde.KDE(data.exog, bandwidth = 1., method=method)
        est = k.fit()
        grid, pdf = est.grid(16, cut=3)
        assert grid.shape == (16, 16)

    @pytest.mark.parametrize('method', methods)
    def test_grid_high_dimensional_N(self, data, method):
        k = kde.KDE(data.exog, bandwidth = 1., method=method)
        est = k.fit()
        with pytest.raises(ValueError):
            est.grid([[1, 2], [3, 4]])

    def test_fallback_1d(self, data):
        exog1d = data.exog[:, [0]]
        k = kde.KDE(exog1d, method=kde_methods.KDEnDMethod)
        est = k.fit()
        assert isinstance(est, kde_methods.Reflection1D)

    def test_fallback_1d_cyclic(self, data):
        exog1d = data.exog[:, [0]]
        k = kde.KDE(exog1d, method=kde_methods.Cyclic)
        est = k.fit()
        assert isinstance(est, kde_methods.Cyclic1D)

    def test_set_bandwidth(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        est.bandwidth = 2.
        assert est.bandwidth == 2.

    def test_set_bandwidth_bad_dimensions(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        with pytest.raises(ValueError):
            est.bandwidth = np.ones((2, 2, 2))

    def test_set_bandwidth_bad_shape(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        with pytest.raises(AssertionError):
            est.bandwidth = np.ones((5, 5))

    def test_update_inputs(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        ws = np.arange(est.npts)
        est.update_inputs(data.exog, weights=ws, adjust=ws)
        assert est.total_weights == ws.sum()

    def test_update_inputs_bad_exog(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        with pytest.raises(ValueError):
            est.update_inputs(np.ones((2, 2, 2)))

    def test_update_inputs_bad_weights(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        with pytest.raises(ValueError):
            est.update_inputs(data.exog, np.ones((2, 2)))

    def test_update_inputs_bad_adjust(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        with pytest.raises(ValueError):
            est.update_inputs(data.exog, adjust=np.ones((2, 2)))

    def test_closed(self, data):
        k = kde.KDE(data.exog, bandwidth = 1, method=kde_methods.KDEnDMethod)
        est = k.fit()
        assert est.closed

    @pytest.mark.parametrize('method', methods)
    def test_pdf_small_output(self, data, method):
        k = kde.KDE(data.exog, method=method)
        k.lower = data.exog.min(axis=0)
        k.upper = data.exog.max(axis=0)
        est = k.fit()
        points = np.asarray([[-1, -1], [0, 0], [1, 1]])
        values = est.pdf(points)
        assert values.shape == (len(points),)

    @pytest.mark.parametrize('method', methods)
    def test_pdf_large_output(self, data, method):
        k = kde.KDE(data.exog, method=method)
        k.lower = data.exog.min(axis=0)
        k.upper = data.exog.max(axis=0)
        est = k.fit()
        assert est.bounded()
        points = np.r_[-k.lower+0.1:k.upper-0.1:1j*(est.npts+2)]
        values = est.pdf(points)
        assert values.shape == (len(points),)
