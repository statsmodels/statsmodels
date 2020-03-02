import pytest
import numpy as np
import numpy.testing as npt
from .kde_datasets import DataSets, createKDE
from .. import kde_nc

all_methods_data = DataSets.poissonnc()


@pytest.mark.parametrize('data', all_methods_data)
class TestNonContinuous(object):
    def test_method_works(self, data):
        k = createKDE(data)
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        est = k.fit()
        xs = np.arange(est.lower, est.upper + 1)
        ys = est(xs)
        tot = ys.sum()
        npt.assert_allclose(tot, 1, rtol=1e-3)

    def test_grid_method_works(self, data):
        k = createKDE(data)
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        est = k.fit()
        mesh, values = est.grid()
        tot = mesh.integrate(values)
        npt.assert_allclose(tot, 1, rtol=1e-4)

    def test_compute_bandwidth(self, data):
        k = createKDE(data)
        k.axis_type = k.method.axis_type
        k.bandwidth = lambda k: 0.05
        est = k.fit()
        assert est.bandwidth == 0.05

    def test_wrong_bandwidth(self, data):
        k = createKDE(data)
        k.axis_type = k.method.axis_type
        k.bandwidth = None
        with pytest.raises(ValueError):
            k.fit()

    def test_wrong_axis_type(self, data):
        k = createKDE(data)
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        est = k.fit()
        with pytest.raises(ValueError):
            est.axis_type = 'C'

    def test_wrong_axis_type_when_fitting(self, data):
        k = createKDE(data)
        k.axis_type = 'C'
        k.bandwidth = 0.2
        with pytest.raises(ValueError):
            k.fit()

    def test_basic_properties(self, data):
        k = createKDE(data)
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        est = k.fit()
        assert est.ndim == 1
        assert est.npts == len(data.exog)
        with pytest.raises(RuntimeError):
            est.epsilon = 0.1
        est.adjust = 0.1
        assert est.adjust == 0.1
        est.adjust = np.ones(est.exog.shape)
        assert est.adjust.shape == (est.npts,)
        del est.adjust
        assert est.adjust == 1.
        est.bandwidth = 0.1
        assert est.bandwidth == 0.1
        with pytest.raises(ValueError):
            est.bandwidth = -0.5

    def test_update_input(self, data):
        k = createKDE(data)
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        est = k.fit()
        est.update_inputs(data.exog[:-2], data.exog[:-2])
        npt.assert_allclose(est.total_weights,
                            data.exog[:-2].sum(),
                            1e-5, 1e-5)

        with pytest.raises(ValueError):
            est.update_inputs([[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            est.update_inputs([1, 2, 3], [2, 3])

        with pytest.raises(ValueError):
            est.update_inputs([1, 2, 3], [2, 3, 1], [4, 5])

        est.update_inputs([1, 2, 3])
        est.weights = [4, 5, 6]
        est.adjust = [1, 2, 1]
        assert est.total_weights == 15

        with pytest.raises(ValueError):
            est.weights = [1, 2]

        est.weights = 5
        assert est.weights == 1
        assert est.total_weights == est.npts

        est.exog = [2, 3, 4]
        with pytest.raises(ValueError):
            est.exog = [1, 2]

    def test_cdf(self, data):
        if data.method.instance == kde_nc.Ordered:
            k = createKDE(data)
            k.axis_type = k.method.axis_type
            k.bandwidth = 0.2
            est = k.fit()
            mesh, values = est.grid_cdf()
            xs = mesh.full().astype(int)
            values2 = est.cdf(xs[::2])
            npt.assert_allclose(values[::2], values2, 1e-5, 1e-5)
