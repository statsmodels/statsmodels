import pytest
from .kde_datasets import DataSets, createKDE
import numpy.testing as npt

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
