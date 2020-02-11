import pytest
from . import kde_test_utils
from .kde_test_utils import kde_tester, datasets
import numpy.testing as npt

all_methods_data = kde_test_utils.generate_methods_data(['norm2d'])

@pytest.mark.parametrize(kde_test_utils.kde_tester_args, all_methods_data)
class TestKDE2D(object):

    @kde_tester
    def test_method_works(self, k, method, data):
        """
        Enure the sum of the kernel over its domain is unity
        """
        est = k.fit()
        val = est([0, 0])
        assert val >= 0
        del k.weights
        del k.adjust
        est = k.fit()
        npt.assert_equal(est.total_weights, k.npts)
        npt.assert_equal(est.adjust, 1.)

    @kde_tester
    def test_grid_method_works(self, k, method, data):
        """
        Enure the sum of the kernel over its domain is unity
        """
        est = k.fit()
        mesh, vals = est.grid(N=32)
        tot = mesh.integrate(vals)
        acc = max(method.normed_accuracy, method.grid_accuracy)
        npt.assert_allclose(tot, 1, rtol=acc, atol=acc)
        del k.weights
        del k.adjust
        est = k.fit()
        npt.assert_equal(est.total_weights, k.npts)
        npt.assert_equal(est.adjust, 1.)
