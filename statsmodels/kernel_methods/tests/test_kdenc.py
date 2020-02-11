import pytest
from . import kde_test_utils
import numpy as np
import numpy.testing as npt
from .kde_test_utils import kde_tester, datasets

all_methods_data = kde_test_utils.generate_methods_data(['nc'])

@pytest.mark.parametrize(kde_test_utils.kde_tester_args, all_methods_data)
class TestNonContinuous(object):

    @kde_tester
    def test_method_works(self, k, method, data):
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        est = k.fit()
        xs = np.arange(est.lower, est.upper+1)
        ys = est(xs)
        tot = ys.sum()
        npt.assert_allclose(tot, 1, rtol=1e-3)

    @kde_tester
    def test_grid_method_works(self, k, method, data):
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        est = k.fit()
        mesh, values = est.grid()
        tot = mesh.integrate(values)
        npt.assert_allclose(tot, 1, rtol=1e-4)
