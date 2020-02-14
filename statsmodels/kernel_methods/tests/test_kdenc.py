import pytest
import numpy as np
import numpy.testing as npt
from .kde_datasets import DataSets, createKDE

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
