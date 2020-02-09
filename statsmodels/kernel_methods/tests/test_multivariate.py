import pytest
from . import kde_utils
from .. import kde, kde_methods
import numpy as np
import numpy.testing as npt
from .kde_utils import kde_tester, datasets

all_methods_data = kde_utils.generate_methods_data(['multivariate'])

@pytest.mark.parametrize(kde_utils.kde_tester_args, all_methods_data)
class TestMultivariate(object):

    @kde_tester
    def test_method_works(self, k, methods, data):
        est = k.fit()
        bt = est.bin_type
        bounds = [None, None]
        if est.methods[0].bin_type == 'D':
            bounds[0] = [est.lower[0], est.upper[0]]
        else:
            if methods[0].bound_low:
                low = data.lower[0]
            else:
                low = est.exog[:, 0].min() - 5*est.bandwidth[0]
            if methods[0].bound_high:
                high = data.upper[0]
            else:
                high = est.exog[:, 0].max() + 5*est.bandwidth[0]
            bounds[0] = [low, high]
        if est.methods[1].bin_type == 'D':
            bounds[1] = [est.lower[1], est.upper[1]]
        else:
            if methods[1].bound_low:
                low = data.lower[1]
            else:
                low = est.exog[:, 1].min() - 5*est.bandwidth[1]
            if methods[1].bound_high:
                high = data.upper[1]
            else:
                high = est.exog[:, 1].max() + 5*est.bandwidth[1]
            bounds[1] = [low, high]
        grid = kde_utils.Grid.fromBounds(bounds, bin_type=bt, shape=128, dtype=float)
        values = est(grid.linear()).reshape(grid.shape)
        tot = grid.integrate(values)
        # Note: the precision is quite bad as we use small number of values!
        acc = 100*max(m.normed_accuracy for m in methods)
        npt.assert_allclose(tot, 1., rtol=acc)
        del k.weights
        del k.adjust
        est = k.fit()
        npt.assert_equal(est.total_weights, est.npts)
        npt.assert_equal(est.adjust, 1.)

    @kde_tester
    def test_grid_method_works(self, k, methods, data):
        est = k.fit()
        mesh, values = est.grid(512)
        tot = mesh.integrate(values)
        acc = max(m.grid_accuracy for m in methods)
        npt.assert_allclose(tot, 1., rtol=acc)
