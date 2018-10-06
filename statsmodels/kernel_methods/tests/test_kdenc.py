from __future__ import division, absolute_import, print_function

from . import kde_utils
from ...tools.testing import assert_allclose
from .. import kde
import numpy as np

class KDETester(object):
    def createKDE(self, data, method, **args):
        all_args = dict(self.args)
        all_args.update(args)
        k = kde.KDE(data, **all_args)
        k.method = method.instance
        k.axis_type = k.method.axis_type
        k.bandwidth = 0.2
        return k

    def test_methods(self):
        for m in self.methods:
            for i in range(len(self.sizes)):
                k = self.createKDE(self.vs[i], m)
                yield self.method_works, k, m, '{0}_{1}'.format(k.method, i)

    def test_grid_methods(self):
        for m in self.methods:
            for i in range(len(self.sizes)):
                k = self.createKDE(self.vs[i], m)
                yield self.grid_method_works, k, m, '{0}_{1}'.format(k.method, i)

    def test_weights_methods(self):
        for m in self.methods:
            for i in range(len(self.sizes)):
                k = self.createKDE(self.vs[i], m)
                k.weights = self.weights[i]
                yield self.method_works, k, m, 'weights_{0}_{1}'.format(k.method, i)

    def test_weights_grid_methods(self):
        for m in self.methods:
            for i in range(len(self.sizes)):
                k = self.createKDE(self.vs[i], m)
                k.weights = self.weights[i]
                yield self.grid_method_works, k, m, 'weights_{0}_{1}'.format(k.method, i)

class TestNonContinuous(KDETester):
    @classmethod
    def setup_class(cls):
        kde_utils.setup_class_nc(cls)

    def method_works(self, k, method, name):
        est = k.fit()
        xs = np.arange(est.lower, est.upper+1)
        ys = est(xs)
        tot = ys.sum()
        assert_allclose(tot, 1, rtol=1e-3)

    def grid_method_works(self, k, method, name):
        est = k.fit()
        mesh, values = est.grid()
        tot = mesh.integrate(values)
        assert_allclose(tot, 1, rtol=1e-4)
