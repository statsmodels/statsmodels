from __future__ import division, absolute_import, print_function

from . import kde_utils
from nose.plugins.attrib import attr
from ...tools.testing import assert_equal, assert_allclose
from .. import kde, kde_methods
import numpy as np

class KDETester(object):
    def createKDE(self, data, methods, **args):
        all_args = dict(self.args)
        all_args.update(args)
        k = kde.KDE(data, **all_args)
        mv = kde_methods.MultivariateKDE()
        k.method = mv

        n = len(methods)
        axis_type = ''
        lower = [-np.inf]*n
        upper = [np.inf]*n
        for i, m in enumerate(methods):
            method_instance = m.instance()
            mv.methods[i] = method_instance
            axis_type += str(method_instance.axis_type)
            if m.bound_low:
                lower[i] = self.lower[i]
            if m.bound_high:
                upper[i] = self.upper[i]
        k.axis_type = axis_type
        k.lower = lower
        k.upper = upper

        return k

    def test_methods(self):
        for i in range(self.nb_methods):
            m1 = self.methods1[i]
            m2 = self.methods2[i]
            name = "{0}_{1}".format(m1.instance.name, m2.instance.name)
            for j in range(len(self.sizes)):
                k = self.createKDE(self.vs[j], [m1, m2])
                yield self.method_works, k, [m1, m2], '{0}_{1}'.format(name, j)

    def test_grid_methods(self):
        for i in range(self.nb_methods):
            m1 = self.methods1[i]
            m2 = self.methods2[i]
            name = "{0}_{1}".format(m1.instance.name, m2.instance.name)
            for j in range(len(self.sizes)):
                k = self.createKDE(self.vs[j], [m1, m2])
                yield self.grid_method_works, k, [m1, m2], '{0}_{1}'.format(name, j)

    def test_weights_methods(self):
        for i in range(self.nb_methods):
            m1 = self.methods1[i]
            m2 = self.methods2[i]
            name = "{0}_{1}".format(m1.instance.name, m2.instance.name)
            for j in range(len(self.sizes)):
                k = self.createKDE(self.vs[j], [m1, m2])
                k.weights = self.weights[j]
                yield self.method_works, k, [m1, m2], '{0}_{1}'.format(name, j)

    def test_weights_grid_methods(self):
        for i in range(self.nb_methods):
            m1 = self.methods1[i]
            m2 = self.methods2[i]
            name = "{0}_{1}".format(m1.instance.name, m2.instance.name)
            for j in range(len(self.sizes)):
                k = self.createKDE(self.vs[j], [m1, m2])
                k.weights = self.weights[j]
                yield self.grid_method_works, k, [m1, m2], '{0}_{1}'.format(name, j)

@attr("kernel_methods")
class TestMultivariate(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_multivariate(cls)

    def method_works(self, k, methods, name):
        est = k.fit()
        values = est(self.grid.linear()).reshape(self.grid.shape)
        tot = self.grid.integrate(values)
        acc = max(m.normed_accuracy for m in methods)
        assert_allclose(tot, 1., rtol=acc)
        del k.weights
        del k.adjust
        est = k.fit()
        assert_equal(est.total_weights, k.npts)
        assert_equal(est.adjust, 1.)

    def grid_method_works(self, k, methods, name):
        est = k.fit()
        mesh, values = est.grid(512)
        tot = mesh.integrate(values)
        acc = max(m.normed_accuracy for m in methods)
        assert_allclose(tot, 1., rtol=acc)
