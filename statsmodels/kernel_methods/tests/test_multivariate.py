from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ...tools.testing import assert_equal, assert_allclose
from . import kde_utils
from .. import kde, kde_methods


class KDETester(object):
    def createKDE(self, data, methods, **args):
        all_args = dict(self.args)
        all_args.update(args)
        d = data.copy()
        k = kde.KDE(d, **all_args)
        mv = kde_methods.Multivariate()
        k.method = mv

        n = len(methods)
        axis_type = ''
        lower = [-np.inf]*n
        upper = [np.inf]*n
        for i, m in enumerate(methods):
            method_instance = m.instance()
            mv.methods[i] = method_instance
            axis_type += str(method_instance.axis_type)
            if method_instance.axis_type != 'c':
                d[:, i] = np.round(d[:, i])
            if m.bound_low:
                lower[i] = self.lower[i]
            if m.bound_high:
                upper[i] = self.upper[i]
        k.exog = d
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
                yield (self.method_works, k, [m1, m2],
                       '{0}_{1}'.format(name, j))

    def test_grid_methods(self):
        for i in range(self.nb_methods):
            m1 = self.methods1[i]
            m2 = self.methods2[i]
            name = "{0}_{1}".format(m1.instance.name, m2.instance.name)
            for j in range(len(self.sizes)):
                k = self.createKDE(self.vs[j], [m1, m2])
                yield (self.grid_method_works, k, [m1, m2],
                       '{0}_{1}'.format(name, j))

    def test_weights_methods(self):
        for i in range(self.nb_methods):
            m1 = self.methods1[i]
            m2 = self.methods2[i]
            name = "{0}_{1}".format(m1.instance.name, m2.instance.name)
            for j in range(len(self.sizes)):
                k = self.createKDE(self.vs[j], [m1, m2])
                k.weights = self.weights[j]
                yield (self.method_works, k, [m1, m2],
                       '{0}_{1}'.format(name, j))

    def test_weights_grid_methods(self):
        for i in range(self.nb_methods):
            m1 = self.methods1[i]
            m2 = self.methods2[i]
            name = "{0}_{1}".format(m1.instance.name, m2.instance.name)
            for j in range(len(self.sizes)):
                k = self.createKDE(self.vs[j], [m1, m2])
                k.weights = self.weights[j]
                yield (self.grid_method_works, k, [m1, m2],
                       '{0}_{1}'.format(name, j))


class TestMultivariate(KDETester):
    @classmethod
    def setup_class(cls):
        kde_utils.setup_class_multivariate(cls)

    def method_works(self, k, methods, name):
        est = k.fit()
        bt = est.bin_type
        bounds = [None, None]
        if est.methods[0].bin_type == 'd':
            bounds[0] = [est.lower[0], est.upper[0]]
        else:
            if methods[0].bound_low:
                low = self.lower[0]
            else:
                low = est.exog[:, 0].min() - 5*est.bandwidth[0]
            if methods[0].bound_high:
                high = self.upper[0]
            else:
                high = est.exog[:, 0].max() + 5*est.bandwidth[0]
            bounds[0] = [low, high]
        if est.methods[1].bin_type == 'd':
            bounds[1] = [est.lower[1], est.upper[1]]
        else:
            if methods[1].bound_low:
                low = self.lower[1]
            else:
                low = est.exog[:, 1].min() - 5*est.bandwidth[1]
            if methods[1].bound_high:
                high = self.upper[1]
            else:
                high = est.exog[:, 1].max() + 5*est.bandwidth[1]
            bounds[1] = [low, high]
        grid = kde_utils.Grid.fromBounds(bounds, bin_type=bt, shape=128,
                                         dtype=float)
        values = est(grid.linear()).reshape(grid.shape)
        tot = grid.integrate(values)
        # Note: the precision is quite bad as we use small number of values!
        acc = 100*max(m.normed_accuracy for m in methods)
        assert_allclose(tot, 1., rtol=acc)
        del k.weights
        del k.adjust
        est = k.fit()
        assert_equal(est.total_weights, est.npts)
        assert_equal(est.adjust, 1.)

    def grid_method_works(self, k, methods, name):
        est = k.fit()
        mesh, values = est.grid(512)
        tot = mesh.integrate(values)
        acc = max(m.grid_accuracy for m in methods)
        assert_allclose(tot, 1., rtol=acc)
