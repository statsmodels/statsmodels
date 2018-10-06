from __future__ import division, absolute_import, print_function

from . import kde_utils
from ...tools.testing import assert_equal, assert_allclose
from .. import kde

class KDETester(object):
    def createKDE(self, data, method, **args):
        all_args = dict(self.args)
        all_args.update(args)
        k = kde.KDE(data, **all_args)
        if method.instance is None:
            del k.method
        else:
            k.method = method.instance
        if method.bound_low:
            k.lower = self.lower
        else:
            del k.lower
        if method.bound_high:
            k.upper = self.upper
        else:
            del k.upper
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

    #def test_adjust_methods(self):
        #for m in self.methods:
            #k = self.createKDE(self.vs[0], m)
            #k.adjust = self.adjust[0]
            #yield self.method_works, k, m, 'adjust_{0}_{1}'.format(k.method, 0)

    #def test_adjust_grid_methods(self):
        #for m in self.methods:
            #k = self.createKDE(self.vs[0], m)
            #k.adjust = self.adjust[0]
            #yield self.grid_method_works, k, m, 'adjust_{0}_{1}'.format(k.method, 0)

    #def kernel_works_(self, k):
        #self.kernel_works(k, 'default')

    #def test_kernels(self):
        #for k in kde_utils.kernels1d:
            #yield self.kernel_works_, k

    #def grid_kernel_works_(self, k):
        #self.grid_kernel_works(k, 'default')

    #def test_grid_kernels(self):
        #for k in kde_utils.kernels1d:
            #yield self.grid_kernel_works_, k


class TestKDE2D(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_normnd(cls, 2)

    def method_works(self, k, method, name):
        """
        Enure the sum of the kernel over its domain is unity
        """
        est = k.fit()
        val = est([0, 0])
        assert val >= 0
        del k.weights
        del k.adjust
        est = k.fit()
        assert_equal(est.total_weights, k.npts)
        assert_equal(est.adjust, 1.)

    def grid_method_works(self, k, method, name):
        """
        Enure the sum of the kernel over its domain is unity
        """
        est = k.fit()
        mesh, vals = est.grid(N=32)
        tot = mesh.integrate(vals)
        acc = max(method.normed_accuracy, method.grid_accuracy)
        assert_allclose(tot, 1, rtol=acc, atol=acc)
        del k.weights
        del k.adjust
        est = k.fit()
        assert_equal(est.total_weights, k.npts)
        assert_equal(est.adjust, 1.)
