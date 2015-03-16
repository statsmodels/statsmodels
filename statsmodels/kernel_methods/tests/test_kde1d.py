from __future__ import division, absolute_import, print_function

from .. import kde_methods, bandwidths
import numpy as np
from numpy.random import randn
from scipy import integrate
from . import kde_utils as kde_utils
from nose.plugins.attrib import attr
from nose.tools import assert_almost_equal, eq_, assert_greater
from .. import kde

class FakeModel(object):
    lower = -np.inf
    upper = np.inf
    weights = np.asarray(1.)

    def __init__(self, exog):
        self.exog = exog

@attr('kernel_methods')
class TestBandwidth(object):
    @classmethod
    def setUpClass(cls):
        cls.ratios = np.array([1., 2., 5.])
        d = randn(500)
        cls.vs = cls.ratios[:, np.newaxis] * np.array([d, d, d])
        cls.ss = [bandwidths._spread(X) for X in cls.vs]

    def methods(self, m):
        bws = np.asfarray([m(FakeModel(v)) for v in self.vs])
        eq_(bws.shape, (3, 1))
        rati = bws[:, 0] / self.ss
        assert_almost_equal(sum((rati - rati[0]) ** 2), 0, delta=1e-6)
        rati = bws[:, 0] / bws[0]
        assert_almost_equal(sum((rati - self.ratios) ** 2), 0, delta=1e-6)

    def test_variance_methods(self):
        yield self.methods, bandwidths.silverman
        yield self.methods, bandwidths.scotts

    def test_botev(self):
        bws = np.array([bandwidths.botev()(FakeModel(v)) for v in self.vs])
        eq_(bws.shape, (3,))
        rati = bws / self.ss
        assert_almost_equal(sum((rati - rati[0]) ** 2), 0, delta=1e-6)
        rati = bws / bws[0]
        assert_almost_equal(sum((rati - self.ratios) ** 2), 0, delta=1e-6)

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

    def test_adjust_methods(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            k.adjust = self.adjust[0]
            yield self.method_works, k, m, 'adjust_{0}_{1}'.format(k.method, 0)

    def test_adjust_grid_methods(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            k.adjust = self.adjust[0]
            yield self.grid_method_works, k, m, 'adjust_{0}_{1}'.format(k.method, 0)

    def kernel_works_(self, k):
        self.kernel_works(k, 'default')

    def test_kernels(self):
        for k in kde_utils.kernels1d:
            yield self.kernel_works_, k

    def grid_kernel_works_(self, k):
        self.grid_kernel_works(k, 'default')

    def test_grid_kernels(self):
        for k in kde_utils.kernels1d:
            yield self.grid_kernel_works_, k

@attr('kernel_methods')
class TestKDE1D(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)

    def method_works(self, k, method, name):
        est = k.fit()
        tot = integrate.quad(est.pdf, est.lower, est.upper, limit=100)[0]
        acc = method.normed_accuracy
        assert_almost_equal(tot, 1, delta=acc)
        del k.weights
        del k.adjust
        est = k.fit()
        eq_(est.total_weights, k.npts)
        eq_(est.adjust, 1.)

    def grid_method_works(self, k, method, name):
        est = k.fit()
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        acc = max(method.normed_accuracy, method.grid_accuracy)
        assert_almost_equal(tot, 1, delta=acc)

    def test_copy(self):
        k = self.createKDE(self.vs[0], self.methods[0])
        k.bandwidth = bandwidths.silverman
        xs = np.r_[self.xs.min():self.xs.max():512j]
        est = k.fit()
        ys = est(xs)
        k1 = k.copy()
        est1 = k1.fit()
        ys1 = est1(xs)
        est2 = est1.copy()
        ys2 = est2(xs)
        np.testing.assert_allclose(ys1, ys, 1e-8, 1e-8)
        np.testing.assert_allclose(ys2, ys, 1e-8, 1e-8)

    def test_bandwidths(self):
        k = self.createKDE(self.vs[0], self.methods[0])
        k.bandwidth = 0.1
        est = k.fit()
        assert_almost_equal(est.bandwidth, 0.1)
        k.bandwidth = bandwidths.botev()
        est = k.fit()
        assert_greater(est.bandwidth, 0)

    def kernel_works(self, ker, name):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        est = k.fit()
        tot = integrate.quad(est.pdf, est.lower, est.upper, limit=100)[0]
        acc = method.normed_accuracy * ker.precision_factor
        assert_almost_equal(tot, 1, delta=acc)

    def grid_kernel_works(self, ker, name):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        est = k.fit()
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        acc = max(method.grid_accuracy, method.normed_accuracy) * ker.precision_factor
        assert_almost_equal(tot, 1, delta=acc)

@attr('kernel_methods')
class LogTestKDE1D(TestKDE1D):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        cls.methods = kde_utils.methods_log

@attr('kernel_methods')
class TestSF(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        est = k.fit()
        xs = kde_methods.generate_grid1d(est, N=32)
        sf = est.sf(xs.linear())
        cdf = est.cdf(xs.linear())
        np.testing.assert_allclose(sf, 1 - cdf, method.accuracy, method.accuracy)

    def grid_method_works(self, k, method, name):
        est = k.fit()
        xs, sf = est.sf_grid()
        _, cdf = est.cdf_grid()
        np.testing.assert_allclose(sf, 1 - cdf, method.accuracy, method.accuracy)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

@attr('kernel_methods')
class TestLogSF(TestSF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        cls.methods = kde_utils.methods_log
        del cls.sizes[1:]

@attr('kernel_methods')
class TestISF(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        est = k.fit()
        sf = np.linspace(0, 1, 32)
        sf_xs = est.isf(sf)
        cdf_xs = est.icdf(1 - sf)
        acc = max(method.accuracy, method.normed_accuracy)
        np.testing.assert_allclose(sf_xs, cdf_xs, acc, acc)

    def grid_method_works(self, k, method, name):
        est = k.fit()
        comp_sf, xs = est.isf_grid()
        step = len(xs) // 16
        ref_sf = est.sf(xs[::step])
        comp_sf = comp_sf.grid[0][::step]
        acc = max(method.grid_accuracy, method.normed_accuracy)
        np.testing.assert_allclose(comp_sf, ref_sf, acc, acc)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

@attr('kernel_methods')
class TestLogISF(TestISF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]

class TestICDF(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        est = k.fit()
        quant = np.linspace(0, 1, 32)
        xs = est.icdf(quant)
        cdf_quant = est.cdf(xs)
        acc = max(method.accuracy, method.normed_accuracy)
        np.testing.assert_allclose(cdf_quant, quant, acc, acc)

    def grid_method_works(self, k, method, name):
        est = k.fit()
        comp_cdf, xs = est.icdf_grid()
        step = len(xs) // 16
        ref_cdf = est.cdf(xs[::step])
        comp_cdf = comp_cdf.grid[0][::step]
        acc = max(method.grid_accuracy, method.normed_accuracy)
        np.testing.assert_allclose(comp_cdf, ref_cdf, acc, acc)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

@attr('kernel_methods')
class TestLogICDF(TestICDF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        cls.methods = kde_utils.methods_log
        del cls.sizes[1:]


@attr('kernel_methods')
class TestHazard(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        est = k.fit()
        xs = kde_methods.generate_grid1d(est, N=32)
        h_comp = est.hazard(xs.linear())
        sf = est.sf(xs.linear())
        h_ref = est.pdf(xs.linear())
        sf = est.sf(xs.linear())
        sf[sf < 0] = 0  # Some methods can produce negative sf
        h_ref /= sf
        sel = sf > np.sqrt(method.accuracy)
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def grid_method_works(self, k, method, name):
        est = k.fit()
        xs, h_comp = est.hazard_grid()
        xs, sf = est.sf_grid()
        sf[sf < 0] = 0  # Some methods can produce negative sf
        h_ref = est.grid()[1]
        h_ref /= sf
        sel = sf > np.sqrt(method.accuracy)
        # Only tests for sf big enough or error is too large
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

@attr('kernel_methods')
class TestLogHazard(TestHazard):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]
        cls.methods = kde_utils.methods_log

@attr('kernel_methods')
class TestCumHazard(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        est = k.fit()
        xs = kde_methods.generate_grid1d(est, N=32)
        h_comp = est.cumhazard(xs.linear())
        sf = est.sf(xs.linear())
        sf[sf < 0] = 0  # Some methods can produce negative sf
        h_ref = -np.log(sf)
        sel = sf > np.sqrt(method.accuracy)
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def grid_method_works(self, k, method, name):
        est = k.fit()
        xs, h_comp = est.cumhazard_grid()
        xs, sf = est.sf_grid()
        sf[sf < 0] = 0  # Some methods can produce negative sf
        h_ref = -np.log(sf)
        sel = sf > np.sqrt(method.accuracy)
        # Only tests for sf big enough or error is too large
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

@attr('kernel_methods')
class TestLogCumHazard(TestCumHazard):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]
        cls.methods = kde_utils.methods_log

if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
