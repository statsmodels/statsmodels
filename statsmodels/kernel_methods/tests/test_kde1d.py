from __future__ import division, absolute_import, print_function

from .. import kde, kde_methods, bandwidths
import numpy as np
from numpy.random import randn
from scipy import integrate
from . import kde_utils as kde_utils
from ...tools.testing import assert_allclose, assert_equal, assert_raises
from ..kde_utils import GridInterpolator


class FakeModel(object):
    lower = -np.inf
    upper = np.inf
    weights = np.asarray(1.)

    def __init__(self, exog):
        self.exog = exog


class TestBandwidth(object):
    @classmethod
    def setUpClass(cls):
        cls.ratios = np.array([1., 2., 5.])
        d = randn(500)
        cls.vs = cls.ratios[:, np.newaxis] * np.array([d, d, d])
        cls.ss = [bandwidths._spread(X) for X in cls.vs]

    def methods(self, m):
        bws = np.asfarray([m(FakeModel(v)) for v in self.vs])
        assert_equal(bws.shape, (3, 1))
        rati = bws[:, 0] / self.ss
        assert_allclose(sum((rati - rati[0]) ** 2), 0, rtol=1e-6, atol=1e-6)
        rati = bws[:, 0] / bws[0]
        assert_allclose(sum((rati - self.ratios) ** 2), 0, rtol=1e-6, atol=1e-6)

    def test_variance_methods(self):
        yield self.methods, bandwidths.silverman
        yield self.methods, bandwidths.scotts

    def test_botev(self):
        bws = np.array([bandwidths.botev()(FakeModel(v)) for v in self.vs])
        assert_equal(bws.shape, (3,))
        rati = bws / self.ss
        assert_allclose(sum((rati - rati[0]) ** 2), 0, rtol=1e-6, atol=1e-6)
        rati = bws / bws[0]
        assert_allclose(sum((rati - self.ratios) ** 2), 0, rtol=1e-6, atol=1e-6)


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


class TestKDE1D(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)

    def method_works(self, k, method, name):
        est = k.fit()
        assert_equal(est.ndim, 1)
        tot = integrate.quad(est.pdf, est.lower, est.upper, limit=100)[0]
        acc = method.normed_accuracy
        assert_allclose(tot, 1, rtol=acc, atol=acc)
        adjust = est.adjust.copy()
        weights = est.weights.copy()
        del est.weights
        del est.adjust
        assert_equal(est.total_weights, k.npts)
        assert_equal(est.adjust, 1.)
        est.adjust = adjust  # Try to set the adjust
        est.weights = weights
        est.upper = k.upper
        est.lower = k.lower
        assert_equal(est.lower, float(k.lower))
        assert_equal(est.upper, float(k.upper))

    def grid_method_works(self, k, method, name):
        est = k.fit()
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        acc = max(method.normed_accuracy, method.grid_accuracy)
        assert_allclose(tot, 1, rtol=acc, atol=acc)

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
        assert_allclose(est.bandwidth, 0.1)
        k.bandwidth = bandwidths.botev()
        est = k.fit()
        assert est.bandwidth > 0

    def kernel_works(self, ker, name):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        est = k.fit()
        tot = integrate.quad(est.pdf, est.lower, est.upper, limit=100)[0]
        acc = method.normed_accuracy * ker.precision_factor
        assert_allclose(tot, 1, rtol=acc, atol=acc)

    def grid_kernel_works(self, ker, name):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        est = k.fit()
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        acc = max(method.grid_accuracy, method.normed_accuracy) * ker.precision_factor
        assert_allclose(tot, 1, rtol=acc, atol=acc)
        assert_equal(type(est.kernel), type(k.kernel.for_ndim(1)))

    def bad_set_axis(self, k, m, name):
        with assert_raises(ValueError):
            k.method.axis_type = 'o'

    def set_axis(self, k, m, name):
        k.method.axis_type = 'c'

    def test_set_axis(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            yield self.set_axis, k, m, 'adjust_{0}_{1}'.format(k.method, 0)

    def test_bad_set_axis(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            yield self.bad_set_axis, k, m, 'adjust_{0}_{1}'.format(k.method, 0)

    def force_span(self, k, m, name):
        est = k.fit()
        span = [est.lower, est.upper]
        if not m.bound_low:
            span[0] = -10
        if not m.bound_high:
            span[1] = 10
        xs, ys = est.grid(span=span)
        x = np.r_[-2:2:64j]
        x = x[(x > est.lower) & (x < est.upper)]
        y1 = est.pdf(x)
        interp = GridInterpolator(xs, ys)
        y2 = interp(x)
        assert_allclose(y1, y2, rtol=1e-4, atol=1e-4)

    def test_force_span(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            yield self.force_span, k, m, str(k.method)

    def test_non1d_data(self):
        d = np.array([[1, 2], [3, 4], [5, 6.]])
        k = kde.KDE(d, method=kde_methods.Reflection1D)
        assert_raises(ValueError, k.fit)

    def test_bad_axis_type(self):
        k = kde.KDE(self.vs[0], method=kde_methods.Reflection1D, axis_type='o')
        assert_raises(ValueError, k.fit)

    def test_change_exog(self):
        k = self.createKDE(self.vs[0], self.methods[0])
        est = k.fit()
        xs1, ys1 = est.grid()
        est.exog = self.vs[0]*3
        xs2, ys2 = est.grid()
        est.exog = self.vs[0]
        xs3, ys3 = est.grid()

        assert sum((ys1 - ys2)**2) > 0
        assert_allclose(ys1, ys3, rtol=1e-8, atol=1e-8)
        assert_allclose(xs1.full(), xs3.full(), rtol=1e-8, atol=1e-8)

    def test_bad_change_exog(self):
        k = self.createKDE(self.vs[0], self.methods[0])
        est = k.fit()
        with assert_raises(ValueError):
            est.exog = self.vs[1]

    def update_inputs(self, k, m, name):
        est = k.fit()
        est.update_inputs(self.vs[1][:-5],
                          weights=self.weights[1][:-5],
                          adjust=self.adjust[1][:-5])
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        assert_allclose(tot, 1, rtol=m.grid_accuracy, atol=m.grid_accuracy)

    def test_update_inputs(self):
        for m in self.methods:
            k = self.createKDE(self.vs[1], m)
            k.weights = self.weights[1]
            k.adjust = self.adjust[1]
            yield self.update_inputs, k, m, str(k.method)

    def bad_update_inputs1(self, k, m, name):
        est = k.fit()
        assert_raises(ValueError, est.update_inputs,
            self.vs[1][:-5],
            weights=self.weights[1],
            adjust=self.adjust[1][:-5]
        )

    def bad_update_inputs2(self, k, m, name):
        est = k.fit()
        assert_raises(ValueError, est.update_inputs,
            self.vs[1][:-5],
            weights=self.weights[1][:-5],
            adjust=self.adjust[1]
        )

    def bad_update_inputs3(self, k, m, name):
        est = k.fit()
        assert_raises(ValueError, est.update_inputs,
            [[1, 2], [2, 3], [3, 4]],
            weights=self.weights[1][:-5],
            adjust=self.adjust[1]
        )

    def test_bad_update_inputs(self):
        for m in self.methods:
            k = self.createKDE(self.vs[1], m)
            k.weights = self.weights[1]
            k.adjust = self.adjust[1]
            yield self.bad_update_inputs1, k, m, str(k.method) + "_bad1"
            yield self.bad_update_inputs2, k, m, str(k.method) + "_bad2"
            yield self.bad_update_inputs3, k, m, str(k.method) + "_bad3"


class TestLogKDE1D(TestKDE1D):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)

    def test_transform(self):
        tr = kde_methods.create_transform(np.log, np.exp)
        log_tr = kde_methods.LogTransform

        xs = np.r_[1:3:16j]
        tol = 1e-6
        assert_allclose(tr.Dinv(xs), log_tr.Dinv(xs), rtol=tol, atol=tol)

        class MyTransform(object):
            def __call__(self, x):
                return np.log(x)

            def inv(self, x):
                return np.exp(x)

            def Dinv(self, x):
                return np.exp(x)

        tr1 = kde_methods.create_transform(MyTransform())
        assert_allclose(tr1.Dinv(xs), log_tr.Dinv(xs), rtol=tol, atol=tol)

    def test_bad_transform1(self):
        assert_raises(AttributeError, kde_methods.create_transform, np.log)


class TestSF(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
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


class TestLogSF(TestSF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]


class TestISF(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
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


class TestLogISF(TestISF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]


class TestICDF(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
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


class TestLogICDF(TestICDF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]


class TestHazard(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
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


class TestLogHazard(TestHazard):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]


class TestCumHazard(KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
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


class TestLogCumHazard(TestCumHazard):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]

if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
