import pytest
from .. import kde, kde_methods, bandwidths
import numpy as np
import numpy.testing as npt
from numpy.random import randn
from scipy import integrate
from . import kde_utils as kde_utils
from .kde_utils import kde_tester, datasets
from ..kde_utils import GridInterpolator

class FakeModel(object):
    lower = -np.inf
    upper = np.inf
    weights = np.asarray(1.)

    def __init__(self, exog):
        self.exog = exog

class TestBandwidth(object):
    @classmethod
    def setup_class(cls):
        cls.ratios = np.array([1., 2., 5.])
        d = randn(500)
        cls.vs = cls.ratios[:, np.newaxis] * np.array([d, d, d])
        cls.ss = [bandwidths._spread(X) for X in cls.vs]

    @pytest.mark.parametrize('m', [bandwidths.silverman, bandwidths.scotts])
    def test_variance_methods(self, m):
        bws = np.asfarray([m(FakeModel(v)) for v in self.vs])
        npt.assert_equal(bws.shape, (3, 1))
        rati = bws[:, 0] / self.ss
        npt.assert_allclose(sum((rati - rati[0]) ** 2), 0, rtol=1e-6, atol=1e-6)
        rati = bws[:, 0] / bws[0]
        npt.assert_allclose(sum((rati - self.ratios) ** 2), 0, rtol=1e-6, atol=1e-6)

    def test_botev(self):
        bws = np.array([bandwidths.Botev()(FakeModel(v)) for v in self.vs])
        npt.assert_equal(bws.shape, (3,))
        rati = bws / self.ss
        npt.assert_allclose(sum((rati - rati[0]) ** 2), 0, rtol=1e-6, atol=1e-6)
        rati = bws / bws[0]
        npt.assert_allclose(sum((rati - self.ratios) ** 2), 0, rtol=1e-6, atol=1e-6)

def make_name(param_name, method):
    return "{0}_{1}".format(param_name, method.instance.name)

all_methods_data = kde_utils.generate_methods_data(['norm', 'lognorm'])
all_methods_small_data = kde_utils.generate_methods_data(['norm', 'lognorm'], indices=[0])

@pytest.mark.parametrize(kde_utils.kde_tester_args, all_methods_data)
class TestKDE1D(object):

    @kde_tester
    def test_method_works(self, k, method, data):
        est = k.fit()
        npt.assert_equal(est.ndim, 1)
        tot = integrate.quad(est.pdf, est.lower, est.upper, limit=100)[0]
        acc = method.normed_accuracy
        npt.assert_allclose(tot, 1, rtol=acc, atol=acc)
        adjust = est.adjust.copy()
        weights = est.weights.copy()
        del est.weights
        del est.adjust
        npt.assert_equal(est.total_weights, k.npts)
        npt.assert_equal(est.adjust, 1.)
        est.adjust = adjust  # Try to set the adjust
        est.weights = weights
        est.upper = k.upper
        est.lower = k.lower
        npt.assert_equal(est.lower, float(k.lower))
        npt.assert_equal(est.upper, float(k.upper))

    @kde_tester
    def grid_method_works(self, k, method, data):
        est = k.fit()
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        acc = max(method.normed_accuracy, method.grid_accuracy)
        npt.assert_allclose(tot, 1, rtol=acc, atol=acc)

class TestKDE1DExtra(object):

    @staticmethod
    @pytest.fixture
    def small_kde(request, datasets):
        obj = request.instance
        try:
            method = request.param
        except AttributeError:
            method = obj.method
        return kde_utils.createKDE(obj.params, obj.data, obj.small_vs, method), method

    @staticmethod
    @pytest.fixture
    def large_kde(request, datasets):
        obj = request.instance
        try:
            method = request.param
        except AttributeError:
            method = obj.method
        return kde_utils.createKDE(obj.params, obj.data, obj.large_vs, method), method

    @classmethod
    def setup_class(cls):
        params = kde_utils.knownParameters['norm']
        cls.params = params
        cls.method = params.methods[0]
        cls.data = params.create_dataset()
        cls.small_vs = cls.data.vs[0]
        cls.large_vs = cls.data.vs[1]

    def test_copy(self, small_kde):
        k, _ = small_kde
        k.bandwidth = bandwidths.silverman
        xs = np.r_[self.params.xs.min():self.params.xs.max():512j]
        est = k.fit()
        ys = est(xs)
        k1 = k.copy()
        est1 = k1.fit()
        ys1 = est1(xs)
        est2 = est1.copy()
        ys2 = est2(xs)
        npt.assert_allclose(ys1, ys, 1e-8, 1e-8)
        npt.assert_allclose(ys2, ys, 1e-8, 1e-8)

    def test_bandwidths(self, small_kde):
        k, _ = small_kde
        k.bandwidth = 0.1
        est = k.fit()
        npt.assert_allclose(est.bandwidth, 0.1)
        k.bandwidth = bandwidths.Botev()
        est = k.fit()
        assert est.bandwidth > 0

    @pytest.mark.parametrize('ker', kde_utils.kernels1d)
    def test_kernels(self, large_kde, ker):
        k, _ = large_kde
        k.kernel = ker.cls()
        est = k.fit()
        tot = integrate.quad(est.pdf, est.lower, est.upper, limit=100)[0]
        acc = self.method.normed_accuracy * ker.precision_factor
        npt.assert_allclose(tot, 1, rtol=acc, atol=acc)

    @pytest.mark.parametrize('ker', kde_utils.kernels1d)
    def test_grid_kernels(self, large_kde, ker):
        k, _ = large_kde
        k.kernel = ker.cls()
        est = k.fit()
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        acc = max(self.method.grid_accuracy, self.method.normed_accuracy) * ker.precision_factor
        npt.assert_allclose(tot, 1, rtol=acc, atol=acc)
        npt.assert_equal(type(est.kernel), type(k.kernel.for_ndim(1)))

    @pytest.mark.parametrize('small_kde', kde_utils.methods_1d, indirect=True)
    def test_bad_set_axis(self, small_kde):
        k, _ = small_kde
        with pytest.raises(ValueError):
            k.method.axis_type = 'O'

    @pytest.mark.parametrize('small_kde', kde_utils.methods_1d, indirect=True)
    def test_set_axis(self, small_kde):
        k, _ = small_kde
        k.method.axis_type = 'C'

    @pytest.mark.parametrize('small_kde', kde_utils.methods_1d, indirect=True)
    def test_force_span(self, small_kde):
        k, m = small_kde
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
        npt.assert_allclose(y1, y2, rtol=1e-4, atol=1e-4)

    def test_non1d_data(self):
        d = np.array([[1, 2], [3, 4], [5, 6.]])
        k = kde.KDE(d, method=kde_methods.Reflection1D)
        with pytest.raises(ValueError):
            k.fit()

    def test_bad_axis_type(self):
        k = kde.KDE(self.small_vs, method=kde_methods.Reflection1D, axis_type='O')
        with pytest.raises(ValueError):
            k.fit()

    def test_change_exog(self, small_kde):
        k, _ = small_kde
        est = k.fit()
        xs1, ys1 = est.grid()
        est.exog = self.small_vs*3
        xs2, ys2 = est.grid()
        est.exog = self.small_vs
        xs3, ys3 = est.grid()

        assert sum((ys1 - ys2)**2) > 0
        npt.assert_allclose(ys1, ys3, rtol=1e-8, atol=1e-8)
        npt.assert_allclose(xs1.full(), xs3.full(), rtol=1e-8, atol=1e-8)

    def test_bad_change_exog(self):
        k = kde_utils.createKDE(self.params, self.data, self.small_vs, self.method)
        est = k.fit()
        with pytest.raises(ValueError):
            est.exog = self.large_vs

    @pytest.mark.parametrize('large_kde', kde_utils.methods_1d, indirect=True)
    def test_update_inputs(self, large_kde):
        k, m = large_kde
        k.weights = self.data.weights[1]
        k.adjust = self.data.adjust[1]
        est = k.fit()
        est.update_inputs(self.data.vs[1][:-5],
                          weights=self.data.weights[1][:-5],
                          adjust=self.data.adjust[1][:-5])
        xs, ys = est.grid()
        tot = xs.integrate(ys)
        npt.assert_allclose(tot, 1, rtol=m.grid_accuracy, atol=m.grid_accuracy)

    @pytest.mark.parametrize('large_kde', kde_utils.methods_1d, indirect=True)
    def test_bad_update_inputs1(self, large_kde):
        k, _ = large_kde
        k.weights = self.data.weights[1]
        k.adjust = self.data.adjust[1]
        est = k.fit()
        with pytest.raises(ValueError):
            est.update_inputs(self.data.vs[1][:-5],
                              weights=self.data.weights[1],
                              adjust=self.data.adjust[1][:-5])

    def test_bad_update_inputs2(self, large_kde):
        k, _ = large_kde
        k.weights = self.data.weights[1]
        k.adjust = self.data.adjust[1]
        est = k.fit()
        with pytest.raises(ValueError):
            est.update_inputs(self.data.vs[1][:-5],
                              weights=self.data.weights[1][:-5],
                              adjust=self.data.adjust[1])

    def test_bad_update_inputs3(self, large_kde):
        k, _ = large_kde
        k.weights = self.data.weights[1]
        k.adjust = self.data.adjust[1]
        est = k.fit()
        with pytest.raises(ValueError):
            est.update_inputs([[1, 2], [2, 3], [3, 4]],
                              weights=self.data.weights[1][:-5],
                              adjust=self.data.adjust[1])

    def test_transform(self):
        tr = kde_methods.create_transform(np.log, np.exp)
        log_tr = kde_methods.LogTransform

        xs = np.r_[1:3:16j]
        tol = 1e-6
        npt.assert_allclose(tr.Dinv(xs), log_tr.Dinv(xs), rtol=tol, atol=tol)

        class MyTransform(object):
            def __call__(self, x):
                return np.log(x)

            def inv(self, x):
                return np.exp(x)

            def Dinv(self, x):
                return np.exp(x)

        tr1 = kde_methods.create_transform(MyTransform())
        npt.assert_allclose(tr1.Dinv(xs), log_tr.Dinv(xs), rtol=tol, atol=tol)

    def test_bad_transform(self):
        with pytest.raises(AttributeError):
            kde_methods.create_transform(np.log)


@pytest.mark.parametrize(kde_utils.kde_tester_args, all_methods_small_data)
class TestSF(object):
    @kde_tester
    def test_method_works(self, k, method, data):
        est = k.fit()
        xs = kde_methods.generate_grid1d(est, N=32)
        sf = est.sf(xs.linear())
        cdf = est.cdf(xs.linear())
        npt.assert_allclose(sf, 1 - cdf, method.accuracy, method.accuracy)

    @kde_tester
    def test_grid_method_works(self, k, method, data):
        est = k.fit()
        xs, sf = est.sf_grid()
        _, cdf = est.cdf_grid()
        npt.assert_allclose(sf, 1 - cdf, method.accuracy, method.accuracy)

@pytest.mark.parametrize(kde_utils.kde_tester_args, all_methods_small_data)
class TestISF(object):

    @kde_tester
    def test_method_works(self, k, method, data):
        est = k.fit()
        sf = np.linspace(0, 1, 32)
        sf_xs = est.isf(sf)
        cdf_xs = est.icdf(1 - sf)
        acc = max(method.accuracy, method.normed_accuracy)
        npt.assert_allclose(sf_xs, cdf_xs, acc, acc)

    @kde_tester
    def test_grid_method_works(self, k, method, data):
        est = k.fit()
        comp_sf, xs = est.isf_grid()
        step = len(xs) // 16
        ref_sf = est.sf(xs[::step])
        comp_sf = comp_sf.grid[0][::step]
        acc = max(method.grid_accuracy, method.normed_accuracy)
        npt.assert_allclose(comp_sf, ref_sf, acc, acc)

@pytest.mark.parametrize(kde_utils.kde_tester_args, all_methods_small_data)
class TestICDF(object):

    @kde_tester
    def test_method_works(self, k, method, data):
        est = k.fit()
        quant = np.linspace(0, 1, 32)
        xs = est.icdf(quant)
        cdf_quant = est.cdf(xs)
        acc = max(method.accuracy, method.normed_accuracy)
        npt.assert_allclose(cdf_quant, quant, acc, acc)

    @kde_tester
    def test_grid_method_works(self, k, method, data):
        est = k.fit()
        comp_cdf, xs = est.icdf_grid()
        step = len(xs) // 16
        ref_cdf = est.cdf(xs[::step])
        comp_cdf = comp_cdf.grid[0][::step]
        acc = max(method.grid_accuracy, method.normed_accuracy)
        npt.assert_allclose(comp_cdf, ref_cdf, acc, acc)


@pytest.mark.parametrize(kde_utils.kde_tester_args, all_methods_small_data)
class TestHazard(object):

    @kde_tester
    def test_method_works(self, k, method, data):
        est = k.fit()
        xs = kde_methods.generate_grid1d(est, N=32)
        h_comp = est.hazard(xs.linear())
        h_ref = est.pdf(xs.linear())
        sf = est.sf(xs.linear())
        # Only tests for sf big enough or error is too large
        sel = sf > np.sqrt(method.accuracy)
        sf = sf[sel]
        h_ref = h_ref[sel]
        h_ref /= sf
        npt.assert_allclose(h_comp[sel], h_ref, method.accuracy, method.accuracy)

    @kde_tester
    def test_grid_method_works(self, k, method, data):
        est = k.fit()
        xs, h_comp = est.hazard_grid()
        xs, sf = est.sf_grid()
        # Only tests for sf big enough or error is too large
        sel = sf > np.sqrt(method.accuracy)
        sf = sf[sel]
        h_ref = est.grid()[1][sel]
        h_ref /= sf
        npt.assert_allclose(h_comp[sel], h_ref, method.accuracy, method.accuracy)


@pytest.mark.parametrize(kde_utils.kde_tester_args, all_methods_small_data)
class TestCumHazard(object):

    @kde_tester
    def test_method_works(self, k, method, data):
        est = k.fit()
        xs = kde_methods.generate_grid1d(est, N=32)
        h_comp = est.cumhazard(xs.linear())
        sf = est.sf(xs.linear())
        # Only tests for sf big enough or error is too large
        sel = sf > np.sqrt(method.accuracy)
        sf = sf[sel]
        h_ref = -np.log(sf)
        npt.assert_allclose(h_comp[sel], h_ref, method.accuracy, method.accuracy)

    @kde_tester
    def test_grid_method_works(self, k, method, data):
        est = k.fit()
        xs, h_comp = est.cumhazard_grid()
        xs, sf = est.sf_grid()
        # Only tests for sf big enough or error is too large
        sel = sf > np.sqrt(method.accuracy)
        sf = sf[sel]
        h_ref = -np.log(sf)
        # Only tests for sf big enough or error is too large
        npt.assert_allclose(h_comp[sel], h_ref, method.accuracy, method.accuracy)
