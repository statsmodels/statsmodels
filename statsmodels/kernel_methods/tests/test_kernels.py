from __future__ import division, absolute_import, print_function

from .. import kernels

from scipy import stats, integrate
import numpy as np
from . import kde_utils
from nose.plugins.attrib import attr

class RefKernel1D(kernels.Kernel1D):
    """
    Reference kernel: force use of explicit integration
    """
    def __init__(self, kernel):
        self.lower = kernel.lower
        self.upper = kernel.upper
        self.cut = kernel.cut
        self.real_kernel = kernel

    def pdf(self, z, out=None):
        return self.real_kernel.pdf(z, out)

tol = 1e-8

@attr("kernel_methods")
class TestKernels(object):
    @classmethod
    def setUpClass(cls, lower=-np.inf, test_width=3):
        cls.lower = float(lower)
        cls.hard_points = ()
        cls.quad_args = dict(limit=100)
        cls.xs = np.r_[-test_width:test_width:17j]
        bw = 0.2
        R = 10
        N = 2**16
        dx = R/(bw*N)
        cls.dx = dx
        cls.N = N
        cls.small = np.array([-5, -1, -0.5, 0, 0.5, 1, 5])

    def unity(self, kernel):
        ker = kernel.cls()
        total = integrate.quad(ker.pdf, -np.inf, np.inf)[0]
        assert abs(total-1) < tol*kernel.precision_factor

    def mean(self, kernel):
        ker = kernel.cls()

        def f(x):
            return x*ker.pdf(x)
        total = integrate.quad(f, -np.inf, np.inf)[0]
        assert abs(total) < tol*kernel.precision_factor

    def variance(self, kernel):
        ker = kernel.cls()

        def f(x):
            return x*x*ker.pdf(x)
        total = integrate.quad(f, -np.inf, np.inf)[0]
        assert abs(total-kernel.var) < tol*kernel.precision_factor

    def cdf(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.cdf(self.xs)
        val = ker.cdf(self.xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)
        tot = ker.cdf(np.inf)
        assert abs(tot-1) < acc, "ker.cdf(inf) = {0}, while it should be close to 1".format(tot)
        short1 = ker.cdf(self.small)
        short2 = [float(ker.cdf(x)) for x in self.small]
        np.testing.assert_allclose(short1, short2, acc, acc)

    def pm1(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm1(self.xs)
        val = ker.pm1(self.xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)
        tot = ker.pm1(np.inf)
        assert abs(tot) < acc, "ker.cdf(inf) = {0}, while it should be close to 0".format(tot)
        short1 = ker.pm1(self.small)
        short2 = [float(ker.pm1(x)) for x in self.small]
        np.testing.assert_allclose(short1, short2, acc, acc)

    def pm2(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm2(self.xs)
        val = ker.pm2(self.xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)
        tot = ker.pm2(np.inf)
        assert abs(tot - kernel.var) < acc, "ker.cdf(inf) = {0}, expected: {1}".format(tot, kernel.var)
        short1 = ker.pm2(self.small)
        short2 = [float(ker.pm2(x)) for x in self.small]
        np.testing.assert_allclose(short1, short2, acc, acc)

    def rfft(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.rfft(self.N, self.dx)
        val = ker.rfft(self.N, self.dx)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)

    def dct(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.dct(self.N, self.dx)
        val = ker.dct(self.N, self.dx)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)

    def test_unity(self):
        for kernel in kde_utils.kernels1d:
            yield self.unity, kernel

    def test_mean(self):
        for kernel in kde_utils.kernels1d:
            yield self.unity, kernel

    def test_variance(self):
        for kernel in kde_utils.kernels1d:
            yield self.variance, kernel

    def test_pm1(self):
        for kernel in kde_utils.kernels1d:
            yield self.pm1, kernel

    def test_pm2(self):
        for kernel in kde_utils.kernels1d:
            yield self.pm2, kernel

    def test_dct(self):
        for kernel in kde_utils.kernels1d:
            yield self.dct, kernel

    def test_fft(self):
        for kernel in kde_utils.kernels1d:
            yield self.rfft, kernel


@attr("kernel_methods")
class TestNormal1d(object):
    @classmethod
    def setUpClass(cls, lower=-np.inf):
        cls.kernel = kernels.normal1d()
        test_width = cls.kernel.cut
        cls.norm_ref = stats.norm(loc=0, scale=1)
        cls.xs = np.r_[-test_width / 2:test_width / 2:17j]

    def attr(self, attr):
        n_ref = self.norm_ref
        n_tst = self.kernel
        ref_vals = getattr(n_ref, attr)(self.xs)
        tst_vals = getattr(n_tst, attr)(self.xs)
        np.testing.assert_allclose(ref_vals, tst_vals, tol, tol)

    def python_attr(self, attr):
        ker = self.kernel
        ref = "_" + attr
        ref_vals = getattr(ker, ref)(self.xs)
        tst_vals = getattr(ker, attr)(self.xs)
        np.testing.assert_allclose(ref_vals, tst_vals, tol, tol)

    def test_pdf(self):
        self.attr('pdf')
        self.python_attr('pdf')

    def test_cdf(self):
        self.attr('cdf')
        self.python_attr('pdf')

    def test_pm1(self):
        self.python_attr('pm1')

    def test_pm2(self):
        self.python_attr('pm2')
