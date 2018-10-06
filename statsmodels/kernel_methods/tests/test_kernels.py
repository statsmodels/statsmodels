from __future__ import division, absolute_import, print_function

from .. import kernels

from scipy import stats, integrate
import numpy as np
from . import kde_utils
from nose.tools import raises
from ...tools.testing import assert_allclose, assert_equal
from ..fast_linbin import fast_linbin
from ..kde_utils import Grid

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


class RefKernelnD(kernels.KernelnD):
    """
    Reference kernel: force use of explicit integration
    """
    def __init__(self, kernel):
        self._ndim = kernel.ndim
        self.lower = kernel.lower
        self.upper = kernel.upper
        self.cut = kernel.cut
        self.real_kernel = kernel

    def pdf(self, z, out=None):
        return self.real_kernel.pdf(z, out)

tol = 1e-8
nd_tol = 1e-5

class TestKernels1D(object):
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
        assert_allclose(total, 1, rtol=tol*kernel.precision_factor)

    def mean(self, kernel):
        ker = kernel.cls()

        def f(x):
            return x*ker.pdf(x)
        total = integrate.quad(f, -np.inf, np.inf)[0]
        assert_allclose(total, 0, atol=tol*kernel.precision_factor)

    def variance(self, kernel):
        ker = kernel.cls()

        def f(x):
            return x*x*ker.pdf(x)
        total = integrate.quad(f, -np.inf, np.inf)[0]
        acc = tol*kernel.precision_factor
        assert_allclose(total, kernel.var, rtol=acc, atol=acc)

    def cdf(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.cdf(self.xs)
        val = ker.cdf(self.xs)
        acc = kernel.precision_factor * tol
        assert_allclose(val, ref, rtol=acc, atol=acc)
        tot = ker.cdf(np.inf)
        assert_allclose(tot, 1, rtol=acc)
        short1 = ker.cdf(self.small)
        short2 = [float(ker.cdf(x)) for x in self.small]
        assert_allclose(short1, short2, rtol=acc, atol=acc)

    def pm1(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm1(self.xs)
        val = ker.pm1(self.xs)
        acc = kernel.precision_factor * tol
        assert_allclose(val, ref, rtol=acc, atol=acc)
        tot = ker.pm1(np.inf)
        assert_allclose(tot, 0, atol=acc)
        short1 = ker.pm1(self.small)
        short2 = [float(ker.pm1(x)) for x in self.small]
        assert_allclose(short1, short2, rtol=acc, atol=acc)

    def pm2(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm2(self.xs)
        val = ker.pm2(self.xs)
        acc = kernel.precision_factor * tol
        assert_allclose(val, ref, rtol=acc, atol=acc)
        tot = ker.pm2(np.inf)
        assert_allclose(tot, kernel.var, rtol=acc, atol=acc)
        short1 = ker.pm2(self.small)
        short2 = [float(ker.pm2(x)) for x in self.small]
        assert_allclose(short1, short2, rtol=acc, atol=acc)

    def rfft(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.rfft(self.N, self.dx)
        val = ker.rfft(self.N, self.dx)
        acc = kernel.precision_factor * tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def rfft_xfx(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.rfft_xfx(self.N, self.dx)
        val = ker.rfft_xfx(self.N, self.dx)
        acc = kernel.precision_factor * tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def dct(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.dct(self.N, self.dx)
        val = ker.dct(self.N, self.dx)
        acc = kernel.precision_factor * tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def convolution(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.convolution(self.xs)
        val = ker.convolution(self.xs)
        acc = kernel.precision_factor * tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_unity(self):
        for kernel in kde_utils.kernels1d:
            yield self.unity, kernel

    def test_mean(self):
        for kernel in kde_utils.kernels1d:
            yield self.unity, kernel

    def test_variance(self):
        for kernel in kde_utils.kernels1d:
            yield self.variance, kernel

    def test_cdf(self):
        for kernel in kde_utils.kernels1d:
            yield self.cdf, kernel

    def test_convolution(self):
        for kernel in kde_utils.kernels1d:
            yield self.convolution, kernel

    def test_pm1(self):
        for kernel in kde_utils.kernels1d:
            yield self.pm1, kernel

    def test_pm2(self):
        for kernel in kde_utils.kernels1d:
            yield self.pm2, kernel

    def test_dct(self):
        for kernel in kde_utils.kernels1d:
            yield self.dct, kernel

    def test_rfft(self):
        for kernel in kde_utils.kernels1d:
            yield self.rfft, kernel

    def test_rfft_xfx(self):
        for kernel in kde_utils.kernels1d:
            yield self.rfft_xfx, kernel

    @raises(ValueError)
    def test_rfftfreq_bad(self):
        kernels.rfftfreq(1.2)

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
        assert_allclose(ref_vals, tst_vals, rtol=tol, atol=tol)

    def python_attr(self, attr):
        ker = self.kernel
        ref = "_" + attr
        ref_vals = getattr(ker, ref)(self.xs)
        tst_vals = getattr(ker, attr)(self.xs)
        assert_allclose(ref_vals, tst_vals, rtol=tol, atol=tol)

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

class TestKernelsnd(object):
    @classmethod
    def setUpClass(cls):
        dist = stats.norm(0, 1)
        cls.ds = np.c_[dist.rvs(200),
                       dist.rvs(200),
                       dist.rvs(200)]
        bw = 0.2
        R = 10

        N = 2**8
        dx = R/(bw*N)
        cls.dx2 = (dx, dx)
        cls.N2 = (N, N)

        N = 2**6
        dx = R/(bw*N)
        cls.dx3 = (dx, dx, dx)
        cls.N3 = (N, N, N)

        cut = 5
        cls.grid2d = Grid.fromSparse(np.ogrid[-cut:cut:512j, -cut:cut:512j])
        cls.grid3d = Grid.fromSparse(np.ogrid[-cut:cut:128j,
                                              -cut:cut:128j,
                                              -cut:cut:128j])

    def unity2d(self, kernel):
        ker = kernel.cls().for_ndim(2)
        vals = ker(self.grid2d.full())
        total = self.grid2d.integrate(vals)
        assert_allclose(total, 1, rtol=nd_tol*kernel.precision_factor)

    def unity3d(self, kernel):
        ker = kernel.cls().for_ndim(3)
        vals = ker(self.grid3d.full())
        total = self.grid3d.integrate(vals)
        assert_allclose(total, 1, rtol=nd_tol*kernel.precision_factor)

    def cdf2d(self, kernel):
        ker = kernel.cls().for_ndim(2)
        ref_ker = RefKernelnD(ker)
        acc = tol*kernel.precision_factor
        assert_allclose(ker.cdf([-np.inf, -np.inf]), 0, rtol=acc, atol=acc)
        assert_allclose(ker.cdf([np.inf, np.inf]), 1, rtol=acc, atol=acc)
        assert_allclose(ker.cdf([0, 0]), ref_ker.cdf([0, 0]), rtol=acc, atol=acc)

    def cdf3d(self, kernel):
        ker = kernel.cls().for_ndim(3)
        acc = tol*kernel.precision_factor
        assert_allclose(ker.cdf([-np.inf, -np.inf, -np.inf]), 0, rtol=acc, atol=acc)
        assert_allclose(ker.cdf([np.inf, np.inf, np.inf]), 1, rtol=acc, atol=acc)

    def rfft2d(self, kernel):
        ker = kernel.cls().for_ndim(2)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.rfft(self.N2, self.dx2)
        val = ker.rfft(self.N2, self.dx2)
        acc = kernel.precision_factor * nd_tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def rfft3d(self, kernel):
        ker = kernel.cls().for_ndim(3)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.rfft(self.N3, self.dx3)
        val = ker.rfft(self.N3, self.dx3)
        acc = 100*kernel.precision_factor * nd_tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def dct2d(self, kernel):
        ker = kernel.cls().for_ndim(2)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.dct(self.N2, self.dx2[:2])
        val = ker.dct(self.N2, self.dx2[:2])
        acc = kernel.precision_factor * nd_tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def dct3d(self, kernel):
        ker = kernel.cls().for_ndim(3)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.dct(self.N3, self.dx3)
        val = ker.dct(self.N3, self.dx3)
        acc = 100*kernel.precision_factor * nd_tol
        assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_unity(self):
        for k in kde_utils.kernelsnd:
            yield self.unity2d, k
            yield self.unity3d, k

    def test_cdf(self):
        for k in kde_utils.kernelsnd:
            yield self.cdf2d, k
            yield self.cdf3d, k

    def test_rfft(self):
        for k in kde_utils.kernelsnd:
            yield self.rfft2d, k
            yield self.rfft3d, k

    def test_dct(self):
        for k in kde_utils.kernelsnd:
            yield self.dct2d, k
            yield self.dct3d, k

class TestKernelnc(object):
    @classmethod
    def setUpClass(cls):
        dist = stats.poisson(10)
        cls.ds = dist.rvs(10)
        cls.xs = np.arange(0, cls.ds.max()+1)[:, None]
        cls.num_levels = cls.ds.max()+1
        cls.mesh, cls.bins = fast_linbin(cls.ds, [0, cls.num_levels], cls.num_levels, bin_type='d')

    def pdf(self, kernel):
        k = kernel.cls()
        assert_equal(k.ndim, 1)
        dst = k.pdf(self.xs, self.ds, 0.2, self.num_levels)
        assert dst.sum() <= self.ds.shape[0] + tol*kernel.precision_factor

    def from_binned(self, kernel):
        k = kernel.cls()
        dst = k.from_binned(self.mesh, self.bins, 0.2)
        assert dst.sum() <= self.ds.shape[0] + tol*kernel.precision_factor

    def test_pdf(self):
        for k in kde_utils.kernelsnc:
            yield self.pdf, k

    def test_from_binned(self):
        for k in kde_utils.kernelsnc:
            yield self.from_binned, k
