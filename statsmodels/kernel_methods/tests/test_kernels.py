import pytest
from .. import kernels1d, kernelsnd, kernels_utils
from scipy import stats, integrate
import numpy as np
import numpy.testing as npt
from . import kde_datasets
from ..fast_linbin import fast_linbin
from ..kde_utils import Grid


@pytest.fixture(params=kde_datasets.kernels_1d)
def kernel1d(request):
    return request.param


@pytest.fixture(params=kde_datasets.kernels_nd)
def kernelnd(request):
    return request.param


@pytest.fixture(params=kde_datasets.kernels_nc)
def kernelnc(request):
    return request.param


class RefKernel1D(kernels1d.Kernel1D):
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


class RefKernelnD(kernelsnd.KernelnD):
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
    def setup_class(cls, lower=-np.inf, test_width=3):
        cls.lower = float(lower)
        cls.hard_points = ()
        cls.quad_args = dict(limit=100)
        cls.xs = np.r_[-test_width:test_width:17j]
        bw = 0.2
        R = 10
        N = 2**16
        dx = R / (bw * N)
        cls.dx = dx
        cls.N = N
        cls.small = np.array([-5, -1, -0.5, 0, 0.5, 1, 5])

    def test_unity(self, kernel1d):
        ker = kernel1d.cls()
        total = integrate.quad(ker.pdf, -np.inf, np.inf)[0]
        npt.assert_allclose(total, 1, rtol=tol * kernel1d.precision_factor)

    def test_mean(self, kernel1d):
        ker = kernel1d.cls()

        def f(x):
            return x * ker.pdf(x)

        total = integrate.quad(f, -np.inf, np.inf)[0]
        npt.assert_allclose(total, 0, atol=tol * kernel1d.precision_factor)

    def test_variance(self, kernel1d):
        ker = kernel1d.cls()

        def f(x):
            return x * x * ker.pdf(x)

        total = integrate.quad(f, -np.inf, np.inf)[0]
        acc = tol * kernel1d.precision_factor
        npt.assert_allclose(total, kernel1d.var, rtol=acc, atol=acc)

    def test_cdf(self, kernel1d):
        ker = kernel1d.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.cdf(self.xs)
        val = ker.cdf(self.xs)
        acc = kernel1d.precision_factor * tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)
        tot = ker.cdf(np.inf)
        npt.assert_allclose(tot, 1, rtol=acc)
        short1 = ker.cdf(self.small)
        short2 = [float(ker.cdf(x)) for x in self.small]
        npt.assert_allclose(short1, short2, rtol=acc, atol=acc)

    def test_pm1(self, kernel1d):
        ker = kernel1d.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm1(self.xs)
        val = ker.pm1(self.xs)
        acc = kernel1d.precision_factor * tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)
        tot = ker.pm1(np.inf)
        npt.assert_allclose(tot, 0, atol=acc)
        short1 = ker.pm1(self.small)
        short2 = [float(ker.pm1(x)) for x in self.small]
        npt.assert_allclose(short1, short2, rtol=acc, atol=acc)

    def test_pm2(self, kernel1d):
        ker = kernel1d.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm2(self.xs)
        val = ker.pm2(self.xs)
        acc = kernel1d.precision_factor * tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)
        tot = ker.pm2(np.inf)
        npt.assert_allclose(tot, kernel1d.var, rtol=acc, atol=acc)
        short1 = ker.pm2(self.small)
        short2 = [float(ker.pm2(x)) for x in self.small]
        npt.assert_allclose(short1, short2, rtol=acc, atol=acc)

    def test_rfft(self, kernel1d):
        ker = kernel1d.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.rfft(self.N, self.dx)
        val = ker.rfft(self.N, self.dx)
        acc = kernel1d.precision_factor * tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_rfft_xfx(self, kernel1d):
        ker = kernel1d.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.rfft_xfx(self.N, self.dx)
        val = ker.rfft_xfx(self.N, self.dx)
        acc = kernel1d.precision_factor * tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_dct(self, kernel1d):
        ker = kernel1d.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.dct(self.N, self.dx)
        val = ker.dct(self.N, self.dx)
        acc = kernel1d.precision_factor * tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_convolution(self, kernel1d):
        ker = kernel1d.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.convolution(self.xs)
        val = ker.convolution(self.xs)
        acc = kernel1d.precision_factor * tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_rfftfreq_bad(self):
        with pytest.raises(ValueError):
            kernels_utils.rfftfreq(1.2)


class TestGaussian1d(object):
    @classmethod
    def setup_class(cls, lower=-np.inf):
        cls.kernel = kernels1d.Gaussian1D()
        test_width = cls.kernel.cut
        cls.norm_ref = stats.norm(loc=0, scale=1)
        cls.xs = np.r_[-test_width / 2:test_width / 2:17j]

    @pytest.mark.parametrize("attr", ['pdf', 'cdf'])
    def test_attr(self, attr):
        n_ref = self.norm_ref
        n_tst = self.kernel
        ref_vals = getattr(n_ref, attr)(self.xs)
        tst_vals = getattr(n_tst, attr)(self.xs)
        npt.assert_allclose(ref_vals, tst_vals, rtol=tol, atol=tol)

    @pytest.mark.parametrize("attr", ['pdf', 'cdf', 'pm1', 'pm2'])
    def test_python_attr(self, attr):
        ker = self.kernel
        ref = "_" + attr
        ref_vals = getattr(ker, ref)(self.xs)
        tst_vals = getattr(ker, attr)(self.xs)
        npt.assert_allclose(ref_vals, tst_vals, rtol=tol, atol=tol)


class TestKernelsnd(object):
    @classmethod
    def setup_class(cls):
        dist = stats.norm(0, 1)
        cls.ds = np.c_[dist.rvs(200), dist.rvs(200), dist.rvs(200)]
        bw = 0.2
        R = 10

        N = 2**8
        dx = R / (bw * N)
        cls.dx2 = (dx, dx)
        cls.N2 = (N, N)

        N = 2**6
        dx = R / (bw * N)
        cls.dx3 = (dx, dx, dx)
        cls.N3 = (N, N, N)

        cut = 5
        cls.grid2d = Grid.fromSparse(np.ogrid[-cut:cut:512j, -cut:cut:512j])
        cls.grid3d = Grid.fromSparse(np.ogrid[-cut:cut:128j, -cut:cut:128j,
                                              -cut:cut:128j])

    def test_unity2d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(2)
        vals = ker(self.grid2d.full())
        total = self.grid2d.integrate(vals)
        npt.assert_allclose(total, 1, rtol=nd_tol * kernelnd.precision_factor)

    def test_unity3d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(3)
        vals = ker(self.grid3d.full())
        total = self.grid3d.integrate(vals)
        npt.assert_allclose(total, 1, rtol=nd_tol * kernelnd.precision_factor)

    def test_cdf2d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(2)
        ref_ker = RefKernelnD(ker)
        acc = tol * kernelnd.precision_factor
        npt.assert_allclose(ker.cdf([-np.inf, -np.inf]), 0, rtol=acc, atol=acc)
        npt.assert_allclose(ker.cdf([np.inf, np.inf]), 1, rtol=acc, atol=acc)
        npt.assert_allclose(ker.cdf([0, 0]),
                            ref_ker.cdf([0, 0]),
                            rtol=acc,
                            atol=acc)

    def test_cdf3d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(3)
        acc = tol * kernelnd.precision_factor
        npt.assert_allclose(ker.cdf([-np.inf, -np.inf, -np.inf]),
                            0,
                            rtol=acc,
                            atol=acc)
        npt.assert_allclose(ker.cdf([np.inf, np.inf, np.inf]),
                            1,
                            rtol=acc,
                            atol=acc)

    def test_rfft2d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(2)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.rfft(self.N2, self.dx2)
        val = ker.rfft(self.N2, self.dx2)
        acc = kernelnd.precision_factor * nd_tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_rfft3d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(3)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.rfft(self.N3, self.dx3)
        val = ker.rfft(self.N3, self.dx3)
        acc = 100 * kernelnd.precision_factor * nd_tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_dct2d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(2)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.dct(self.N2, self.dx2[:2])
        val = ker.dct(self.N2, self.dx2[:2])
        acc = kernelnd.precision_factor * nd_tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)

    def test_dct3d(self, kernelnd):
        ker = kernelnd.cls().for_ndim(3)
        ref_ker = RefKernelnD(ker)
        ref = ref_ker.dct(self.N3, self.dx3)
        val = ker.dct(self.N3, self.dx3)
        acc = 100 * kernelnd.precision_factor * nd_tol
        npt.assert_allclose(val, ref, rtol=acc, atol=acc)


class TestKernelnc(object):
    @classmethod
    def setup_class(cls):
        dist = stats.poisson(10)
        cls.ds = dist.rvs(10)
        cls.xs = np.arange(0, cls.ds.max() + 1)[:, None]
        cls.num_levels = cls.ds.max() + 1
        cls.mesh, cls.bins = fast_linbin(cls.ds, [0, cls.num_levels],
                                         cls.num_levels,
                                         bin_type='D')

    def test_pdf(self, kernelnc):
        k = kernelnc.cls()
        npt.assert_equal(k.ndim, 1)
        dst = k.pdf(self.xs, self.ds, 0.2, self.num_levels)
        assert dst.sum() <= self.ds.shape[0] + tol * kernelnc.precision_factor

    def test_from_binned(self, kernelnc):
        k = kernelnc.cls()
        dst = k.from_binned(self.mesh, self.bins, 0.2)
        assert dst.sum() <= self.ds.shape[0] + tol * kernelnc.precision_factor
