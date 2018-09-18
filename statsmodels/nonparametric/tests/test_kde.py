import os
import numpy.testing as npt
import numpy as np
import pytest
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import statsmodels.sandbox.nonparametric.kernels as kernels
from scipy import stats

# get results from Stata

curdir = os.path.dirname(os.path.abspath(__file__))
rfname = os.path.join(curdir,'results','results_kde.csv')
#print rfname
KDEResults = np.genfromtxt(open(rfname, 'rb'), delimiter=",", names=True)

rfname = os.path.join(curdir,'results','results_kde_univ_weights.csv')
KDEWResults = np.genfromtxt(open(rfname, 'rb'), delimiter=",", names=True)

# get results from R
curdir = os.path.dirname(os.path.abspath(__file__))
rfname = os.path.join(curdir,'results','results_kcde.csv')
#print rfname
KCDEResults = np.genfromtxt(open(rfname, 'rb'), delimiter=",", names=True)


# setup test data

np.random.seed(12345)
Xi = mixture_rvs([.25,.75], size=200, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))

class TestKDEExceptions(object):

    @classmethod
    def setup_class(cls):
        cls.kde = KDE(Xi)
        cls.weights_200 = np.linspace(1, 100, 200)
        cls.weights_100 = np.linspace(1, 100, 100)

    def test_check_is_fit_exception(self):
        with pytest.raises(ValueError):
            self.kde.evaluate(0)

    def test_non_weighted_fft_exception(self):
        with pytest.raises(NotImplementedError):
            self.kde.fit(kernel="gau", gridsize=50, weights=self.weights_200,
                         fft=True, bw="silverman")

    def test_wrong_weight_length_exception(self):
        with pytest.raises(ValueError):
            self.kde.fit(kernel="gau", gridsize=50, weights=self.weights_100,
                         fft=False, bw="silverman")

    def test_non_gaussian_fft_exception(self):
        with pytest.raises(NotImplementedError):
            self.kde.fit(kernel="epa", gridsize=50, fft=True, bw="silverman")

class CheckKDE(object):

    decimal_density = 7

    def test_density(self):
        npt.assert_almost_equal(self.res1.density, self.res_density,
                self.decimal_density)

    def test_evaluate(self):
        # disable test
        # fails for Epan, Triangular and Biweight, only Gaussian is correct
        # added it as test method to TestKDEGauss below
        # inDomain is not vectorized
        #kde_vals = self.res1.evaluate(self.res1.support)
        kde_vals = [np.squeeze(self.res1.evaluate(xi)) for xi in self.res1.support]
        kde_vals = np.squeeze(kde_vals)  #kde_vals is a "column_list"
        mask_valid = np.isfinite(kde_vals)
        # TODO: nans at the boundaries
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(kde_vals, self.res_density,
                                self.decimal_density)


class TestKDEGauss(CheckKDE):
    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="gau", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["gau_d"]

    def test_evaluate(self):
        #kde_vals = self.res1.evaluate(self.res1.support)
        kde_vals = [self.res1.evaluate(xi) for xi in self.res1.support]
        kde_vals = np.squeeze(kde_vals)  #kde_vals is a "column_list"
        mask_valid = np.isfinite(kde_vals)
        # TODO: nans at the boundaries
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(kde_vals, self.res_density,
                                self.decimal_density)

    # The following tests are regression tests
    # Values have been checked to be very close to R 'ks' package (Dec 2013)
    def test_support_gridded(self):
        kde = self.res1
        support = KCDEResults['gau_support']
        npt.assert_allclose(support, kde.support)

    def test_cdf_gridded(self):
        kde = self.res1
        cdf = KCDEResults['gau_cdf']
        npt.assert_allclose(cdf, kde.cdf)

    def test_sf_gridded(self):
        kde = self.res1
        sf = KCDEResults['gau_sf']
        npt.assert_allclose(sf, kde.sf)

    def test_icdf_gridded(self):
        kde = self.res1
        icdf = KCDEResults['gau_icdf']
        npt.assert_allclose(icdf, kde.icdf)


class TestKDEEpanechnikov(CheckKDE):
    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="epa", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["epa2_d"]

class TestKDETriangular(CheckKDE):
    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="tri", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["tri_d"]

class TestKDEBiweight(CheckKDE):
    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="biw", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["biw_d"]

#NOTE: This is a knownfailure due to a definitional difference of Cosine kernel
#class TestKDECosine(CheckKDE):
#    @classmethod
#    def setup_class(cls):
#        res1 = KDE(Xi)
#        res1.fit(kernel="cos", fft=False, bw="silverman")
#        cls.res1 = res1
#        cls.res_density = KDEResults["cos_d"]

#weighted estimates taken from matlab so we can allow len(weights) != gridsize
class TestKdeWeights(CheckKDE):

    @classmethod
    def setup_class(cls):
        res1 = KDE(Xi)
        weights = np.linspace(1,100,200)
        res1.fit(kernel="gau", gridsize=50, weights=weights, fft=False,
                    bw="silverman")
        cls.res1 = res1
        rfname = os.path.join(curdir,'results','results_kde_weights.csv')
        cls.res_density = np.genfromtxt(open(rfname, 'rb'), skip_header=1)

    def test_evaluate(self):
        #kde_vals = self.res1.evaluate(self.res1.support)
        kde_vals = [self.res1.evaluate(xi) for xi in self.res1.support]
        kde_vals = np.squeeze(kde_vals)  #kde_vals is a "column_list"
        mask_valid = np.isfinite(kde_vals)
        # TODO: nans at the boundaries
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(kde_vals, self.res_density,
                                self.decimal_density)


class TestKDEGaussFFT(CheckKDE):
    @classmethod
    def setup_class(cls):
        cls.decimal_density = 2 # low accuracy because binning is different
        res1 = KDE(Xi)
        res1.fit(kernel="gau", fft=True, bw="silverman")
        cls.res1 = res1
        rfname2 = os.path.join(curdir,'results','results_kde_fft.csv')
        cls.res_density = np.genfromtxt(open(rfname2, 'rb'))

class CheckKDEWeights(object):

    @classmethod
    def setup_class(cls):
        cls.x = x = KDEWResults['x']
        weights = KDEWResults['weights']
        res1 = KDE(x)
        # default kernel was scott when reference values computed
        res1.fit(kernel=cls.kernel_name, weights=weights, fft=False, bw="scott")
        cls.res1 = res1
        cls.res_density = KDEWResults[cls.res_kernel_name]

    decimal_density = 7

    def t_est_density(self):
        npt.assert_almost_equal(self.res1.density, self.res_density,
                self.decimal_density)

    def test_evaluate(self):
        if self.kernel_name == 'cos':
            pytest.skip("Cosine kernel fails against Stata")
        kde_vals = [self.res1.evaluate(xi) for xi in self.x]
        kde_vals = np.squeeze(kde_vals)  #kde_vals is a "column_list"
        npt.assert_almost_equal(kde_vals, self.res_density,
                                self.decimal_density)

    def test_compare(self):
        xx = self.res1.support
        kde_vals = [np.squeeze(self.res1.evaluate(xi)) for xi in xx]
        kde_vals = np.squeeze(kde_vals)  #kde_vals is a "column_list"
        mask_valid = np.isfinite(kde_vals)
        # TODO: nans at the boundaries
        kde_vals[~mask_valid] = 0
        npt.assert_almost_equal(self.res1.density, kde_vals,
                                self.decimal_density)

        # regression test, not compared to another package
        nobs = len(self.res1.endog)
        kern = self.res1.kernel
        v = kern.density_var(kde_vals, nobs)
        v_direct = kde_vals * kern.L2Norm / kern.h / nobs
        npt.assert_allclose(v, v_direct, rtol=1e-10)

        ci = kern.density_confint(kde_vals, nobs)
        crit = 1.9599639845400545 #stats.norm.isf(0.05 / 2)
        hw = kde_vals - ci[:, 0]
        npt.assert_allclose(hw, crit * np.sqrt(v), rtol=1e-10)
        hw = ci[:, 1] - kde_vals
        npt.assert_allclose(hw, crit * np.sqrt(v), rtol=1e-10)

    def test_kernel_constants(self):
        kern = self.res1.kernel

        nc = kern.norm_const
        # trigger numerical integration
        kern._norm_const = None
        nc2 = kern.norm_const
        npt.assert_allclose(nc, nc2, rtol=1e-10)

        l2n = kern.L2Norm
        # trigger numerical integration
        kern._L2Norm = None
        l2n2 = kern.L2Norm
        npt.assert_allclose(l2n, l2n2, rtol=1e-10)

        v = kern.kernel_var
        # trigger numerical integration
        kern._kernel_var = None
        v2 = kern.kernel_var
        npt.assert_allclose(v, v2, rtol=1e-10)


class TestKDEWGauss(CheckKDEWeights):

    kernel_name = "gau"
    res_kernel_name = "x_gau_wd"


class TestKDEWEpa(CheckKDEWeights):

    kernel_name = "epa"
    res_kernel_name = "x_epan2_wd"


class TestKDEWTri(CheckKDEWeights):

    kernel_name = "tri"
    res_kernel_name = "x_" + kernel_name + "_wd"


class TestKDEWBiw(CheckKDEWeights):

    kernel_name = "biw"
    res_kernel_name = "x_bi_wd"


class TestKDEWCos(CheckKDEWeights):

    kernel_name = "cos"
    res_kernel_name = "x_cos_wd"


class TestKDEWCos2(CheckKDEWeights):

    kernel_name = "cos2"
    res_kernel_name = "x_cos_wd"


class T_estKDEWRect(CheckKDEWeights):
    #TODO in docstring but not in kernel_switch
    kernel_name = "rect"
    res_kernel_name = "x_rec_wd"


class T_estKDEWPar(CheckKDEWeights):
    # TODO in docstring but not implemented in kernels
    kernel_name = "par"
    res_kernel_name = "x_par_wd"


class TestKdeRefit():
    np.random.seed(12345)
    data1 = np.random.randn(100) * 100
    pdf = KDE(data1)
    pdf.fit()

    data2 = np.random.randn(100) * 100
    pdf2 = KDE(data2)
    pdf2.fit()

    for attr in ['icdf', 'cdf', 'sf']:
        npt.assert_(not np.allclose(getattr(pdf, attr)[:10],
                                    getattr(pdf2, attr)[:10]))


class TestNormConstant():

    def test_norm_constant_calculation(self):
        custom_gauss = kernels.CustomKernel(lambda x: np.exp(-x**2/2.0))
        gauss_true_const = 0.3989422804014327
        npt.assert_almost_equal(gauss_true_const, custom_gauss.norm_const)
