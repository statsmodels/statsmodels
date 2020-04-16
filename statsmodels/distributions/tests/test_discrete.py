import statsmodels.api as sm
from scipy.stats import poisson, nbinom
from numpy.testing import assert_allclose


class TestGenpoisson_p(object):
    """
    Test Generalized Poisson Destribution
    """

    def test_pmf_p1(self):
        poisson_pmf = poisson.pmf(1, 1)
        genpoisson_pmf = sm.distributions.genpoisson_p.pmf(1, 1, 0, 1)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_pmf_p2(self):
        poisson_pmf = poisson.pmf(2, 2)
        genpoisson_pmf = sm.distributions.genpoisson_p.pmf(2, 2, 0, 2)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_pmf_p5(self):
        poisson_pmf = poisson.pmf(10, 2)
        genpoisson_pmf_5 = sm.distributions.genpoisson_p.pmf(10, 2, 1e-25, 5)
        assert_allclose(poisson_pmf, genpoisson_pmf_5, rtol=1e-12)

    def test_logpmf_p1(self):
        poisson_pmf = poisson.logpmf(5, 2)
        genpoisson_pmf = sm.distributions.genpoisson_p.logpmf(5, 2, 0, 1)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_logpmf_p2(self):
        poisson_pmf = poisson.logpmf(6, 1)
        genpoisson_pmf = sm.distributions.genpoisson_p.logpmf(6, 1, 0, 2)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

class TestZIPoisson(object):
    """
    """
    def test_pmf_zero(self):
        poisson_pmf = poisson.pmf(3, 2)
        zipoisson_pmf = sm.distributions.zipoisson.pmf(3, 2, 0)
        assert_allclose(poisson_pmf, zipoisson_pmf, rtol=1e-12)

    def test_logpmf_zero(self):
        poisson_logpmf = poisson.logpmf(5, 1)
        zipoisson_logpmf = sm.distributions.zipoisson.logpmf(5, 1, 0)
        assert_allclose(poisson_logpmf, zipoisson_logpmf, rtol=1e-12)

    def test_pmf(self):
        poisson_pmf = poisson.pmf(2, 2)
        zipoisson_pmf = sm.distributions.zipoisson.pmf(2, 2, 0.1)
        assert_allclose(poisson_pmf, zipoisson_pmf, rtol=5e-2, atol=5e-2)

    def test_logpmf(self):
        poisson_logpmf = poisson.logpmf(7, 3)
        zipoisson_logpmf = sm.distributions.zipoisson.logpmf(7, 3, 0.1)
        assert_allclose(poisson_logpmf, zipoisson_logpmf, rtol=5e-2, atol=5e-2)

    def test_cdf_zero(self):
        poisson_cdf = poisson.cdf(3, 2)
        zipoisson_cdf = sm.distributions.zipoisson.cdf(3, 2, 0)
        assert_allclose(poisson_cdf, zipoisson_cdf, rtol=1e-12)

    def test_ppf_zero(self):
        poisson_ppf = poisson.ppf(5, 1)
        zipoisson_ppf = sm.distributions.zipoisson.ppf(5, 1, 0)
        assert_allclose(poisson_ppf, zipoisson_ppf, rtol=1e-12)

class TestZIGeneralizedPoisson(object):
    def test_pmf_zero(self):
        gp_pmf = sm.distributions.genpoisson_p.pmf(3, 2, 1, 1)
        zigp_pmf = sm.distributions.zigenpoisson.pmf(3, 2, 1, 1, 0)
        assert_allclose(gp_pmf, zigp_pmf, rtol=1e-12)

    def test_logpmf_zero(self):
        gp_logpmf = sm.distributions.genpoisson_p.logpmf(7, 3, 1, 1)
        zigp_logpmf = sm.distributions.zigenpoisson.logpmf(7, 3, 1, 1, 0)
        assert_allclose(gp_logpmf, zigp_logpmf, rtol=1e-12)

    def test_pmf(self):
        gp_pmf = sm.distributions.genpoisson_p.pmf(3, 2, 2, 2)
        zigp_pmf = sm.distributions.zigenpoisson.pmf(3, 2, 2, 2, 0.1)
        assert_allclose(gp_pmf, zigp_pmf, rtol=5e-2, atol=5e-2)

    def test_logpmf(self):
        gp_logpmf = sm.distributions.genpoisson_p.logpmf(2, 3, 0, 2)
        zigp_logpmf = sm.distributions.zigenpoisson.logpmf(2, 3, 0, 2, 0.1)
        assert_allclose(gp_logpmf, zigp_logpmf, rtol=5e-2, atol=5e-2)


class TestZiNBP(object):
    """
    Test Truncated Poisson distribution
    """
    def test_pmf_p2(self):
        n, p = sm.distributions.zinegbin.convert_params(30, 0.1, 2)
        nb_pmf = nbinom.pmf(100, n, p)
        tnb_pmf = sm.distributions.zinegbin.pmf(100, 30, 0.1, 2, 0.01)
        assert_allclose(nb_pmf, tnb_pmf, rtol=1e-5, atol=1e-5)

    def test_logpmf_p2(self):
        n, p = sm.distributions.zinegbin.convert_params(10, 1, 2)
        nb_logpmf = nbinom.logpmf(200, n, p)
        tnb_logpmf = sm.distributions.zinegbin.logpmf(200, 10, 1, 2, 0.01)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=1e-2, atol=1e-2)

    def test_cdf_p2(self):
        n, p = sm.distributions.zinegbin.convert_params(30, 0.1, 2)
        nbinom_cdf = nbinom.cdf(10, n, p)
        zinbinom_cdf = sm.distributions.zinegbin.cdf(10, 30, 0.1, 2, 0)
        print(nbinom_cdf, zinbinom_cdf)
        assert_allclose(nbinom_cdf, zinbinom_cdf, rtol=1e-12, atol=1e-12)

    def test_ppf_p2(self):
        n, p = sm.distributions.zinegbin.convert_params(100, 1, 2)
        nbinom_ppf = nbinom.ppf(0.27, n, p)
        zinbinom_ppf = sm.distributions.zinegbin.ppf(0.27, 100, 1, 2, 0)
        print(nbinom_ppf, zinbinom_ppf)
        assert_allclose(nbinom_ppf, zinbinom_ppf, rtol=1e-12, atol=1e-12)

    def test_pmf(self):
        n, p = sm.distributions.zinegbin.convert_params(1, 0.9, 1)
        nb_logpmf = nbinom.pmf(2, n, p)
        tnb_pmf = sm.distributions.zinegbin.pmf(2, 1, 0.9, 2, 0.5)
        assert_allclose(nb_logpmf, tnb_pmf * 2, rtol=1e-7)

    def test_logpmf(self):
        n, p = sm.distributions.zinegbin.convert_params(5, 1, 1)
        nb_logpmf = nbinom.logpmf(2, n, p)
        tnb_logpmf = sm.distributions.zinegbin.logpmf(2, 5, 1, 1, 0.005)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=1e-2, atol=1e-2)

    def test_cdf(self):
        n, p = sm.distributions.zinegbin.convert_params(1, 0.9, 1)
        nbinom_cdf = nbinom.cdf(2, n, p)
        zinbinom_cdf = sm.distributions.zinegbin.cdf(2, 1, 0.9, 2, 0)
        assert_allclose(nbinom_cdf, zinbinom_cdf, rtol=1e-12, atol=1e-12)

    def test_ppf(self):
        n, p = sm.distributions.zinegbin.convert_params(5, 1, 1)
        nbinom_ppf = nbinom.ppf(0.71, n, p)
        zinbinom_ppf = sm.distributions.zinegbin.ppf(0.71, 5, 1, 1, 0)
        assert_allclose(nbinom_ppf, zinbinom_ppf, rtol=1e-12, atol=1e-12)

    def test_convert(self):
        n, p = sm.distributions.zinegbin.convert_params(25, 0.85, 2)
        n_true, p_true = 1.1764705882352942, 0.04494382022471911
        assert_allclose(n, n_true, rtol=1e-12, atol=1e-12)
        assert_allclose(p, p_true, rtol=1e-12, atol=1e-12)

        n, p = sm.distributions.zinegbin.convert_params(7, 0.17, 1)
        n_true, p_true = 41.17647058823529, 0.8547008547008547
        assert_allclose(n, n_true, rtol=1e-12, atol=1e-12)
        assert_allclose(p, p_true, rtol=1e-12, atol=1e-12)
