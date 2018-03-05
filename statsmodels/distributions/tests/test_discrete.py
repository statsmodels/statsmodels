import os
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson, nbinom
from numpy.testing import (assert_, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose,
                           assert_array_less)


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

class TestZIGneralizedPoisson(object):
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

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
