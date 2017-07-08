import os
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
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


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            exit=False)
