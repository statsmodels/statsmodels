"""Copula tests.

The reference results are coming from the R package Copula. The following
script is used:

    library(copula)
    sessionInfo()

    u <- matrix(c(0.33706249, 0.62232507,
                  0.2001457 , 0.77166391,
                  0.98534253, 0.72755898,
                  0.05943888, 0.0962475 ,
                  0.35496733, 0.44513594,
                  0.6075078 , 0.06241089,
                  0.54027684, 0.40610225,
                  0.99212789, 0.25913165,
                  0.61044613, 0.67585563,
                  0.79584436, 0.23050014),
                nrow=10)

    print("Vector u")
    print(u)

    gaussian <- normalCopula(0.8, dim = 2)
    student <- tCopula(0.8, dim = 2, df = 2)
    frank <- frankCopula(dim = 2, param = 3)
    clayton <- claytonCopula(dim = 2, param = 1.2)
    gumbel <- gumbelCopula(dim = 2, param = 1.5)

    # Compute the density and CDF
    pdf <- dCopula(u, gaussian)
    cdf <- pCopula(u, gaussian)
    print("Gaussian")
    print(pdf)
    print(cdf)

    pdf <- dCopula(u, student)
    cdf <- pCopula(u, student)
    print("Student")
    print(pdf)
    print(cdf)

    pdf <- dCopula(u, frank)
    cdf <- pCopula(u, frank)
    print("Frank")
    print(pdf)
    print(cdf)

    pdf <- dCopula(u, clayton)
    cdf <- pCopula(u, clayton)
    print("Clayton")
    print(pdf)
    print(cdf)

    pdf <- dCopula(u, gumbel)
    cdf <- pCopula(u, gumbel)
    print("Gumbel")
    print(pdf)
    print(cdf)

Which produces the following output:

    R version 4.0.3 (2020-10-10)
    Platform: x86_64-pc-linux-gnu (64-bit)
    Running under: Ubuntu 20.04.1 LTS

    Matrix products: default
    BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.9.0
    LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.9.0

    locale:
     [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C
     [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8
     [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8
     [7] LC_PAPER=en_US.UTF-8       LC_NAME=C
     [9] LC_ADDRESS=C               LC_TELEPHONE=C
    [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C

    attached base packages:
    [1] stats     graphics  grDevices utils     datasets  methods   base

    other attached packages:
    [1] copula_1.0-0

    loaded via a namespace (and not attached):
     [1] compiler_4.0.3      Matrix_1.2-18       ADGofTest_0.3
     [4] pspline_1.0-18      gsl_2.1-6           mvtnorm_1.1-1
     [7] grid_4.0.3          pcaPP_1.9-73        numDeriv_2016.8-1.1
    [10] stats4_4.0.3        lattice_0.20-41     stabledist_0.7-1
    [1] "Vector u"
                [,1]       [,2]
     [1,] 0.33706249 0.60750780
     [2,] 0.62232507 0.06241089
     [3,] 0.20014570 0.54027684
     [4,] 0.77166391 0.40610225
     [5,] 0.98534253 0.99212789
     [6,] 0.72755898 0.25913165
     [7,] 0.05943888 0.61044613
     [8,] 0.09624750 0.67585563
     [9,] 0.35496733 0.79584436
    [10,] 0.44513594 0.23050014
    [1] "Gaussian"
     [1]  1.03308741  0.06507279  0.72896012  0.65389439 16.45012399 0.34813218
     [7]  0.06768115  0.08168840  0.40521741  1.26723470
     [1] 0.31906854 0.06230196 0.19284669 0.39952707 0.98144792 0.25677003
     [7] 0.05932818 0.09605404 0.35211017 0.20885480
    [1] "Student"
     [1]  0.8303065  0.1359839  0.5157746  0.4776421 26.2173959  0.3070661
     [7]  0.1349173  0.1597064  0.3303230  1.0482301
     [1] 0.31140349 0.05942746 0.18548601 0.39143974 0.98347259 0.24894028
     [7] 0.05653947 0.09210693 0.34447385 0.20429882
    [1] "Frank"
     [1] 0.9646599 0.5627195 0.8941964 0.8364614 2.9570945 0.6665601 0.5779906
     [8] 0.5241333 0.7156741 1.1074024
     [1] 0.27467496 0.05492539 0.15995939 0.36750702 0.97782283 0.23412757
     [7] 0.05196265 0.08676979 0.32803721 0.16320730
    [1] "Clayton"
     [1] 1.0119836 0.2072728 0.8148839 0.9481976 2.1419659 0.6828507 0.2040454
     [8] 0.2838497 0.8197787 1.1096360
     [1] 0.28520375 0.06101690 0.17703377 0.36848218 0.97772088 0.24082057
     [7] 0.05811908 0.09343934 0.33012582 0.18738753
    [1] "Gumbel"
     [1]  1.0391696  0.6539579  0.9878446  0.8679504 16.6030932  0.7542073
     [7]  0.6668307  0.6275887  0.7477991  1.1564864
     [1] 0.27194634 0.05484380 0.15668190 0.37098420 0.98176346 0.23422865
     [7] 0.05188260 0.08659615 0.33086960 0.15803914




"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy import stats

from statsmodels.distributions.joint_distribution import (
    JointDistribution
)
from statsmodels.distributions.copula.other_copulas import IndependentCopula
from statsmodels.distributions.copula.elliptical import GaussianCopula, \
    StudentCopula
from statsmodels.distributions.copula.archimedean import ClaytonCopula, \
    FrankCopula, GumbelCopula


class CopulaTests:
    """Generic tests for copula."""
    copula = None
    dim = None
    u = np.array([[0.33706249, 0.6075078],
                  [0.62232507, 0.06241089],
                  [0.2001457, 0.54027684],
                  [0.77166391, 0.40610225],
                  [0.98534253, 0.99212789],
                  [0.72755898, 0.25913165],
                  [0.05943888, 0.61044613],
                  [0.0962475, 0.67585563],
                  [0.35496733, 0.79584436],
                  [0.44513594, 0.23050014]])
    pdf_u = None
    cdf_u = None

    def test_visualization(self):
        sample = self.copula.random(10000)
        assert sample.shape == (10000, 2)
        # h = sns.jointplot(sample[:, 0], sample[:, 1], kind='hex')
        # h.set_axis_labels('X1', 'X2', fontsize=16)

    def test_uniform_marginals(self):
        sample = self.copula.random(10000)
        assert_array_almost_equal(
            np.mean(sample, axis=0), np.repeat(0.5, self.dim), decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, self.dim),
            decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, self.dim),
            decimal=2
        )

    def test_pdf(self):
        pdf_u_test = self.copula.pdf(self.u)
        assert_array_almost_equal(self.pdf_u, pdf_u_test)

    def test_cdf(self):
        cdf_u_test = self.copula.cdf(self.u)
        assert_array_almost_equal(self.cdf_u, cdf_u_test)

    def test_validate_params(self):
        pass

    def test_random_follow_pdf(self):
        pass


class TestIndependentCopula(CopulaTests):
    copula = IndependentCopula()
    dim = 2
    pdf_u = np.ones((10, 1))
    cdf_u = np.prod(CopulaTests.u, axis=1)


class TestGaussianCopula(CopulaTests):
    copula = GaussianCopula(cov=[[1., 0.8], [0.8, 1.]])
    dim = 2
    pdf_u = [1.03308741, 0.06507279, 0.72896012, 0.65389439, 16.45012399,
             0.34813218, 0.06768115, 0.08168840, 0.40521741, 1.26723470]
    cdf_u = [0.31906854, 0.06230196, 0.19284669, 0.39952707, 0.98144792,
             0.25677003, 0.05932818, 0.09605404, 0.35211017, 0.20885480]


class TestStudentCopula(CopulaTests):
    copula = StudentCopula(cov=[[1., 0.8], [0.8, 1.]], df=2)
    dim = 2
    pdf_u = [0.8303065, 0.1359839, 0.5157746, 0.4776421, 26.2173959,
             0.3070661, 0.1349173, 0.1597064, 0.3303230, 1.0482301]
    cdf_u = [0.31140349, 0.05942746, 0.18548601, 0.39143974, 0.98347259,
             0.24894028, 0.05653947, 0.09210693, 0.34447385, 0.20429882]

    def test_cdf(self, *args):
        pytest.skip("Not implemented.")


class TestClaytonCopula(CopulaTests):
    copula = ClaytonCopula(theta=1.2)
    dim = 2
    pdf_u = [1.0119836, 0.2072728, 0.8148839, 0.9481976, 2.1419659,
             0.6828507, 0.2040454, 0.2838497, 0.8197787, 1.1096360]
    cdf_u = [0.28520375, 0.06101690, 0.17703377, 0.36848218, 0.97772088,
             0.24082057, 0.05811908, 0.09343934, 0.33012582, 0.18738753]


class TestFrankCopula(CopulaTests):
    copula = FrankCopula(theta=3)
    dim = 2
    pdf_u = [0.9646599, 0.5627195, 0.8941964, 0.8364614, 2.9570945,
             0.6665601, 0.5779906, 0.5241333, 0.7156741, 1.1074024]
    cdf_u = [0.27467496, 0.05492539, 0.15995939, 0.36750702, 0.97782283,
             0.23412757, 0.05196265, 0.08676979, 0.32803721, 0.16320730]


class TestGumbelCopula(CopulaTests):
    copula = GumbelCopula(theta=1.5)
    dim = 2
    pdf_u = [1.0391696, 0.6539579, 0.9878446, 0.8679504, 16.6030932,
             0.7542073, 0.6668307, 0.6275887, 0.7477991, 1.1564864]
    cdf_u = [0.27194634, 0.05484380, 0.15668190, 0.37098420, 0.98176346,
             0.23422865, 0.05188260, 0.08659615, 0.33086960, 0.15803914]


def test_joint_distribution():
    dists = [stats.gamma(2), stats.norm]
    copula = GaussianCopula(cov=[[1., 0.5], [0.5, 1.]])
    joint_dist = JointDistribution(dists, copula)
    sample = joint_dist.random(512)
    assert sample.shape == (512, 2)
