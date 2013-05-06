# Copyright (c) 2013 Ana Martinez Pardo <anamartinezpardo@gmail.com>
# License: BSD-3 [see LICENSE.txt]

import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import mv_mixture_rvs, MixtureDistribution
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats

class TestMixtureDistributions(npt.TestCase):

    def test_mixture_rvs(self):
        # We use a sample of 1M observations to compare using up to 2 decimal
        # precission with confidence
        mix = MixtureDistribution()
        res = mix.rvs([.75,.25], 1000000, dist=[stats.norm, stats.norm], kwargs =
                (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
        npt.assert_almost_equal(
                np.array([res.std(),res.mean(),res.var()]),
                np.array([1,-0.5,1]),
                decimal=2)

    def test_mv_mixture_rvs(self):
        cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                       [ 0.5 ,  1.5 ,  0.6 ],
                       [ 0.75,  0.6 ,  2.  ]])
        mu = np.array([-1, 0.0, 2.0])
        mu2 = np.array([4, 2.0, 2.0])
        mvn3 = mvd.MVNormal(mu, cov3)
        mvn32 = mvd.MVNormal(mu2, cov3/2., 4)
        res = mv_mixture_rvs([0.4, 0.6], 1000000, [mvn3, mvn32], 3)
        npt.assert_almost_equal(
                np.array([res.std(),res.mean(),res.var()]),
                np.array([1.874,1.733,3.512]),
                decimal=1)

    def test_mixture_pdf(self):
        mix = MixtureDistribution()
        grid = np.linspace(-4,4, 1000000)
        res = mix.pdf(grid, [1/3.,2/3.], dist=[stats.norm, stats.norm], kwargs=
                (dict(loc=-1,scale=.25),dict(loc=1,scale=.75)))
        npt.assert_almost_equal(
                np.array([res.std(),res.mean(),res.var()]),
                np.array([0.1486,0.1249,0.022]),
                decimal=2)

    def test_mixture_cdf(self):
        mix = MixtureDistribution()
        grid = np.linspace(-4,4, 1000000)
        res = mix.cdf(grid, [1/3.,2/3.], dist=[stats.norm, stats.norm], kwargs=
                   (dict(loc=-1,scale=.25),dict(loc=1,scale=.75)))
        npt.assert_almost_equal(
                np.array([res.std(),res.mean(),res.var()]),
                np.array([0.4088,0.4583,0.1671]),
                decimal=2)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
