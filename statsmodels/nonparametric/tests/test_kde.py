import os
import numpy.testing as npt
import numpy as np
from statsmodels.sandbox.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDE
from scipy import stats

# get results from Stata

curdir = os.path.dirname(os.path.abspath(__file__))
rfname = os.path.join(curdir,'results','results_kde.csv')
#print rfname
KDEResults = np.genfromtxt(open(rfname, 'rb'), delimiter=",", names=True)

# setup test data

np.random.seed(12345)
Xi = mixture_rvs([.25,.75], size=200, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))

class CheckKDE(object):
    decimal_density = 7
    def test_density(self):
        npt.assert_almost_equal(self.res1.density, self.res_density,
                self.decimal_density)

class TestKDEGauss(CheckKDE):
    @classmethod
    def setupClass(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="gau", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["gau_d"]

class TestKDEEpanechnikov(CheckKDE):
    @classmethod
    def setupClass(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="epa", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["epa2_d"]

class TestKDETriangular(CheckKDE):
    @classmethod
    def setupClass(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="tri", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["tri_d"]

class TestKDEBiweight(CheckKDE):
    @classmethod
    def setupClass(cls):
        res1 = KDE(Xi)
        res1.fit(kernel="biw", fft=False, bw="silverman")
        cls.res1 = res1
        cls.res_density = KDEResults["biw_d"]

#NOTE: This is a knownfailure due to a definitional difference of Cosine kernel
#class TestKDECosine(CheckKDE):
#    @classmethod
#    def setupClass(cls):
#        res1 = KDE(Xi)
#        res1.fit(kernel="cos", fft=False, bw="silverman")
#        cls.res1 = res1
#        cls.res_density = KDEResults["cos_d"]

#weighted estimates taken from matlab so we can allow len(weights) != gridsize
class TestKdeWeights(CheckKDE):
    @classmethod
    def setupClass(cls):
        res1 = KDE(Xi)
        weights = np.linspace(1,100,200)
        res1.fit(kernel="gau", gridsize=50, weights=weights, fft=False,
                    bw="silverman")
        cls.res1 = res1
        rfname = os.path.join(curdir,'results','results_kde_weights.csv')
        cls.res_density = np.genfromtxt(open(rfname, 'rb'), skip_header=1)


class TestKDEGaussFFT(CheckKDE):
    @classmethod
    def setupClass(cls):
        cls.decimal_density = 2 # low accuracy because binning is different
        res1 = KDE(Xi)
        res1.fit(kernel="gau", fft=True, bw="silverman")
        cls.res1 = res1
        rfname2 = os.path.join(curdir,'results','results_kde_fft.csv')
        cls.res_density = np.genfromtxt(open(rfname2, 'rb'))


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
