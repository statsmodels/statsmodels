'''subclassing kde

Author: josef pktd
'''

import numpy as np
import scipy
from scipy import stats
import matplotlib.pylab as plt


def plotkde(covfact):
    gkde.reset_covfact(covfact)
    kdepdf = gkde.evaluate(ind)
    plt.figure()
    # plot histgram of sample
    plt.hist(xn, bins=20, normed=1)
    # plot estimated density
    plt.plot(ind, kdepdf, label='kde', color="g")
    # plot data generating density
    plt.plot(ind, alpha * stats.norm.pdf(ind, loc=mlow) +
                  (1-alpha) * stats.norm.pdf(ind, loc=mhigh),
                  color="r", label='DGP: normal mix')
    plt.title('Kernel Density Estimation - ' + str(gkde.covfact))
    plt.legend()

from numpy.testing import assert_array_almost_equal, \
               assert_almost_equal, assert_
def test_kde_1d():
    np.random.seed(8765678)
    n_basesample = 500
    xn = np.random.randn(n_basesample)
    xnmean = xn.mean()
    xnstd = xn.std(ddof=1)
    print(xnmean, xnstd)

    # get kde for original sample
    gkde = stats.gaussian_kde(xn)

    # evaluate the density funtion for the kde for some points
    xs = np.linspace(-7,7,501)
    kdepdf = gkde.evaluate(xs)
    normpdf = stats.norm.pdf(xs, loc=xnmean, scale=xnstd)
    print('MSE', np.sum((kdepdf - normpdf)**2))
    print('axabserror', np.max(np.abs(kdepdf - normpdf)))
    intervall = xs[1] - xs[0]
    assert_(np.sum((kdepdf - normpdf)**2)*intervall < 0.01)
    #assert_array_almost_equal(kdepdf, normpdf, decimal=2)
    print(gkde.integrate_gaussian(0.0, 1.0))
    print(gkde.integrate_box_1d(-np.inf, 0.0))
    print(gkde.integrate_box_1d(0.0, np.inf))
    print(gkde.integrate_box_1d(-np.inf, xnmean))
    print(gkde.integrate_box_1d(xnmean, np.inf))

    assert_almost_equal(gkde.integrate_box_1d(xnmean, np.inf), 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box_1d(-np.inf, xnmean), 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box(xnmean, np.inf), 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box(-np.inf, xnmean), 0.5, decimal=1)

    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*intervall, decimal=2)
    assert_almost_equal(gkde.integrate_gaussian(xnmean, xnstd**2),
                        (kdepdf*normpdf).sum()*intervall, decimal=2)
##    assert_almost_equal(gkde.integrate_gaussian(0.0, 1.0),
##                        (kdepdf*normpdf).sum()*intervall, decimal=2)
