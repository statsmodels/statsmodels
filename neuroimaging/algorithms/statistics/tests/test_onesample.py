
import numpy as np
from  neuroimaging.algorithms.statistics import onesample
from scipy.stats import norm

from neuroimaging.testing import *

def test_estimate_varatio(p=1.0e-04, sigma2=1):
# This is a random test, but is design to fail only rarely....
    ntrial = 300
    n = 10
    random = np.zeros(10)
    rsd = np.zeros(n)
    sd = np.multiply.outer(np.linspace(0,1,40), np.ones(ntrial)) + np.ones((40,ntrial))

    for i in range(n):
        Y = np.random.standard_normal((40,ntrial)) * np.sqrt((sd**2 + sigma2))
        results = onesample.estimate_varatio(Y, sd)
        results = onesample.estimate_varatio(Y, sd)
        random[i] = results['random'].mean()
        rsd[i] = results['random'].std()

    # Compute the mean just to be sure it works

    W = 1. / (sd**2 + results['random'])
    mu = onesample.estimate_mean(Y, np.sqrt(sd**2 + results['random']))['mu']
    assert_almost_equal(mu, (W*Y).sum(0) / W.sum(0))

    rsd = np.sqrt((rsd**2).mean() / ntrial)
    T = np.fabs((random.mean() - sigma2) / (rsd / np.sqrt(n)))

    # should fail one in every 1/p trials at least for sigma > 0,
    # small values of sigma seem to have some bias
    if T > norm.ppf(1-p/2):
        raise ValueError('large T value, but algorithm works, could be a statistical failure')
