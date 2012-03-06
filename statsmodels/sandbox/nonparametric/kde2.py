# -*- coding: utf-8 -*-

import numpy as np
import kernel as kernels
import bandwidths as bw

#TODO: should this be a function?
class KDE(object):
    """
    Kernel Density Estimator

    Parameters
    ----------
    x : array-like
        N-dimensional array from which the density is to be estimated
    kernel : Kernel Class
        Should be a class from *

    """
    #TODO: amend docs for Nd case?
    def __init__(self, x, kernel = None):

        x = np.asarray(x)

        if x.ndim == 1:
            x = x[:,None]

        nobs, n_series = x.shape
#        print "%s dimensions" % n_series

        if kernel is None:
            kernel = kernels.Gaussian() # no meaningful bandwidth yet

        if n_series > 1:
            if isinstance( kernel, kernels.CustomKernel ):
                kernel = kernels.NdKernel(n_series, kernels = kernel)
        self.kernel = kernel
        self.n = n_series # TODO change attribute
        self.x = x

    def density(self, x):
        return self.kernel.density(self.x, x)

    def __call__(self, x, h = "scott"):
        return np.array([self.density(xx) for xx in x])

    def evaluate(self, x, h = "silverman"):
        density = self.kernel.density
        return np.array([density(xx) for xx in x])

if __name__ == "__main__":
    PLOT = True
    from numpy import random
    import matplotlib.pyplot as plt
    import bandwidths as bw

    # 1 D case
    random.seed(142)
    x = random.standard_t(4.2, size = 50)
    h = bw.bw_silverman(x)
    #NOTE: try to do it with convolution
    support = np.linspace(-10,10,512)


    kern = kernels.Gaussian(h = h)
    kde = KDE( x, kern)
    print kde.density(1.015469)
    print 0.2034675
    Xs = np.arange(-10,10,0.1)

    if PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(Xs, kde(Xs), "-")
        ax.set_ylim(-10, 10)
        ax.set_ylim(0,0.4)
        plt.show()

    # 2 D case
#    from statsmodels.sandbox.nonparametric.testdata import kdetest
#    x = zip(kdetest.faithfulData["eruptions"], kdetest.faithfulData["waiting"])
#    x = np.array(x)
#    H = kdetest.Hpi
#    kern = kernel.NdKernel( 2 )
#    kde = KernelEstimate( x, kern )
#    print kde.density( np.matrix( [1,2 ]).T )


    # 5 D case
#    random.seed(142)
#    mu = [1.0, 4.0, 3.5, -2.4, 0.0]
#    sigma = np.matrix(
#        [[ 0.6 - 0.1*abs(i-j) if i != j else 1.0 for j in xrange(5)] for i in xrange(5)])
#    x = random.multivariate_normal(mu, sigma, size = 100)
#    kern = kernel.Gaussian()
#    kde = KernelEstimate( x, kern )
