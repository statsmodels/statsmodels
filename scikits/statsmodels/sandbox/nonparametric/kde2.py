# -*- coding: utf-8 -*-

import numpy as np
import kernel

class KernelEstimate(object):
    def __init__(self, x, Kernel = None):
        xshape= np.shape(x)
        if len(xshape) == 1:
            n = 1
        else:
            n = xshape[1]
        print "%s dimensions" % n
        if Kernel is None:
            Kernel = kernel.Gaussian()
        if n > 1:
            if isinstance( Kernel, kernel.CustomKernel ):
                cov = np.identity(n)
                Kernel = kernel.NdKernel( n, kernels = Kernel, cov = cov )
        self._kernel = Kernel
        self.n = n
        self.x = x

    def density(self, x):
        return self._kernel.density(self.x, x)

    def __call__(self, x):
        return np.array([self.density(xx) for xx in x])

if __name__ == "__main__":
    PLOT = True
    from numpy import random
    import matplotlib.pyplot as plt
    # 1 D case
    random.seed(142)
    x = random.standard_t(4.2, size = 50)
    kern = kernel.Gaussian()
    kde = KernelEstimate( x, kern)
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

    if False:
        # 5 D case (easy)
        random.seed(142)
        mu = [1.0, 4.0, 3.5, -2.4, 0.0]
        sigma = np.matrix(
            [[ 0.6 - 0.1*abs(i-j) if i != j else 1.0 for j in xrange(5)] for i in xrange(5)])
        x = random.multivariate_normal(mu, sigma, size = 500)
        kern = kernel.Gaussian()
        kde = KernelEstimate( x, kern)

        # 3 D case (hard)
        random.seed(142)

