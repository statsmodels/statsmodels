# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 15:17:52 2011

Author: Mike
Author: Josef

mainly script for checking Kernel Regression
"""
import numpy as np

if __name__ == "__main__":
    #from statsmodels.sandbox.nonparametric import smoothers as s
    from statsmodels.sandbox.nonparametric import smoothers, kernels
    import matplotlib.pyplot as plt
    #from numpy import sin, array, random

    import time
    np.random.seed(500)
    nobs = 250
    sig_fac = 0.5
    #x = np.random.normal(size=nobs)
    x = np.random.uniform(-2, 2, size=nobs)
    #y = np.array([np.sin(i*5)/i + 2*i + (3+i)*np.random.normal() for i in x])
    y = np.sin(x*5)/x + 2*x + sig_fac * (3+x)*np.random.normal(size=nobs)

    K = kernels.Biweight(0.25)
    K2 = kernels.CustomKernel(lambda x: (1 - x*x)**2, 0.25, domain = [-1.0,
                               1.0])

    KS = smoothers.KernelSmoother(x, y, K)
    KS2 = smoothers.KernelSmoother(x, y, K2)


    KSx = np.arange(-3, 3, 0.1)
    start = time.time()
    KSy = KS.conf(KSx)
    KVar = KS.std(KSx)
    print(time.time() - start)    # This should be significantly quicker...
    start = time.time()          #
    KS2y = KS2.conf(KSx)         #
    K2Var = KS2.std(KSx)         #
    print(time.time() - start)    # ...than this.

    KSConfIntx, KSConfInty = KS.conf(15)

    print("Norm const should be 0.9375")
    print(K2.norm_const)

    print("L2 Norms Should Match:")
    print(K.L2Norm)
    print(K2.L2Norm)

    print("Fit values should match:")
    #print zip(KSy, KS2y)
    print(KSy[28])
    print(KS2y[28])

    print("Var values should match:")
    #print zip(KVar, K2Var)
    print(KVar[39])
    print(K2Var[39])

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(x, y, "+")
    ax.plot(KSx, KSy, "-o")
    #ax.set_ylim(-20, 30)
    ax2 = fig.add_subplot(222)
    ax2.plot(KSx, KVar, "-o")

    ax3 = fig.add_subplot(223)
    ax3.plot(x, y, "+")
    ax3.plot(KSx, KS2y, "-o")
    #ax3.set_ylim(-20, 30)
    ax4 = fig.add_subplot(224)
    ax4.plot(KSx, K2Var, "-o")

    fig2 = plt.figure()
    ax5 = fig2.add_subplot(111)
    ax5.plot(x, y, "+")
    ax5.plot(KSConfIntx, KSConfInty, "-o")

    import statsmodels.nonparametric.smoothers_lowess as lo
    ys = lo.lowess(y, x)
    ax5.plot(ys[:,0], ys[:,1], 'b-')
    ys2 = lo.lowess(y, x, frac=0.25)
    ax5.plot(ys2[:,0], ys2[:,1], 'b--', lw=2)

    #need to sort for matplolib plot ?
    xind = np.argsort(x)
    pmod = smoothers.PolySmoother(5, x[xind])
    pmod.fit(y[xind])

    yp = pmod(x[xind])
    ax5.plot(x[xind], yp, 'k-')
    ax5.set_title('Kernel regression, lowess - blue, polysmooth - black')

    #plt.show()
