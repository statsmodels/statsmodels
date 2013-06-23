#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Quantile Regression and diagnostics

http://pastebin.com/egLfUkms
translated from matlab by Christian Prinoth
see pystatsmodel mailing list

'''
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.linalg import inv
from statsmodels.tools.tools import chain_dot as dot

def bootstrp(n_boot, func, *args):
    #matlab imitation, actually not needed
    pass


#function [beta tstats VCboot itrat PseudoR2 betaboot]=quantilereg(y,x,p)
def quantilereg(y,x,p):
    '''
     estimates quantile regression based on weighted least squares.
     constant term is added to x matrix
    __________________________________________________________________________

     Inputs:
      y: Dependent variable
      x: matrix of independent variables
      p: quantile

     Outputs:
      beta:     estimated Coefficients.
      tstats:   T- students of the coefficients.
      VCboot:   Variance-Covariance of coefficients by bootstrapping method.
      itrat:    number of iterations for convergence to roots.
      PseudoR2: in quatile regression another definition of R2 is used namely PseudoR2.
      betaboot: estimated coefficients by bootstrapping method.

     This code can be used for quantile regression estimation as whole,and LAD
     regression as special case of it, when one sets p=0.5.

     Copyright(c) Shapour Mohammadi, University of Tehran, 2008
     shmohammadi@gmail.com

     Translated to python with permission from original author by Christian Prinoth (christian at prinoth dot name)

    Ref:
     1-Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
     2-Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
     3-LeSage, J. P.(1999),Applied Econometrics Using MATLAB,

     Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
     Regression, Robust Estimation.
    __________________________________________________________________________
    '''

    tstats = 0
    VCboot = 0
    itrat = 0
    PseudoR2 = 0
    betaboot = 0

    ry = len(y)      # nobs
    rx, cx = x.shape  #nobs, k_vars
    x = np.c_[np.ones(rx), x]  # adding constant -> TODO: remove
    cx = cx + 1
    #______________Finding first estimates by solving the system_______________
    # Some lines of this section is based on a code written by
    # James P. Lesage in Applied Econometrics Using MATLAB(1999).PP. 73-4.
    itrat = 0
    xstar = x
    diff = 1
    beta = np.ones(cx)  # TODO: better start, inf? used only for convergence check
    z = np.zeros((rx, cx))
    while itrat < 1000 and diff > 1e-6:   #TODO: Why `and` instead of `or`?
        # Iterative Weighted Least Squares
        itrat += 1
        beta0 = beta
        #JP: solve linear system, use pinv or linalg.solve
        beta = dot(inv(dot(xstar.T, x)), xstar.T, y)
        resid = y - dot(x, beta)
        #JP: bound resid away from zero,
        #    why not symmetric: * np.sign(resid), shouldn't matter
        resid[np.abs(resid) < .000001] = .000001
        resid[resid < 0] = p * resid[resid < 0]
        resid[resid > 0] = (1 - p) * resid[resid > 0]
        resid = np.abs(resid)  # these are the new weights

        for i in range(cx):
            #TODO: this should just be broadcasting
            z[:,i] = x[:,i] / resid

        xstar = z     #TODO: why do we have z and xstar, z looks unnecessary
        beta1 = beta  #TODO: doesn't look like it's needed
        diff = np.max(np.abs(beta1 - beta0))

    return beta

    #JP: most of the following should go into a Results class

    #_______estimating variances based on Green 2008(quantile regression)______

    e = y - dot(x, beta)
    #JP: the following shows up similarily also in nonparametric
    iqre = stats.scoreatpercentile(e, 75) - stats.scoreatpercentile(e, 25)
    if p == 0.5:
        h = 0.9 * np.std(e) / (ry**0.2)
    else:
        h = 0.9 * np.min(np.std(e), iqre / 1.34) / (ry**0.2)

    u = e / h
    fhat0 = (1. / (ry * h)) * (np.sum(np.exp(-u) / ((1 + np.exp(-u))**2)))

    #TODO: We don't need D, we can just work with diagon (diagonal 1-D)
    D = np.zeros((ry, ry))
    diagon = np.diag(D)
    diagon[e > 0] = (p / fhat0)**2
    diagon[e <= 0] = ((1-p) / fhat0)**2
    D = np.diag(diagon)
    VCQ = np.dot(inv(dot(x.T, x)), dot(x.T, D, x), inv(np.dot(x.T, x)))

    #____________________Standarad errors and t-stats_________________________

    tstats = beta / np.sqrt(np.diag(VCQ))
    stderrors = np.sqrt(np.diag(VCQ))
    PValues = 2 * (1 - stats.t.cdf(np.abs(tstats), ry - cx))

    #______________________________ Quasi R square_____________________________

    ef = y - dot(x, beta)
    ef[ef < 0] = (1 - p) * ef[ef < 0]
    ef[ef > 0] = p * ef[ef > 0]
    ef = np.abs(ef)

    ered = y - stats.scoreatpercentile(y, p * 100)
    ered[ered < 0] = (1 - p) * ered[ered < 0]
    ered[ered > 0] = p * ered[ered > 0]
    ered = np.abs(ered)

    PseudoR2 = 1 - np.sum(ef) / np.sum(ered)

    #__________________Bootstrap standard deviation (Green 2008)_______________

    betaboot = np.zeros((cx, cx))
    for ii in range(100):
        #bootm, estar = bootstrp(1, np.mean, e)
        #Note bootm is not used, estar is index
        #BUG: bootstrp undefined, matlab function ?
        # Parametric Bootstrap
        # use simple sampling of index instead
        estar = np.random.randint(0, len(e), size=len(e))
        ystar = dot(x, beta) + e[estar]
        #
        itratstar = 0
        xstarstar = x
        diffstar = 1
        betastar = np.ones(cx)
        while itratstar < 1000 and diffstar > 1e-6:
            itratstar = itratstar + 1
            betastar0 = betastar
            betastar = dot(inv(dot(xstarstar.T, x)), xstarstar.T, ystar)
            #
            residstar = ystar - dot(x, betastar)
            residstar[np.abs(residstar) < .000001] = .000001
            residstar[residstar < 0] = p * residstar[residstar < 0]
            residstar[residstar > 0] = (1 - p) * residstar[residstar > 0]
            residstar = np.abs(residstar)
            zstar = np.zeros((rx, cx))

            for i in range(cx):
                # TODO: broadcast
                zstar[:,i] = x[:,i] / residstar

            xstarstar = zstar
            beta1star = betastar
            diffstar = np.max(np.abs(beta1star - betastar0))
        #
        betaboot = [betaboot + dot((betastar - beta), (betastar - beta).T)]

    VCboot = betaboot / 100.
    #
    tstatsboot = beta / np.diag(VCboot)**0.5
    stderrorsboot = np.diag(VCboot)**0.5
    PValuesboot = 2 * (1 - stats.t.cdf(np.abs(tstatsboot), ry - cx))

    #_______________________________Display Results____________________________

    print
    print(' Results of Quantile Regression')
    print('_'*70)
    print("%10s %10s %10s %10s %10s %10s %10s" % ('Coef.', 'SE.Ker', 't.Ker',
                                                  'P.Ker', 'SE.Boot', 't.Boot',
                                                  'P.Boot'))
    print('_'*70)
    print("%10f %10f %10f %10f %10f %10f %10f" % (beta, stderrors, tstats,
                                                  PValues, stderrorsboot,
                                                  tstatsboot, PValuesboot))
    print('_'*70)
    print('Pseudo R2: %10f' % PseudoR2 )
    print('_'*70)
