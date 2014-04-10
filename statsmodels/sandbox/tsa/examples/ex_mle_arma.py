# -*- coding: utf-8 -*-
"""
TODO: broken because of changes to arguments and import paths
fixing this needs a closer look

Created on Thu Feb 11 23:41:53 2010
Author: josef-pktd
copyright: Simplified BSD see license.txt
"""
from __future__ import print_function
import numpy as np
from numpy.testing import assert_almost_equal

import matplotlib.pyplot as plt

import numdifftools as ndt

import statsmodels.api as sm
from statsmodels.sandbox import tsa
from statsmodels.tsa.arma_mle import Arma  # local import
from statsmodels.tsa.arima_process import arma_generate_sample

examples = ['arma']
if 'arma' in examples:

    print("\nExample 1")
    print('----------')
    ar = [1.0, -0.8]
    ma = [1.0,  0.5]
    y1 = arma_generate_sample(ar,ma,1000,0.1)
    y1 -= y1.mean() #no mean correction/constant in estimation so far

    arma1 = Arma(y1)
    arma1.nar = 1
    arma1.nma = 1
    arma1res = arma1.fit_mle(order=(1,1), method='fmin')
    print(arma1res.params)

    #Warning need new instance otherwise results carry over
    arma2 = Arma(y1)
    arma2.nar = 1
    arma2.nma = 1
    res2 = arma2.fit(method='bfgs')
    print(res2.params)
    print(res2.model.hessian(res2.params))
    print(ndt.Hessian(arma1.loglike, stepMax=1e-2)(res2.params))
    arest = tsa.arima.ARIMA(y1)
    resls = arest.fit((1,0,1))
    print(resls[0])
    print(resls[1])



    print('\nparameter estimate - comparing methods')
    print('---------------------------------------')
    print('parameter of DGP ar(1), ma(1), sigma_error')
    print([-0.8, 0.5, 0.1])
    print('mle with fmin')
    print(arma1res.params)
    print('mle with bfgs')
    print(res2.params)
    print('cond. least squares uses optim.leastsq ?')
    errls = arest.error_estimate
    print(resls[0], np.sqrt(np.dot(errls,errls)/errls.shape[0]))

    err = arma1.geterrors(res2.params)
    print('cond least squares parameter cov')
    #print(np.dot(err,err)/err.shape[0] * resls[1])
    #errls = arest.error_estimate
    print(np.dot(errls,errls)/errls.shape[0] * resls[1])
#    print('fmin hessian')
#    print(arma1res.model.optimresults['Hopt'][:2,:2])
    print('bfgs hessian')
    print(res2.model.optimresults['Hopt'][:2,:2])
    print('numdifftools inverse hessian')
    print(-np.linalg.inv(ndt.Hessian(arma1.loglike, stepMax=1e-2)(res2.params))[:2,:2])

    print('\nFitting Arma(1,1) to squared data')
    arma3 = Arma(y1**2)
    res3 = arma3.fit(method='bfgs')
    print(res3.params)

    print('\nFitting Arma(3,3) to data from DGP Arma(1,1)')
    arma4 = Arma(y1)
    arma4.nar = 3
    arma4.nma = 3
    #res4 = arma4.fit(method='bfgs')
    res4 = arma4.fit(start_params=[-0.5, -0.1,-0.1,0.2,0.1,0.1,0.5])
    print(res4.params)
    print('numdifftools inverse hessian')
    pcov = -np.linalg.inv(ndt.Hessian(arma4.loglike, stepMax=1e-2)(res4.params))
    #print(pcov)
    print('standard error of parameter estimate from Hessian')
    pstd = np.sqrt(np.diag(pcov))
    print(pstd)
    print('t-values')
    print(res4.params/pstd)
    print('eigenvalues of pcov:')
    print(np.linalg.eigh(pcov)[0])
    print('sometimes they are negative')


    print("\nExample 2 - DGP is Arma(3,3)")
    print('-----------------------------')
    ar = [1.0, -0.6, -0.2, -0.1]
    ma = [1.0,  0.5, 0.1, 0.1]
    y2 = arest.generate_sample(ar,ma,1000,0.1)
    y2 -= y2.mean() #no mean correction/constant in estimation so far


    print('\nFitting Arma(3,3) to data from DGP Arma(3,3)')
    arma4 = Arma(y2)
    arma4.nar = 3
    arma4.nma = 3
    #res4 = arma4.fit(method='bfgs')
    print('\ntrue parameters')
    print('ar', ar[1:])
    print('ma', ma[1:])
    res4 = arma4.fit(start_params=[-0.5, -0.1,-0.1,0.2,0.1,0.1,0.5])
    print(res4.params)
    print('numdifftools inverse hessian')
    pcov = -np.linalg.inv(ndt.Hessian(arma4.loglike, stepMax=1e-2)(res4.params))
    #print(pcov)
    print('standard error of parameter estimate from Hessian')
    pstd = np.sqrt(np.diag(pcov))
    print(pstd)
    print('t-values')
    print(res4.params/pstd)
    print('eigenvalues of pcov:')
    print(np.linalg.eigh(pcov)[0])
    print('sometimes they are negative')

    arma6 = Arma(y2)
    arma6.nar = 3
    arma6.nma = 3
    res6 = arma6.fit(start_params=[-0.5, -0.1,-0.1,0.2,0.1,0.1,0.5],
                      method='bfgs')
    print('\nmle with bfgs')
    print(res6.params)
    print('pstd with bfgs hessian')
    hopt = res6.model.optimresults['Hopt']
    print(np.sqrt(np.diag(hopt)))

    #fmin estimates for coefficients in ARMA(3,3) look good
    #but not inverse Hessian, sometimes negative values for variance

