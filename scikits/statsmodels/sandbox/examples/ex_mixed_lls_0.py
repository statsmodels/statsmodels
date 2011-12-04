# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 10:15:55 2011

Author: Josef Perktold
"""

import numpy as np

from scikits.statsmodels.sandbox.mixed import Mixed, Unit

examples = ['ex1']

if 'ex1' in examples:
    #np.random.seed(54321)
    np.random.seed(978326)
    nsubj = 100   #400 is too slow for testing, 38seconds for 100 iterations
    units  = []

    nobs_i = 4 #number of observations per unit

    from scikits.statsmodels.sandbox.formula import Term
    fixed = Term('f')
    random = Term('r')
    response = Term('y')

    nx = 4  #number fixed effects
    nz = 2 ##number fixed effects
    beta = np.ones(nx)
    gamma = 0.5 * np.ones(nz)   #mean of random effect
    gamma[0] = 0
    gamma_re_true = []
    for i in range(nsubj):
        #created as observation in columns
        gamma_re = gamma + 0.2 * np.random.standard_normal(nz) #random effect/coefficient
        gamma_re_true.append(gamma_re)
        if i > 20: nobs_i = 6
        X = np.random.standard_normal((nx, nobs_i)) #.T
        Z = np.random.standard_normal((nz-1, nobs_i)) #.T
        Z = np.vstack((np.ones(nobs_i), Z)) #eps sig_e
        noise = 0.1 * np.random.randn(nobs_i)
        #Y = R.standard_normal((n,)) + d * 4
        Y = np.dot(X.T, beta) + np.dot(Z.T, gamma_re) + noise
        #Y = np.dot(X, beta) + d * 1.
        X = np.vstack((X,Z))  #necessary to force mean of RE to zero !?
        units.append(Unit({'f':X, 'r':Z, 'y':Y}))

    #m = Mixed(units, response)#, fixed, random)
    m = Mixed(units, response, fixed, random)
    #m = Mixed(units, response, fixed + random, random)
    import time
    t0 = time.time()
    m.initialize()
    m.fit(niter=2000, rtol=1.0e-5, params_rtol=1e-6, params_atol=1e-6)
    t1 = time.time()
    print 'time for initialize and fit', t1-t0
    print 'number of iterations', m.iterations
    #print dir(m)
    #print vars(m)
    print '\nestimates for fixed effects'
    print m.a
    print m.params
    bfixed_cov = m.cov_fixed()
    print 'beta fixed standard errors'
    print np.sqrt(np.diag(bfixed_cov))

    print m.bse
    b_re = m.params_random_units
    print 'RE mean:', b_re.mean(0)
    print 'RE columns std', b_re.std(0)
    print 'np.cov(b_re, rowvar=0)'
    print np.cov(b_re, rowvar=0)
    print 'std of above'
    print np.sqrt(np.diag(np.cov(b_re, rowvar=0)))
    print 'm.cov_random()'
    print m.cov_random()
    print 'std of above'
    print np.sqrt(np.diag(m.cov_random()))

    print '\n(non)convergence of llf'
    print m.history['llf'][-4:]
    print 'convergence of parameters'
    #print np.diff(np.vstack(m.history[-4:])[:,1:],axis=0)
    print np.diff(np.vstack(m.history['params'][-4:]),axis=0)
    print 'convergence of D'
    print np.diff(np.array(m.history['D'][-4:]), axis=0)

    #zdotb = np.array([np.dot(unit.Z, unit.b) for unit in m.units])
    zb = np.array([(unit.Z * unit.b[None,:]).sum(0) for unit in m.units])
    '''if Z is not included in X:
    >>> np.dot(b_re.T, b_re)/100
    array([[ 0.03270611, -0.00916051],
           [-0.00916051,  0.26432783]])
    >>> m.cov_random()
    array([[ 0.0348722 , -0.00909159],
           [-0.00909159,  0.26846254]])
    >>> #note cov_random doesn't subtract mean!
    '''
    print '\nchecking the random effects distribution and prediction'
    gamma_re_true = np.array(gamma_re_true)
    print 'mean of random effect true', gamma_re_true.mean(0)
    print 'mean from fixed effects   ', m.params[-2:]
    print 'mean of estimated RE      ', b_re.mean(0)

    print
    absmean_true = np.abs(gamma_re_true).mean(0)
    mape = ((m.params[-2:] + b_re) / gamma_re_true - 1).mean(0)*100
    mean_abs_perc = np.abs((m.params[-2:] + b_re) - gamma_re_true).mean(0) \
                       / absmean_true*100
    median_abs_perc = np.median(np.abs((m.params[-2:] + b_re) - gamma_re_true), 0) \
                         / absmean_true*100
    rmse_perc = ((m.params[-2:] + b_re) - gamma_re_true).std(0) \
                  / absmean_true*100
    print 'mape           ', mape
    print 'mean_abs_perc  ', mean_abs_perc
    print 'median_abs_perc', median_abs_perc
    print 'rmse_perc (std)', rmse_perc
