# -*- coding: utf-8 -*-
"""

Created on Sun Aug 19 15:25:07 2012

Author: Josef Perktold


TODO:
    add examples for nearest_cov from script log
    convert to tests

"""

import numpy as np
from statsmodels.tools.correlation_tools import (
                 nearest_corr, clipped_corr, nearest_cov)

examples = ['all']

if 'all' in examples:
    x = np.array([[1, -0.2, -0.9], [-0.2, 1, -0.2], [-0.9, -0.2, 1]])
    x = np.array([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]])
    #x = np.array([[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]])

    n_fact = 2

    print np.linalg.eigvals(x)
    y = nearest_corr(x, n_fact=100)
    print np.linalg.eigvals(y)
    print y

    y = nearest_corr(x, n_fact=100, threshold=1e-16)
    print np.linalg.eigvals(y)
    print y

    y = clipped_corr(x, threshold=1e-16)
    print np.linalg.eigvals(y)
    print y


    print '\nMini Monte Carlo'
    #original can be far away from positive definite
    k_vars = 5
    diag_idx = np.arange(k_vars)
    for ii in range(10):
        print
        x = np.random.uniform(-1, 1, size=(k_vars, k_vars))
        x = (x + x.T) * 0.5
        x[diag_idx, diag_idx] = 1
        #x_std = np.sqrt(np.diag(x))
        #x = x / x_std / x_std[:,None]

        print np.sort(np.linalg.eigvals(x)), 'original'

        yn = nearest_corr(x, threshold=1e-15, n_fact=200)
        print np.sort(np.linalg.eigvals(yn)), ((yn - x)**2).sum()

        yc = clipped_corr(x, threshold=1e-15)
        print np.sort(np.linalg.eigvals(yc)), ((yc - x)**2).sum()

    import time
    t0 = time.time()
    for _ in range(100):
        nearest_corr(x, threshold=1e-15, n_fact=100)

    t1 = time.time()
    for _ in range(1000):
        clipped_corr(x, threshold=1e-15)
    t2 = time.time()

    print 'time:', t1 - t0, t2 - t1

    x = np.array([ 1,     0.477, 0.644, 0.478, 0.651, 0.826,
                   0.477, 1,     0.516, 0.233, 0.682, 0.75,
                   0.644, 0.516, 1,     0.599, 0.581, 0.742,
                   0.478, 0.233, 0.599, 1,     0.741, 0.8,
                   0.651, 0.682, 0.581, 0.741, 1,     0.798,
                   0.826, 0.75,  0.742, 0.8,   0.798, 1]).reshape(6,6)

    y1 = nearest_corr(x, threshold=1e-15, n_fact=200)
    y2 = clipped_corr(x, threshold=1e-15)
