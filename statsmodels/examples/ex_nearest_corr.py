# -*- coding: utf-8 -*-
"""Find near positive definite correlation and covariance matrices

Created on Sun Aug 19 15:25:07 2012
Author: Josef Perktold


TODO:
    add examples for cov_nearest from script log

Notes
-----
We are looking at eigenvalues before and after the conversion to psd matrix.
As distance measure for how close the change in the matrix is, we consider
the sum of squared differences (Frobenious norm without taking the square root)

"""

from __future__ import print_function
import numpy as np
from statsmodels.stats.correlation_tools import (
                 corr_nearest, corr_clipped, cov_nearest)

examples = ['all']

if 'all' in examples:
    # x0 is positive definite
    x0 = np.array([[1, -0.2, -0.9], [-0.2, 1, -0.2], [-0.9, -0.2, 1]])
    # x has negative eigenvalues, not definite
    x = np.array([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]])
    #x = np.array([[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]])

    n_fact = 2

    print('evals original', np.linalg.eigvalsh(x))
    y = corr_nearest(x, n_fact=100)
    print('evals nearest', np.linalg.eigvalsh(y))
    print(y)

    y = corr_nearest(x, n_fact=100, threshold=1e-16)
    print('evals nearest', np.linalg.eigvalsh(y))
    print(y)

    y = corr_clipped(x, threshold=1e-16)
    print('evals clipped', np.linalg.eigvalsh(y))
    print(y)

    np.set_printoptions(precision=4)
    print('\nMini Monte Carlo')
    # we are simulating a uniformly distributed symmetric matrix
    #     and find close positive definite matrix
    # original can be far away from positive definite,
    #     then original and converted matrices can be far apart in norm
    # results are printed for visual inspection of different cases

    k_vars = 5
    diag_idx = np.arange(k_vars)
    for ii in range(10):
        print()
        x = np.random.uniform(-1, 1, size=(k_vars, k_vars))
        x = (x + x.T) * 0.5
        x[diag_idx, diag_idx] = 1
        #x_std = np.sqrt(np.diag(x))
        #x = x / x_std / x_std[:,None]
        print()
        print(np.sort(np.linalg.eigvals(x)), 'original')

        yn = corr_nearest(x, threshold=1e-12, n_fact=200)
        print(np.sort(np.linalg.eigvals(yn)), ((yn - x)**2).sum(), 'nearest')

        yc = corr_clipped(x, threshold=1e-12)
        print(np.sort(np.linalg.eigvals(yc)), ((yc - x)**2).sum(), 'clipped')

    import time
    t0 = time.time()
    for _ in range(100):
        corr_nearest(x, threshold=1e-15, n_fact=100)

    t1 = time.time()
    for _ in range(1000):
        corr_clipped(x, threshold=1e-15)
    t2 = time.time()

    print('\ntime (nearest, clipped):', t1 - t0, t2 - t1)

if 'all' in examples:
    # example for test case against R
    x2 = np.array([ 1,     0.477, 0.644, 0.478, 0.651, 0.826,
                   0.477, 1,     0.516, 0.233, 0.682, 0.75,
                   0.644, 0.516, 1,     0.599, 0.581, 0.742,
                   0.478, 0.233, 0.599, 1,     0.741, 0.8,
                   0.651, 0.682, 0.581, 0.741, 1,     0.798,
                   0.826, 0.75,  0.742, 0.8,   0.798, 1]).reshape(6,6)

    y1 = corr_nearest(x2, threshold=1e-15, n_fact=200)
    y2 = corr_clipped(x2, threshold=1e-15)
    print('\nmatrix 2')
    print(np.sort(np.linalg.eigvals(x2)), 'original')
    print(np.sort(np.linalg.eigvals(y1)), ((y1 - x2)**2).sum(), 'nearest')
    print(np.sort(np.linalg.eigvals(y1)), ((y2 - x2)**2).sum(), 'clipped')
