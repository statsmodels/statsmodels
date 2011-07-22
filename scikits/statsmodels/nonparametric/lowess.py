"""
Univariate lowess function, like in R. 

References
----------
Hastie, Tibshirani, Friedman. (2009) The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition: Chapter 6.

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association 74 (368): 829-836.

"""

import numpy as np
from scipy.linalg import lstsq

def lowess(x,y, frac = 2./3):
    
    if x.ndim != 1:
        raise ValueError('x must be a vector')
    if y.ndim != 1:
        raise ValueError('y must be a vector')
    if y.shape[0] != x.shape[0] :
        raise ValueError('x and y must have same length')

    n = x.shape[0]
    fitted = np.zeros(n)
    
    k = int(frac * n)

    index_array = np.argsort(x)
    xcpy = x[index_array]
    ycpy = y[index_array]

    X = np.ones((k,2))

    nn_indices = [0,k]

    for i in xrange(n):
        
        left_width = xcpy[i] - xcpy[nn_indices[0]]
        right_width = xcpy[nn_indices[1]] - xcpy[i]
        width = max(left_width, right_width)

        weights = (xcpy[nn_indices[0]:nn_indices[1]] - xcpy[i])/width
        weights = (1-np.abs(weights)**3)**3
        weights = np.sqrt(weights)

        X[:,1] = xcpy[nn_indices[0]:nn_indices[1]]
        y_i = weights.reshape(k,) * ycpy[nn_indices[0]:nn_indices[1]]
        weights.shape = (k,1)

        beta = lstsq(weights * X, y_i)[0]
    
        fitted[i] = np.dot(beta, [1, xcpy[i]])
        
        update_nn(xcpy, nn_indices, i+1)

    out = np.array([xcpy, fitted]).T
    out.shape = (n,2)

    return out


def update_nn(x, cur_nn,i):
    
    if cur_nn[1]<x.size-1:
        left_dist = x[i] - x[cur_nn[0]]
        new_right_dist = x[cur_nn[1]+1] - x[i]
        if new_right_dist < left_dist:
            cur_nn[0] = cur_nn[0] + 1
            cur_nn[1] = cur_nn[1] + 1

def tricube(t):
    return np.where(np.abs(t)<1, (1-np.abs(t)**3)**3, 0)


