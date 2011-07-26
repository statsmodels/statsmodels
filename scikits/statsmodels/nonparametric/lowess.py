"""
Univariate lowess function, like in R. 

References
----------
Hastie, Tibshirani, Friedman. (2009) The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition: Chapter 6.

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association 74 (368): 829-836.
"""

import numpy as np
from scipy.linalg import lstsq

def lowess(x,y, frac = 2./3, it = 3):
    """
    LOWESS (Locally Weighted Scatterplot Smoothing)
    
    A lowess function that outs smoothed estimates of y
    at the given x values from points (x,y)

    Parameters
    ----------
    x: 1-D numpy array
        The x-values of the observed points
    y: 1-D numpy array
        The y-values of the observed points
    frac: float
        Between 0 and 1. The fraction of the data used
        when estimating each y-value. 
    it: int
        The number of residual-based reweightings 
        to perform.

    Returns
    -------
    out: numpy array
        A numpy array with two columns. The first column
        is the sorted x values and the second column the
        associated estimated y-values. 

    Notes
    -----
    This lowess function implements the algorithm given in the
    reference below using local linear estimates. 

    Suppose the input data has N points. The algorithm works by
    estimating the true y_i by taking the frac*N closest points 
    to (x_i,y_i) based on their x values and estimating y_i
    using a weighted linear regression. The weight for (x_j,y_j)
    is tricube function applied to |x_i-x_j|. 
    
    If iter>0, then further weighted local linear regressions
    are performed, where the weights are the same as above
    times the bisquare function of the residuals. Each iteration
    takes approximately the same amount of time as the original fit,
    so these iterations are expensive. They are most useful when 
    the noise has extremely heavy tails, such as Cauchy noise. 
    Noise with less heavy-tails, such as t-distributions with df>2,
    are less problematic. The weights downgrade the influence of 
    points with large residuals. In the extreme case, points whose 
    residuals are larger than 6 times the median absolute residual
    are given weight 0.

    Some experimentation is likely required to find a good
    choice of frac and iter for a particular dataset.


    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression 
    and Smoothing Scatterplots". Journal of the American Statistical 
    Association 74 (368): 829-836.


    Examples
    --------
    The below allows a comparison between how different the fits from 
    lowess for different values of frac can be.
    
    >>> import numpy as np
    >>> import scikits.statsmodels.api as sm
    >>> x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=500)
    >>> y = np.sin(x) + np.random.normal(size=len(x))
    >>> z = lowess(x, y)
    >>> w = lowess(x, y, frac=1./3)

    This gives a similar comparison for when it is 0 vs not. 
    
    >>> import numpy as np
    >>> import scipy.stats as stats
    >>> import scikits.statsmodels.api as sm
    >>> x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=500)
    >>> y = np.sin(x) + stats.cauchy.rvs(size=len(x))
    >>> z = lowess(x, y, frac= 1./3, it=0)
    >>> w = lowess(x, y, frac=1./3)

    """




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
    xcpy = np.array(x[index_array], dtype ='float32')
    ycpy = y[index_array]
    
    fitted = initial_fit(xcpy, ycpy, k, n)

    if it>0:
        for i in xrange(it):
            fitted = stage_two_fit(xcpy, ycpy, fitted, k, n)

    out = np.array([xcpy, fitted]).T
    out.shape = (n,2)

    return out


def initial_fit(xcpy, ycpy, k, n):
   """
   The initial weighted local linear regression for lowess.
   """
    weights = np.zeros((k,1), dtype = xcpy.dtype)
    nn_indices = [0,k]
    
    X = np.ones((k,2))
    fitted = np.zeros(n)

    #beta = np.array([0,1])
    for i in xrange(n):
        
        left_width = xcpy[i] - xcpy[nn_indices[0]]
        right_width = xcpy[nn_indices[1]] - xcpy[i]
        width = max(left_width, right_width)

        #weights = (xcpy[nn_indices[0]:nn_indices[1]] - xcpy[i])/width
        wt_standardize(weights, xcpy[nn_indices[0]:nn_indices[1]], 
                            xcpy[i], width)
        #weights[:,0] = xcpy[nn_indices[0]:nn_indices[1]]
        #weights -= xcpy[i]
        #weights /= width
        tricube(weights)
        np.sqrt(weights, out=weights)

        X[:,1] = xcpy[nn_indices[0]:nn_indices[1]]
        y_i = weights.reshape(k,) * ycpy[nn_indices[0]:nn_indices[1]]
        weights.shape = (k,1)

        beta = lstsq(weights * X, y_i)[0]
    
        fitted[i] = beta[0] + beta[1]*xcpy[i]
        
        update_nn(xcpy, nn_indices, i+1)
    
    return fitted


def wt_standardize(weights, new_entries, xcpy_i, width):
    """
    The initial phase of creating the weights.
    Subtract the current x_i and divide by the width.
    """
    weights[:,0] = new_entries
    weights -= xcpy_i
    weights /= width



def stage_two_fit(xcpy, ycpy, fitted, k, n):
    """
    Additional weighted local linear regressions, if
    iter>0. They take into account the sizes of the residuals,
    to eliminate the effect of extreme outliers.
    """
    weights = np.zeros((k,1), dtype = xcpy.dtype)
    nn_indices = [0,k]
    X = np.ones((k,2))
    
    residual_weights = np.copy(ycpy)
    residual_weights -= fitted
    np.absolute(residual_weights, out=residual_weights)
    s = np.median(residual_weights)
    residual_weights /= (6*s)
    too_big = residual_weights>=1
    bisquare(residual_weights)
    residual_weights[too_big] = 0
    

    for i in xrange(n):
        
        left_width = xcpy[i] - xcpy[nn_indices[0]]
        right_width = xcpy[nn_indices[1]] - xcpy[i]
        width = max(left_width, right_width)
        
        #weights = (xcpy[nn_indices[0]:nn_indices[1]] - xcpy[i])/width
        weights[:,0] = xcpy[nn_indices[0]:nn_indices[1]]
        weights -= xcpy[i]
        weights /= width
        
        tricube(weights)
        np.sqrt(weights, out=weights)
        
        weights[:,0] *= residual_weights[nn_indices[0]:nn_indices[1]]

        X[:,1] = xcpy[nn_indices[0]:nn_indices[1]]
        y_i = weights.reshape(k,) * ycpy[nn_indices[0]:nn_indices[1]]
        weights.shape = (k,1)

        beta = lstsq(weights * X, y_i)[0]
    
        fitted[i] = beta[0] + beta[1] * xcpy[i]
        
        update_nn(xcpy, nn_indices, i+1)
    
    return fitted




def update_nn(x, cur_nn,i):
    """
    Update the endpoints of the nearest neighbors to
    the ith point.
    """
    if cur_nn[1]<x.size-1:
        left_dist = x[i] - x[cur_nn[0]]
        new_right_dist = x[cur_nn[1]+1] - x[i]
        if new_right_dist < left_dist:
            cur_nn[0] = cur_nn[0] + 1
            cur_nn[1] = cur_nn[1] + 1

def tricube(t):
    """
    The tricube function applied to a numpy array
    """
    #t = (1-np.abs(t)**3)**3
    np.absolute(t, out=t)
    mycube(t)
    np.negative(t, out = t)
    t += 1
    mycube(t)

def mycube(t):
    """
    Fast matrix cube
    """
    #t **= 3
    t2 = t*t
    t *= t2

def bisquare(t):
    """
    The bisquare function applied to a numpy array
    """
    #t = (1-t**2)**2
    t *= t
    np.negative(t, out=t)
    t += 1
    t *= t









