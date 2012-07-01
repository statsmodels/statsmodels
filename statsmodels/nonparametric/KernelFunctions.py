# This module adds to kernels that are able to handle
# categorical variables (both ordered and unordered)


# In addition, this module contains kernel functions
# This is a slight deviation from the current approach in statsmodels.
# nonparametric.kernels where each kernel is a class object

# Having kernel functions rather than classes makes extension to a multivariate
# kernel density estimation much easier

# NOTE: As it is, this module does not interact with the existing API

import numpy as np


def AitchisonAitken(h, Xi, x, num_levels=False):
    """
    Returns the value of the Aitchison and Aitken's (1976) Kernel. Used for
    unordered discrete random variable

    Parameters
    ----------
    h : 1D array of length K
        The bandwidths used to estimate the value of the kernel function
    Xi : 2D array with shape (N,K) array 
        The value of the training set
    x: 1D array of length K
        The value at which the kernel density is being estimated
    num_levels: Boolean
        Gives the user the option to specify the number of levels for the RV
        If False then the number of levels for the RV are calculated
        from the data
    Returns
    -------
    kernel_value : K-dimensional array
    The value of the kernel function at each training point for each var

    Notes
    -----

    see [2] p.18 for details

    The value of the kernel L if
    
    .. math::`X_{i}=x`

    is:

    .. math:: 1-\lambda

    else:

    \frac{\lambda}{c-1}

    where :math:`c` is the number of levels plus one of the RV
 
    References
    ----------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
    """
##    Xi=np.asarray(Xi)
##    N = np.shape(Xi)[0]
##    if Xi.ndim>1:
##        K=np.shape(Xi)[1]
##    else:
##        K=1
##
##    if K==0: return Xi
##    h = np.asarray(h, dtype = float)
    Xi = np.abs(np.asarray(Xi, dtype=int))
    x = np.abs(np.asarray(x, dtype=int))
    
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
        N = np.shape(Xi)[0]
    elif Xi.ndim == 1:
        K = 1
        N = np.shape(Xi)[0]
    else:  # ndim ==0 so Xi is a single point (number)
        K = 1
        N = 1    
    if K == 0:
        return Xi
    
    h = np.asarray(h, dtype=float)
    Xi = Xi.reshape([N, K])
    c = np.asarray([len(np.unique(Xi[:, i])) for i in range(K)],
                   dtype=int)
    if num_levels:
        c = num_levels
    kernel_value = np.empty(Xi.shape)
    kernel_value.fill(h / (c - 1))
    inDom = (Xi == x) * (1 - h)
    kernel_value[Xi == x] = inDom[Xi == x]
    kernel_value = kernel_value.reshape([N, K])
    return kernel_value


def WangRyzin(h, Xi, x):    
    """
    Returns the value of the Wang and van Ryzin's (1981) Kernel. Used for
    ordered discrete random variable
    
    Parameters
    ----------
    h : K dimensional array
        The bandwidths used to estimate the value of the kernel function
    Xi : K dimensional array 
        The value of the training set
    x: 1D array of length K
        The value at which the kernel density is being estimated
    Returns
    -------
    kernel_value : float
        The value of the kernel function

    Notes
    -----
    See p. 19 in [2]
    
    The value of the kernel L if
    
    .. math::`X_{i}=x`

    is:

    .. math:: 1-\lambda

    else:

    \frac{1-\lambda}{2}\lambda^{|X_{i}-x|}        
    References
    ----------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
    
    """
    Xi = np.abs(np.asarray(Xi, dtype=int))
    x = np.abs(np.asarray(x, dtype=int))
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
        N = np.shape(Xi)[0]
    elif Xi.ndim == 1:
        K = 1
        N = np.shape(Xi)[0]
    else:  # ndim ==0 so Xi is a single point (number)
        K = 1
        N = 1
    if K == 0:
        return Xi
    h = np.asarray(h, dtype=float)
    Xi = Xi.reshape([N, K])
    kernel_value = (0.5 * (1 - h) * (h ** abs(Xi - x)))
    kernel_value = kernel_value.reshape([N, K])
    inDom = (Xi == x) * (1 - h)
    kernel_value[Xi == x] = inDom[Xi == x]
    return kernel_value


def Epanechnikov(h, Xi, x):
    """
    Returns the value of the Epanechnikov (1969) Kernel. Used for
    continuous random variables
    
    Parameters
    ----------
    h : K dimensional array
        The bandwidths used to estimate the value of the kernel function
    Xi : K dimensional array
    The value of the training set
    x: 1D array of length K
        The value at which the kernel density is being estimated
    Returns
    -------
    kernel_value : 1D array
        The value of the kernel function
        
    References
    ----------
    Racine, J. "Nonparametric econometrics: A Primer" : Foundations and Trends in Econometrics.
    """    
    Xi = np.asarray(Xi)
    x = np.asarray(x)
    h = np.asarray(h, dtype=float)
    N = np.shape(Xi)[0]
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
    else:
        K = 1
    if K == 0:
        return Xi
    z = (Xi - x) / h
    InDom = (-np.sqrt(5) <= z) * (z <= np.sqrt(5))
    kernel_value = InDom * (0.75 * (1 - 0.2 * z ** 2) * (1 / np.sqrt(5)))
    kernel_value = kernel_value.reshape([N, K])
    # NOTE: There is a slight discrepancy between this
    # formulation of the Epanechnikov kernel and
    # the one in kernels.py.
    # TODO: Check Silverman and reconcile the discrepancy
    return kernel_value


def Gaussian(h, Xi, x):
    Xi = np.asarray(Xi)
    x = np.asarray(x)
    h = np.asarray(h, dtype=float)
    N = np.shape(Xi)[0]
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
    else:
        K = 1
    if K == 0:
        return Xi
    z = (Xi - x) / h
    kernel_value = (1. / np.sqrt(2 * np.pi)) * np.exp(- z ** 2 / 2.)
    kernel_value = kernel_value.reshape([N, K])
    return kernel_value


def Gaussian_Convolution(h, Xi, x):
    """
    Calculates the Gaussian Convolution Kernel
    """
    Xi = np.asarray(Xi)
    x = np.asarray(x)
    h = np.asarray(h, dtype=float)
    N = np.shape(Xi)[0]
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
    else:
        K = 1
    if K == 0:
        return Xi
    z = (Xi - x) / h
    kernel_value = (1. / np.sqrt(4 * np.pi)) * np.exp(- z ** 2 / 4.)
    kernel_value = kernel_value.reshape([N, K])
    return kernel_value


def WangRyzin_Convolution(h, Xi, Xj):
    # This is the equivalent of the convolution case with the Gaussian Kernel
    # However it is not exactly convolution. Think of a better name
    # References http://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&ved=0CGUQFjAD&url=http%3A%2F%2Feconweb.tamu.edu%2Fli%2FUncond1.pdf&ei=2THUT5i7IIOu8QSvmrXlAw&usg=AFQjCNH4aGzQbKT22sLBbZqHtPOyeFXNIQ
    Xi = np.abs(np.asarray(Xi, dtype=int))
    Xj = np.abs(np.asarray(Xj, dtype=int))
    h = np.asarray(h, dtype=float)
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
        N = np.shape(Xi)[0]
    elif Xi.ndim == 1:
        K = 1
        N = np.shape(Xi)[0]
    else:  # ndim ==0 so Xi is a single point (number)
        K = 1
        N = 1
    if K == 0:
        return Xi
    Xi = Xi.reshape([N, K])
    Xj = Xj.reshape((K, ))
    h = h.reshape((K, ))
    Dom_x = [np.unique(Xi[:, i]) for i in range(K)]
    Ordered = np.empty([N, K])
    for i in range(K):
        Sigma_x = 0
        # TODO: Think about vectorizing this for optimal performance
        for x in Dom_x[i]:
            Sigma_x += WangRyzin(h[i], Xi[:, i],
                                  int(x)) * WangRyzin(h[i], Xj[i], int(x))
            Ordered[:, i] = Sigma_x[:, 0]
    return Ordered


def AitchisonAitken_Convolution(h, Xi, Xj):
    Xi = np.abs(np.asarray(Xi, dtype=int))
    Xj = np.abs(np.asarray(Xj, dtype=int))
    h = np.asarray(h, dtype=float)
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
        N = np.shape(Xi)[0]
    elif Xi.ndim == 1:
        K = 1
        N = np.shape(Xi)[0]
    else:  # ndim ==0 so Xi is a single point (number)
        K = 1
        N = 1   
    if K == 0:
        return Xi
    Xi = Xi.reshape([N, K])
#    Xj = Xj.reshape((K, ))
    h = h.reshape((K, ))
    Dom_x = [np.unique(Xi[:, i]) for i in range(K)]
    Ordered = np.empty([N, K])
    for i in range(K):
        Sigma_x = 0
        # TODO: This can be vectorized
        for x in Dom_x[i]:
            Sigma_x += AitchisonAitken (h[i], Xi[:, i], int(x),
                                        num_levels=len(Dom_x[i])) * AitchisonAitken(h[i], Xj[i], int(x), num_levels=len(Dom_x[i]))       
        Ordered[:, i] = Sigma_x[:, 0]
    return Ordered
