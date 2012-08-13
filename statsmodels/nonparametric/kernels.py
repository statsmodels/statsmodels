"""
Module of kernels that are able to handle continuous as well as categorical
variables (both ordered and unordered).

This is a slight deviation from the current approach in
statsmodels.nonparametric.kernels where each kernel is a class object.

Having kernel functions rather than classes makes extension to a multivariate
kernel density estimation much easier.

NOTE: As it is, this module does not interact with the existing API
"""

import numpy as np
from scipy.special import erf


def _get_shape_and_transform(h, Xi, x=None):
    """
    Utility function to transform arrays and check shape parameters.
    """
    x = np.asarray(x)
    h = np.asarray(h, dtype=float)
    Xi = np.asarray(Xi)
    # More than one variable with more than one observations
    if Xi.ndim > 1:
        K = np.shape(Xi)[1]
        N = np.shape(Xi)[0]
    elif Xi.ndim == 1:  # One variable with many observations
        K = 1
        N = np.shape(Xi)[0]
    else:  # ndim ==0 so Xi is a single point (number)
        K = 1
        N = 1

    #assert N >= K  # Need more observations than variables
    Xi = Xi.reshape([N, K])
    return h, Xi, x, N, K


def AitchisonAitken(h, Xi, x, num_levels=False):
    """
    The Aitchison-Aitken kernel, used for unordered discrete random variables.

    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 2-D ndarray, shape (N, K)
        The value of the training set.
    x: 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.
    num_levels: bool, optional
        Gives the user the option to specify the number of levels for the
        random variable.  If False, the number of levels is calculated from
        the data.

    Returns
    -------
    kernel_value : ndarray, shape (N, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p.18 of [2]_ for details.  The value of the kernel L if :math:`X_{i}=x`
    is :math:`1-\lambda`, otherwise it is :math:`\frac{\lambda}{c-1}`.
    Here :math:`c` is the number of levels plus one of the RV.

    References
    ----------
    .. [1] J. Aitchison and C.G.G. Aitken, "Multivariate binary discrimination
           by the kernel method", Biometrika, vol. 63, pp. 413-420, 1976.
    .. [2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
    """
    h, Xi, x, N, K = _get_shape_and_transform(h, Xi, x)
    Xi = np.abs(np.asarray(Xi, dtype=int))
    x = np.abs(np.asarray(x, dtype=int))
    if K == 0:
        return Xi

    c = np.asarray([len(np.unique(Xi[:, i])) for i in range(K)], dtype=int)
    if num_levels:
        c = num_levels

    kernel_value = np.tile(h / (c - 1), (N, 1))
    inDom = (Xi == x) * (1 - h)
    kernel_value[Xi == x] = inDom[Xi == x]
    kernel_value = kernel_value.reshape([N, K])
    return kernel_value


def WangRyzin(h, Xi, x):
    """
    The Wang-Ryzin kernel, used for ordered discrete random variables.

    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (N, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p. 19 in [1]_ for details.  The value of the kernel L if
    :math:`X_{i}=x` is :math:`1-\lambda`, otherwise it is
    :math:`\frac{1-\lambda}{2}\lambda^{|X_{i}-x|}`.

    References
    ----------
    .. [1] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
           http://dx.doi.org/10.1561/0800000009
    .. [2] M.-C. Wang and J. van Ryzin, "A class of smooth estimators for
           discrete distributions", Biometrika, vol. 68, pp. 301-309, 1981.
    """
    h, Xi, x, N, K = _get_shape_and_transform(h, Xi, x)
    Xi = np.abs(np.asarray(Xi, dtype=int))
    x = np.abs(np.asarray(x, dtype=int))
    if K == 0:
        return Xi

    kernel_value = (0.5 * (1 - h) * (h ** abs(Xi - x)))
    kernel_value = kernel_value.reshape([N, K])
    inDom = (Xi == x) * (1 - h)
    kernel_value[Xi == x] = inDom[Xi == x]
    return kernel_value


def Gaussian(h, Xi, x):
    """
    Gaussian Kernel for continuous variables
    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (N, K)
        The value of the kernel function at each training point for each var.

    """
    h, Xi, x, N, K = _get_shape_and_transform(h, Xi, x)
    if K == 0:
        return Xi

    z = (Xi - x) / h
    kernel_value = (1. / np.sqrt(2 * np.pi)) * np.exp(- z ** 2 / 2.)
    kernel_value = kernel_value.reshape([N, K])
    return kernel_value


def Gaussian_Convolution(h, Xi, x):
    """ Calculates the Gaussian Convolution Kernel """
    h, Xi, x, N, K = _get_shape_and_transform(h, Xi, x)
    if K == 0:
        return Xi

    z = (Xi - x) / h
    kernel_value = (1. / np.sqrt(4 * np.pi)) * np.exp(- z ** 2 / 4.)
    kernel_value = kernel_value.reshape([N, K])
    return kernel_value


def WangRyzin_Convolution(h, Xi, Xj):
    # This is the equivalent of the convolution case
    # with the Gaussian Kernel
    # However it is not exactly convolution. Think of a better name
    # References
    h, Xi, x, N, K = _get_shape_and_transform(h, Xi)
    Xi = np.abs(np.asarray(Xi, dtype=int))
    Xj = np.abs(np.asarray(Xj, dtype=int))
    if K == 0:
        return Xi

    Xi = Xi.reshape([N, K])
    Xj = Xj.reshape((K, ))
    h = h.reshape((K, ))
    Dom_x = [np.unique(Xi[:, i]) for i in range(K)]
    Ordered = np.empty([N, K])
    for i in range(K):
        Sigma_x = 0
        for x in Dom_x[i]:
            Sigma_x += WangRyzin(h[i], Xi[:, i],
                                 int(x)) * WangRyzin(h[i], Xj[i], int(x))

        Ordered[:, i] = Sigma_x[:, 0]

    return Ordered


def AitchisonAitken_Convolution(h, Xi, Xj):
    h, Xi, x, N, K = _get_shape_and_transform(h, Xi)
    Xi = np.abs(np.asarray(Xi, dtype=int))
    Xj = np.abs(np.asarray(Xj, dtype=int))
    if K == 0:
        return Xi

    Xi = Xi.reshape([N, K])
    h = h.reshape((K, ))
    Dom_x = [np.unique(Xi[:, i]) for i in range(K)]
    Ordered = np.empty([N, K])
    for i in range(K):
        Sigma_x = 0
        for x in Dom_x[i]:
            Sigma_x += AitchisonAitken(h[i], Xi[:, i], int(x),
                                       num_levels=len(Dom_x[i])) * \
            AitchisonAitken(h[i], Xj[i], int(x), num_levels=len(Dom_x[i]))

        Ordered[:, i] = Sigma_x[:, 0]

    return Ordered


def Gaussian_cdf(h, Xi, x):
    h, Xi, x, N, K = _get_shape_and_transform(h, Xi, x)
    if K == 0:
        return Xi

    cdf = 0.5 * h * (1 + erf((x - Xi) / (h * np.sqrt(2))))
    cdf = cdf.reshape([N, K])
    return cdf


def AitchisonAitken_cdf(h, Xi, x_u):
    Xi = np.abs(np.asarray(Xi, dtype=int))
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
    Dom_x = [np.unique(Xi[:, i]) for i in range(K)]
    Ordered = np.empty([N, K])
    for i in range(K):
        Sigma_x = 0
        for x in Dom_x[i]:
            if x <= x_u:
                Sigma_x += AitchisonAitken(h[i], Xi[:, i], int(x),
                                           num_levels=len(Dom_x[i]))

        Ordered[:, i] = Sigma_x[:, 0]

    return Ordered


def WangRyzin_cdf(h, Xi, x_u):
    Xi = np.abs(np.asarray(Xi, dtype=int))
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
    h = h.reshape((K, ))
    Dom_x = [np.unique(Xi[:, i]) for i in range(K)]
    Ordered = np.empty([N, K])
    for i in range(K):
        Sigma_x = 0
        for x in Dom_x[i]:
            if x <= x_u:
                Sigma_x += WangRyzin(h[i], Xi[:, i], int(x))

        Ordered[:, i] = Sigma_x[:, 0]
    return Ordered

def D_Gaussian(h, Xi, x):
    # The derivative of the Gaussian Kernel
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
    value = np.exp(-z ** 2 / 2.) * (Xi - x) / (np.sqrt(2 * np.pi) * h ** 2)
    value = 2 * ( x - Xi) * Gaussian(h, Xi, x) / (h ** 2)
    return value
