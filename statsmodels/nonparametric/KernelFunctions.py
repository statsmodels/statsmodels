# This module adds to kernels that are able to handle
# categorical variables (both ordered and unordered)


# In addition, this module contains kernel functions
# This is a slight deviation from the current approach in statsmodels.
# nonparametric.kernels where each kernel is a class object

# Having kernel functions rather than classes makes extension to a multivariate
# kernel density estimation much easier

# NOTE: As it is, this module does not interact with the existing API

import numpy as np

def AitchisonAitken(h,Xi,x,c):
    """
    Returns the value of the Aitchison and Aitken's (1976) Kernel. Used for
    unordered discrete random variable

    Parameters
    ----------
    h : float
        The bandwidth used to estimate the value of the kernel function
    Xi : array like
        The the training set
    x: int
        The value at which the kernel density is being estimated
    c: int
        The number of possible values that the variable can take
    
    Returns
    -------
    kernel_value : array like float
        The value of the kernel function
        
    References
    ----------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
    
    """
    Xi=np.asarray(Xi)
    kernel_value = np.empty(Xi.shape)
    kernel_value.fill(h/(c-1))
    kernel_value[Xi==x] = 1-h
    
    return kernel_value



def WangRyzin (h, Xi, x, c):
    """
    Returns the value of the Wang and van Ryzin's (1981) Kernel. Used for
    ordered discrete random variable
    
    Parameters
    ----------
    h : float
        The bandwidth used to estimate the value of the kernel function
    Xi : int
        The value of a training point
    x: int
        The value at which the kernel density is being estimated
    c: int
        The number of possible values that the variable can take
    
    Returns
    -------
    kernel_value : float
        The value of the kernel function
        
    References
    ----------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
    
    """
    h=float(h)
    Xi=np.asarray(Xi)
    kernel_value = np.empty(Xi.shape)
    kernel_value.fill(0.5*(1-h)*h**abs(Xi-x))
    kernel_value[Xi==x] = 1-h
    return kernel_value

def Epanechnikov (h, Xi, x):

    """
    Returns the value of the Epanechnikov (1969) Kernel. Used for
    continuous random variables
    
    Parameters
    ----------
    h : float
        The bandwidth used to estimate the value of the kernel function
    Xi : array float
        The value of a training point
    x: int
        The value at which the kernel density is being estimated
    Returns
    -------
    kernel_value : float
        The value of the kernel function
        
    References
    ----------
    Racine, J. "Nonparametric econometrics: A Primer" : Foundations and Trends in Econometrics.
    
    """
    Xi=np.asarray(Xi)
    h = float(h)
    n = len(Xi)
    z = (Xi - x)/h
    InDom = (-np.sqrt(5) <= z) * (z <= np.sqrt(5))
    kernel_value = InDom*(0.75*(1-0.2*z**2)*(1/np.sqrt(5)))

    # NOTE: There is a slight discrepancy between this formulation of the Epanechnikov kernel and
    #       the one in kernels.py.
    # TODO: Check Silverman and reconcile the discrepancy

    return kernel_value
