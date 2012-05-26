# This module adds to kernels that are able to handle
# categorical variables (both ordered and unordered)


# In addition, this module contains kernel functions
# This is a slight deviation from the current approach in statsmodels.
# nonparametric.kernels where each kernel is a class object

# Having kernel functions rather than classes makes extension to a multivariate
# kernel density estimation much easier

# NOTE: As it is, this module does not interact with the existing API

import numpy as np

def AitchisonAitken(h,Xi,x):
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
    c: 1D array of length K
        The number of possible values that each variable can take
    
    Returns
    -------
    kernel_value : K-dimensional array
        The value of the kernel function at each training point for each var
        
    References
    ----------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
    
    """
    Xi=np.asarray(Xi)
    N = np.shape(Xi)[0]
    if Xi.ndim>1:
        K=np.shape(Xi)[1]
    else:
        K=1

    if K==0: return Xi
    h = np.asarray(h, dtype = float)
    c=np.asarray([len(np.unique(Xi[:,i])) for i in range(K)],dtype=int)
    
    kernel_value = np.empty(Xi.shape)
    kernel_value.fill(h/(c-1))
    inDom=(Xi==x) *( 1-h)
    kernel_value[Xi==x]=inDom[Xi==x]
    kernel_value=kernel_value.reshape([N,K])
    
    
    return kernel_value



def WangRyzin (h, Xi, x):
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
        
    References
    ----------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
    
    """
    Xi=np.asarray(Xi,dtype=int)
    x=np.asarray(x,dtype=int)
    N = np.shape(Xi)[0]
    if Xi.ndim>1:
        K=np.shape(Xi)[1]
    else:
        K=1
    
    if K==0: return Xi
    
    h = np.asarray(h,dtype=float)
    
    Xi=Xi.reshape([N,K])
    
    kernel_value=(0.5*(1-h)*(h**abs(Xi-x)))
    
    inDom=(Xi==x) *( 1-h)
    kernel_value[Xi==x]=inDom[Xi==x]
    kernel_value=kernel_value.reshape([N,K])
    return kernel_value

def Epanechnikov (h, Xi, x):

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
    
    Xi=np.asarray(Xi)
    x=np.asarray(x)
    h = np.asarray(h,dtype=float)
    
    N = np.shape(Xi)[0]
    if Xi.ndim>1:
        K=np.shape(Xi)[1]
    else:
        K=1
    if K==0: return Xi
    
    z = (Xi - x)/h
    InDom = (-np.sqrt(5) <= z) * (z <= np.sqrt(5))
    kernel_value = InDom*(0.75*(1-0.2*z**2)*(1/np.sqrt(5)))
    kernel_value=kernel_value.reshape([N,K])

    # NOTE: There is a slight discrepancy between this formulation of the Epanechnikov kernel and
    #       the one in kernels.py.
    # TODO: Check Silverman and reconcile the discrepancy

    return kernel_value

def Gaussian(h,Xi,x):
    Xi=np.asarray(Xi)
    x=np.asarray(x)
    h = np.asarray(h,dtype=float)
    
    N = np.shape(Xi)[0]
    if Xi.ndim>1:
        K=np.shape(Xi)[1]
    else:
        K=1
    if K==0: return Xi
    z = (Xi - x)/h
    kernel_value=(1./np.sqrt(2*np.pi))*np.e**(-z**2/2.)
    kernel_value=kernel_value.reshape([N,K])
    return kernel_value
