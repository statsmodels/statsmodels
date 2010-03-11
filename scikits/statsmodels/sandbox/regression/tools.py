'''gradient/Jacobian of normal loglikelihood

use chain rule


A: josef-pktd

'''


import numpy as np


def norm_lls(y, params):
    '''normal loglikelihood given observations and mean mu and variance sigma2

    Parameters
    ----------
    y : array, 1d
        normally distributed random variable
    params: array, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows
    '''

    mu, sigma2 = params.T
    lls = -0.5*(np.log(2*np.pi) + np.log(sigma2) + (y-mu)**2/sigma2)
    return lls

def norm_lls_grad(y, params):
    '''Jacobian of normal loglikelihood wrt mean mu and variance sigma2

    Parameters
    ----------
    y : array, 1d
        normally distributed random variable
    params: array, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows

    Returns
    -------
    grad : array (nobs, 2)
        derivative of loglikelihood for each observation wrt mean in first
        column, and wrt variance in second column
    '''
    mu, sigma2 = params.T
    dllsdmu = (y-mu)/sigma2
    dllsdsigma2 = ((y-mu)**2/sigma2 - 1)/np.sqrt(sigma2)
    return np.column_stack((dllsdmu, dllsdsigma2))


def mean_grad(x, beta):
    '''gradient/Jacobian for d (x*beta)/ d beta
    '''
    return x

def normgrad(y, x, params):
    '''Jacobian of normal loglikelihood wrt mean mu and variance sigma2

    Parameters
    ----------
    y : array, 1d
        normally distributed random variable with mean x*beta, and variance sigma2
    x : array, 2d
        explanatory variables, observation in rows, variables in columns
    params: array_like, (nvars + 1)
        array of coefficients and variance (beta, sigma2)

    Returns
    -------
    grad : array (nobs, 2)
        derivative of loglikelihood for each observation wrt mean in first
        column, and wrt variance in second column
    assume params = (beta, sigma2)

    Notes
    -----
    TODO: for heteroscedasticity need sigma to be a 1d array

    '''
    beta = params[:-1]
    sigma2 = params[-1]*np.ones((len(y),1))
    dmudbeta = mean_grad(x, beta)
    mu = np.dot(x, beta)
    #print beta, sigma2
    params2 = np.column_stack((mu,sigma2)) #np.concatenate((beta, np.atleast_1d(sigma2)))
    dllsdms = norm_lls_grad(y,params2)
    grad = np.column_stack((dllsdms[:,:1]*dmudbeta, dllsdms[:,:1]))
    return grad

sig = 0.1
beta = np.ones(2)
rvs = np.random.randn(10,3)
x = rvs[:,1:]
y = np.dot(x,beta) + sig*rvs[:,0]

params = [1,1,0.1**2]
print normgrad(y, x, params)




