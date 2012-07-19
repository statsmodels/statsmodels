"""
Holds files for the lasso regularization
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import pdb
# pdb.set_trace


def _fit_lasso(f, score, start_params, fargs, kwargs, disp=None, maxiter=100, 
        callback=None, retall=False, full_output=False, hess=None):
    """
    """

    if callback:
        print "Callback will be ignored with lasso"

    ### Extract values
    fargs += (f,score) 
    # P is total number of covariates, possibly including a leading constant.
    K = len(start_params)  
    fargs += (K,)
    # offset determines which parts of x are used for the dummy variables 'u'
    constant = kwargs.setdefault('constant', False)
    offset = 1 if constant else 0
    fargs += (offset,)
    # The start point
    x0 = np.append(start_params, np.fabs(start_params[offset:]))
    # alpha is the regularization parameter
    alpha = kwargs['alpha']
    fargs += (alpha,)
    # Epsilon is used for approximating derivatives
    epsilon = kwargs.setdefault('epsilon', 1.49e-8)
    # Convert display parameters to scipy.optimize form
    if disp or retall:
        if disp:
            disp_slsqp = 1
        if retall:
            disp_slsqp = 2
    else:
        disp_slsqp = 0

    acc = kwargs.setdefault('acc', 1e-6)
    results = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons, fprime=fprime, acc=acc,
            args=fargs, iter=maxiter, disp=disp_slsqp, full_output=full_output, 
            fprime_ieqcons=fprime_ieqcons, epsilon=epsilon)


    ### Pack up return values for statsmodels optimizers
    if full_output:
        x, fx, its, imode, smode = results
        x = np.array(x)
        params = x[:K]
        fopt = fx
        converged = smode
        iterations = its
        # TODO Should gopt be changed to accomidate the regularization term?
        gopt = score(params)  
        hopt = hess(params)
        retvals = {'fopt':fopt, 'converged':converged, 'iterations':iterations, 
                'gopt':gopt, 'hopt':hopt}
    else:
        x = np.array(results)
        params = x[:K]

    ### Post-process 
    QA_results(x, params, K, constant, acc) 
    if kwargs.get('trim_params'):
        trim_params(params, acc, constant)

    if full_output:
        return params, retvals
    else:
        return params

def QA_results(x, params, K, constant, acc):
    """
    Raises exception if:
        The dummy variables u are not equal to absolute value of params to within
        min(-log10(10*acc), 10) decimal places
    """
    u = x[K:]
    decimal = min(int(-np.log10(10*acc)), 10)
    offset = 1 if constant else 0
    np.testing.assert_array_almost_equal(np.fabs(params[offset:]), u, decimal=decimal)

def trim_params(params, acc, constant):
    """
    Trims params that are within max(10*acc, 1e-10) of zero.
    """
    trim_tol = max(10*acc, 1e-10)
    small_param_idx = params < trim_tol
    # If we have a constant column, then don't trim the constant param,
    # since this param was not meant to be regularized.
    if constant:
        small_param_idx[0] = False
    params[params < trim_tol] = 0.0

def xxx_fit_lasso(f, score, start_params, fargs, kwargs, disp=None, maxiter=100, 
        callback=None, retall=False, full_output=False, hess=None):
    """
    """

    if callback:
        print "Callback will be ignored with lasso"

    #### Extract values
    ## The start point
    #x0 = np.append(start_params, np.fabs(start_params[offset:]))
    #fargs += (f,score) 
    ## P is total number of covariates, possibly including a leading constant.
    #K = len(start_params)  
    #fargs += (K,)
    ## offset determines which parts of x are used for the dummy variables 'u'
    #offset = 0
    #if 'constant' in kwargs:
    #    if kwargs['constant']:
    #        offset = 1
    #fargs += (offset,)
    ## alpha is the regularization parameter
    #alpha = kwargs['alpha']
    #fargs += (alpha,)
    ## Epsilon is used for approximating derivatives
    #epsilon = kwargs.setdefault('epsilon', None)
    ## Convert display parameters to scipy.optimize form
    if disp or retall:
        if disp:
            disp_slsqp = 1
        if retall:
            disp_slsqp = 2
    else:
        disp_slsqp = 0

    #results = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons, fprime=fprime, 
    #        args=fargs, iter=maxiter, disp=disp_slsqp, full_output=full_output, 
    #        epsilon=epsilon)
    # TODO Added tol
    tol = 1e-8
    epsilon = 1.49e-8
    results = fmin_slsqp(f, start_params, fprime=score, acc=tol, epsilon=epsilon,
            args=fargs, iter=maxiter, disp=disp_slsqp, full_output=full_output)

    ### The return values for statsmodels optimizers
    if full_output:
        x, fx, its, imode, smode = results
        #xopt = np.array(x[:K])
        xopt = np.array(x)
        fopt = fx
        converged = smode
        iterations = its
        # TODO Should gopt be changed to accomidate the regularization term?
        gopt = score(xopt)  
        hopt = hess(xopt)
        retvals = {'fopt':fopt, 'converged':converged, 'iterations':iterations, 
                'gopt':gopt, 'hopt':hopt}
        return xopt, retvals
    else:
        xopt = np.array(results)
        return xopt

def func(x, *fargs):
    """
    The regularized objective function
    """
    args = fargs[:-5]
    f, score, K, offset, alpha = fargs[-5:]
    params = x[:K]
    nonconst_params = x[offset:K]
    u = x[K:]

    return f(params, *args) + alpha * u.sum()

def fprime(x, *fargs):
    """
    The regularized derivative
    """
    args = fargs[:-5]
    f, score, K, offset, alpha = fargs[-5:]
    params = x[:K]
    nonconst_params = x[offset:K]
    u = x[K:]
    # The derivative just appends a vector of constants
    return np.append(score(params, *args), alpha * np.ones(K-offset))

def f_ieqcons(x, *fargs):
    args = fargs[:-5]
    f, score, K, offset, alpha = fargs[-5:]
    params = x[:K]
    nonconst_params = x[offset:K]
    u = x[K:]
    # All entries in this vector must be \geq 0 in a feasible solution
    return np.append(nonconst_params + u, u - nonconst_params)

def fprime_ieqcons(x, *fargs):
    args = fargs[:-5]
    f, score, K, offset, alpha = fargs[-5:]
    params = x[:K]
    nonconst_params = x[offset:K]
    u = x[K:]

    I = np.eye(K-offset)
    A = np.concatenate((I,I), axis=1)
    B = np.concatenate((-I,I), axis=1)
    C = np.concatenate((A,B), axis=0)

    if offset == 0:
        return C
    elif offset == 1:
        one_column = np.zeros((2*K-2, 1))
        return np.concatenate((one_column, C), axis=1)
