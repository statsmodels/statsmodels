"""
Holds files for l1 regularization of LikelihoodModel, using 
scipy.optimize.slsqp
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import pdb
# pdb.set_trace


def _fit_l1_slsqp(f, score, start_params, args, kwargs, disp=None, 
        maxiter=100, callback=None, retall=False, full_output=False, hess=None
        ):
    """
    Solve the l1 regularized problem using scipy.optimize.fmin_slsqp().  

    Specifically:  We solve the convex but non-smooth problem
    .. math:: \\min_\\beta f(\\beta) + \\sum_k\\alpha_k |\\beta_k|
    via the transformation to the smooth, convex, constrained problem in twice
    as many variables
    .. math:: \\min_{\\beta,u} f(\\beta) + \\sum_k\\alpha_k u_k,
    subject to
    .. math:: -u_k \\leq \\beta_k \\leq u_k.

    Parameters
    ----------
    Call structure is similar to e.g. 
        _fit_mle_newton().

    Special Parameters
    ------------------
    alpha : Float or array like.
        The regularization parameter.  If a float, then all covariates (even
        the constant) are regularized with alpha.  If array-like, then we use
        np.array(alpha).ravel(order='F').  

    TODO: alpha is currently passed through by calling function and reshaped
        here.  This is not consistent with the previous practice of having
        e.g. MNLogit reshape passed parameters.
    """

    if callback:
        print "Callback will be ignored with l1"
    if hess: 
        print "Hessian not used with l1"

    # TODO fargs should be passed to f, another name for all the args
    ### Extract values
    # K is total number of covariates, possibly including a leading constant.
    K = len(start_params)  
    # The start point
    x0 = np.append(start_params, np.fabs(start_params))
    # alpha is the regularization parameter
    alpha = np.array(kwargs['alpha']).ravel(order='F')
    # Convert display parameters to scipy.optimize form
    if disp or retall:
        if disp:
            disp_slsqp = 1
        if retall:
            disp_slsqp = 2
    else:
        disp_slsqp = 0
    # Set/retrieve the desired accuracy
    acc = kwargs.setdefault('acc', 1e-6)

    ### Wrap up for use in fmin_slsqp
    func = lambda x : objective_func(f, x, K, alpha, *args)
    f_ieqcons_wrap = lambda x : f_ieqcons(x, K)
    fprime_wrap = lambda x : fprime(score, x, K, alpha)
    fprime_ieqcons_wrap = lambda x : fprime_ieqcons(x, K)

    ### Call the optimization
    results = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons_wrap, 
            fprime=fprime_wrap, acc=acc, iter=maxiter, 
            disp=disp_slsqp, full_output=full_output, 
            fprime_ieqcons=fprime_ieqcons_wrap)

    ### Post-process 
    trim_tol = kwargs.setdefault('trim_tol', 1e-4)
    if kwargs.get('trim_params'):
        results = trim_params(results, full_output, K, alpha, trim_tol)

    ### Pack up return values for statsmodels optimizers
    # TODO These retvals are returned as mle_retvals...but the fit wasn't ML
    if full_output:
        x, fx, its, imode, smode = results
        x = np.array(x)
        params = x[:K]
        fopt = func(x)
        converged = 'True' if imode == 0 else smode
        iterations = its
        gopt = float('nan')     # Objective is non-differentiable
        hopt = float('nan') 
        retvals = {'fopt':fopt, 'converged':converged, 'iterations':iterations, 
                'gopt':gopt, 'hopt':hopt}
    else:
        x = np.array(results)
        params = x[:K]

    ### Return results
    if full_output:
        return params, retvals
    else:
        return params

def trim_params(results, full_output, K, alpha, trim_tol):
    """
    Trims (sets = 0) params that are within trim_tol of zero.  
    If alpha[i] == 0, then don't trim the ith param.
    """
    ## Extract x from the results
    if full_output:
        x, fx, its, imode, smode = results
    else:
        x = results
    ## Trim the small params
    # Don't bother triming the dummy variables 'u'
    for i in xrange(K):
        if abs(x[i]) < trim_tol and alpha[i] != 0:
            x[i] = 0.0
    ## Pack back up
    if full_output:
        return x, fx, its, imode, smode
    else:
        return x

def objective_func(f, x, K, alpha, *args):
    """
    The regularized objective function
    """
    params = x[:K]
    u = x[K:]
    ## Return
    return f(params, *args) + (alpha * u).sum()

def fprime(score, x, K, alpha):
    """
    The regularized derivative
    """
    params = x[:K]
    # The derivative just appends a vector of constants
    return np.append(score(params), alpha * np.ones(K))

def f_ieqcons(x, K):
    """
    The inequality constraints.
    """
    params = x[:K]
    nonconst_params = x[:K]
    u = x[K:]
    # All entries in this vector must be \geq 0 in a feasible solution
    return np.append(nonconst_params + u, u - nonconst_params)

def fprime_ieqcons(x, K):
    """
    Derivative of the inequality constraints
    """
    I = np.eye(K)
    A = np.concatenate((I,I), axis=1)
    B = np.concatenate((-I,I), axis=1)
    C = np.concatenate((A,B), axis=0)
    ## Return
    return C
