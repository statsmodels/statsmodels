"""
Holds files for l1 regularization of LikelihoodModel
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import pdb
# pdb.set_trace


def _fit_l1(f, score, start_params, fargs, kwargs, disp=None, maxiter=100, 
        callback=None, retall=False, full_output=False, hess=None):
    """
    Called by base.LikelihoodModel.fit.  Call structure is similar to e.g. 
        _fit_mle_newton().

    The optimization is done by scipy.optimize.fmin_slsqp().  With :math:`L` 
    the log-likelihood, we solve the convex but non-smooth problem
    .. math:: \\min_\\beta -\\ln L + \\alpha \\sum_k|\\beta_k|
    via the transformation to the smooth, convex, constrained problem in twice
    as many variables
    .. math:: \\min_{\\beta,u} -\\ln L + \\alpha \\sum_ku_k,
    subject to
    .. math:: -u_k \\leq \\beta_k \\leq u_k.

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
        print "Hessian not used with l1, since l1 uses fmin_slsqp"

    ### Extract values
    fargs += (f,score) 
    # P is total number of covariates, possibly including a leading constant.
    K = len(start_params)  
    fargs += (K,)
    # The start point
    x0 = np.append(start_params, np.fabs(start_params))
    # alpha is the regularization parameter
    alpha = np.array(kwargs['alpha']).ravel(order='F')
    fargs += (alpha,)
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

    ### Call the optimization
    results = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons, fprime=fprime, acc=acc,
            args=fargs, iter=maxiter, disp=disp_slsqp, full_output=full_output, 
            fprime_ieqcons=fprime_ieqcons)

    ### Post-process 
    #QA_results(x, params, K, acc) 
    if kwargs.get('trim_params'):
        results = trim_params(results, full_output, K, func, fargs, acc, alpha)

    ### Pack up return values for statsmodels optimizers
    if full_output:
        x, fx, its, imode, smode = results
        x = np.array(x)
        params = x[:K]
        fopt = fx
        converged = 'True' if imode == 0 else smode
        iterations = its
        # TODO Possibly get rid of gopt, hopt altogether?
        gopt = score(params)  
        hopt = hess(params)
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

def QA_results(x, params, K, acc):
    """
    Raises exception if:
        The dummy variables u are not equal to absolute value of params to within
        min(-log10(10*acc), 10) decimal places
    """
    u = x[K:]
    decimal = min(int(-np.log10(10*acc)), 10)
    abs_params = np.fabs(params)
    try:
        np.testing.assert_array_almost_equal(abs_params, u, decimal=decimal)
    except AssertionError:
        print "abs_params = \n%s\nu = %s"%(abs_params, u)
        raise

def trim_params(results, full_output, K, func, fargs, acc, alpha):
    """
    Trims (sets = 0) params that are within min(max(10*acc, 1e-10), 1e-3)
    of zero.  If alpha[i] == 0, then don't trim the ith param.
    """
    ## Extract params from the results
    trim_tol = min(max(100*acc, 1e-10), 1e-3)
    if full_output:
        x, fx, its, imode, smode = results
    else:
        x = results
    ## Trim the small params
    # Don't bother triming the dummy variables 'u'
    # TODO Vectorize this
    for i in xrange(len(x) / 2):
        if abs(x[i]) < trim_tol and alpha[i] != 0:
            x[i] = 0.0
    ## Recompute things
    if full_output:
        fx = func(np.array(x), *fargs)
        return x, fx, its, imode, smode
    else:
        return x

def func(x, *fargs):
    """
    The regularized objective function
    """
    args = fargs[:-4]
    f, score, K, alpha = fargs[-4:]
    params = x[:K]
    nonconst_params = x[:K]
    u = x[K:]

    return f(params, *args) + (alpha * u).sum()

def fprime(x, *fargs):
    """
    The regularized derivative
    """
    args = fargs[:-4]
    f, score, K, alpha = fargs[-4:]
    params = x[:K]
    nonconst_params = x[:K]
    u = x[K:]
    # The derivative just appends a vector of constants
    return np.append(score(params, *args), alpha * np.ones(K))

def f_ieqcons(x, *fargs):
    """
    The inequality constraints.
    """
    args = fargs[:-4]
    f, score, K, alpha = fargs[-4:]
    params = x[:K]
    nonconst_params = x[:K]
    u = x[K:]
    # All entries in this vector must be \geq 0 in a feasible solution
    return np.append(nonconst_params + u, u - nonconst_params)

def fprime_ieqcons(x, *fargs):
    """
    Derivative of the inequality constraints
    """
    args = fargs[:-5]
    f, score, K, alpha = fargs[-4:]
    params = x[:K]
    nonconst_params = x[:K]
    u = x[K:]

    I = np.eye(K)
    A = np.concatenate((I,I), axis=1)
    B = np.concatenate((-I,I), axis=1)
    C = np.concatenate((A,B), axis=0)

    return C
