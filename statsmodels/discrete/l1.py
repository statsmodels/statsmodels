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
    alpha : Float.  The regularization parameter.
    do_not_regularize_first_param : Boolean.  If the first covariate is a 
        constant, then you almost never regularize it.  
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
    # offset determines which parts of x are used for the dummy variables 'u'
    do_not_regularize_first_param = kwargs.setdefault(
            'do_not_regularize_first_param', False)
    offset = 1 if do_not_regularize_first_param else 0
    fargs += (offset,)
    # The start point
    x0 = np.append(start_params, np.fabs(start_params[offset:]))
    # alpha is the regularization parameter
    alpha = kwargs['alpha']
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
    #QA_results(x, params, K, do_not_regularize_first_param, acc) 
    if kwargs.get('trim_params'):
        results = trim_params(results, full_output, K, func, fargs, acc, offset)

    ### Pack up return values for statsmodels optimizers
    if full_output:
        x, fx, its, imode, smode = results
        x = np.array(x)
        params = x[:K]
        fopt = fx
        converged = 'True' if imode == 0 else smode
        iterations = its
        # TODO Should gopt be changed to accomidate the regularization term?
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

def QA_results(x, params, K, do_not_regularize_first_param, acc):
    """
    Raises exception if:
        The dummy variables u are not equal to absolute value of params to within
        min(-log10(10*acc), 10) decimal places
    """
    u = x[K:]
    decimal = min(int(-np.log10(10*acc)), 10)
    offset = 1 if do_not_regularize_first_param else 0
    abs_params = np.fabs(params[offset:])
    try:
        np.testing.assert_array_almost_equal(abs_params, u, decimal=decimal)
    except AssertionError:
        print "abs_params = \n%s\nu = %s"%(abs_params, u)
        raise

def trim_params(results, full_output, K, func, fargs, acc, offset):
    """
    Trims (sets = 0) params that are within max(10*acc, 1e-10) of zero.
    """
    ## Extract params from the results
    trim_tol = min(max(100*acc, 1e-10), 1e-3)
    if full_output:
        x, fx, its, imode, smode = results
    else:
        x = results
    ## Trim the small params
    # If we have a constant column, then don't trim the constant param,
    # since this param was not meant to be regularized.
    for i in xrange(offset, len(x)):
        if abs(x[i]) < trim_tol:
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
    """
    The inequality constraints.
    """
    args = fargs[:-5]
    f, score, K, offset, alpha = fargs[-5:]
    params = x[:K]
    nonconst_params = x[offset:K]
    u = x[K:]
    # All entries in this vector must be \geq 0 in a feasible solution
    return np.append(nonconst_params + u, u - nonconst_params)

def fprime_ieqcons(x, *fargs):
    """
    Derivative of the inequality constraints
    """
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

def nnz_params(results):
    """
    Returns the number of nonzero parameters
    """
    return len(np.nonzero(np.fabs(results.params.ravel())>0)[0]) 

def modified_df_model(results, model):
    """
    TODO: Not sure if this is done right since I'm not sure what df_model 
        is supposed to be.

    In statsmodels, df_model is set first in 
        DiscreteModel.initialize() with  
            self.df_model = float(tools.rank(self.exog) - 1) 
        and then in MultinomialModel.__init__ with 
            self.df_model *= (self.J-1)
    """
    number_nonzero_params = nnz_params(results)
    df_model = number_nonzero_params - (model.J - 1)

    return df_model

def modified_bic(results, model):
    """
    Replacement for the "call" results.bic 
        (in discrete_model.py:MultinomialResults).  This version uses 
        modified_df_model rather than the built in df_model.
    """
    df_model = modified_df_model(results, model)
    return -2*results.llf + np.log(results.nobs)*(df_model+model.J-1)

def modified_aic(results, model):
    """
    Replacement for the "call" results.aic 
        (in discrete_model.py:MultinomialResults).  This version uses 
        modified_df_model rather than the built in df_model.
    """
    df_model = modified_df_model(results, model)
    return -2*(results.llf - (df_model+model.J-1))












