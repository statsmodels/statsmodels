"""
Holds files for l1 regularization of LikelihoodModel, using
scipy.optimize.fmin  (Nelder-Mead)
"""
import numpy as np
from scipy.optimize import fmin
import pdb
# pdb.set_trace


def _fit_l1_nm(
        f, score, start_params, args, kwargs, disp=None, maxiter=100,
        callback=None, retall=False, full_output=False, hess=None):
    """
    Solve the l1 regularized problem using scipy.optimize.fmin().

    Parameters
    ----------
    All the usual parameters from LikelhoodModel.fit

    Special kwargs
    ------------------
    alpha : non-negative scalar or numpy array (same size as parameters)
        The weight multiplying the l1 penalty term
    trim_params : boolean (default True)
        Set small parameters to zero
    trim_tol : float or 'auto' (default = 'auto')
        If auto, trim params based on the optimality condition
        If float, trim params whose absolute value < trim_tol to zero
    xtol : float (default = 1e-4)
        Tolerance for the parameters
    ftol : float (default = 1e-4)
        Tolerance for the objective function
    """

    if callback:
        print "Callback will be ignored with l1"
    if hess:
        print "Hessian not used with l1"
    start_params = np.array(start_params).ravel('F')

    # TODO fargs should be passed to f, another name for all the args
    ### Extract values
    # K is total number of covariates, possibly including a leading constant.
    K = len(start_params)
    # The start point
    x0 = start_params
    # alpha is the regularization parameter
    alpha = np.array(kwargs['alpha']).ravel('F')
    # Make sure it's a vector
    alpha = alpha * np.ones(K)
    assert alpha.min() >= 0
    # Set/retrieve the desired accuracy
    xtol = kwargs.setdefault('xtol', 1e-4)
    ftol = kwargs.setdefault('ftol', 1e-4)

    ### Wrap up for use in fmin_slsqp
    # TODO Fix func, then run with debugger and see
    func = lambda x: objective_func(f, x, alpha, *args)

    ### Call the optimization
    results = fmin(func, x0, xtol=xtol, ftol=ftol, maxiter=maxiter,
        maxfun=None, full_output=True, disp=disp, retall=retall,
        callback=None)
    # Unpack
    if retall:
        xopt, fopt, its, funcalls, warnflag, allvecs = results
        print allvecs
    else:
        xopt, fopt, its, funcalls, warnflag = results
    if warnflag == 1:
        warn_msg = 'Maximum function evaluations reached'
    elif warnflag == 2:
        warn_msg = 'Maximum iterations reached'
    xopt = np.array(xopt)

    ### Post-process
    trim_params = kwargs.setdefault('trim_params', True)
    if trim_params:
        trim_tol = kwargs.setdefault('trim_tol', 'auto')
        xopt, trimmed = do_trim_params(xopt, alpha, trim_tol, score)

    ### Pack up return values for statsmodels optimizers
    # TODO These retvals are returned as mle_retvals...but the fit wasn't ML
    if full_output:
        fopt = func(xopt)
        converged = 'True' if warnflag == 0 else warn_msg
        iterations = its
        gopt = float('nan')     # Objective is non-differentiable
        hopt = float('nan')
        retvals = {
            'fopt': fopt, 'converged': converged, 'iterations': iterations,
            'gopt': gopt, 'hopt': hopt, 'trimmed': trimmed}

    ### Return results
    if full_output:
        return xopt, retvals
    else:
        return xopt


def do_trim_params(x, alpha, trim_tol, score):
    """
    If trim_tol == 'auto', then trim params if the derivative in that direction
        is significantly smaller than alpha.  Theory says the nonzero params
        should have magnitude equal to alpha at a minimum.

    If trim_tol is a float, then trim (set = 0) params that are within trim_tol
        of zero.  

    In all cases, if alpha[i] == 0, then don't trim the ith param.  
    In all cases, do nothing with the dummy variables.
    """
    ## Trim the small params
    K = len(x)
    trimmed = [False] * K  
    # Don't bother triming the dummy variables 'u'
    if trim_tol == 'auto':
        fprime = score(np.array(x))
        for i in xrange(K):
            if alpha[i] != 0:
                # TODO Magic number !!
                magic_tol = 0.03
                if alpha[i] - abs(fprime[i]) > magic_tol:
                    x[i] = 0.0
                    trimmed[i] = True
                # If fprime is too big, then we didn't converge properly 
                # and we shouldn't trust the automatic trimming
                elif alpha[i] - abs(fprime[i]) < -magic_tol:
                    raise Exception(
                        "Unable to trim params automatically with "\
                        "this low optimization accuracy")
    else:
        for i in xrange(K):
            if alpha[i] != 0:
                if abs(x[i]) < trim_tol:
                    x[i] = 0.0
                    trimmed[i] = True
    return x, np.array(trimmed)


def objective_func(f, x, alpha, *args):
    """
    The regularized objective function
    """
    ## Return
    return f(x, *args) + (np.fabs(alpha * x)).sum()


