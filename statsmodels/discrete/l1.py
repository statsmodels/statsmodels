"""
Holds files for l1 regularization of LikelihoodModel
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import pdb
# pdb.set_trace


def _fit_l1_slsqp(f, score, start_params, fargs, kwargs, disp=None, 
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
        print "Hessian not used with l1, since l1 uses fmin_slsqp"

    # TODO fargs should be passed to f, another name for all the args
    ### Extract values
    fargs += (f,score) 
    # K is total number of covariates, possibly including a leading constant.
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

def _fit_l1_cvxopt_cp(f, score, start_params, fargs, kwargs, disp=None, 
        maxiter=100, callback=None, retall=False, full_output=False, hess=None
        ):
    """
    Solve the l1 regularized problem using cvxopt.solvers.cp

    Cut and paste docstring from _fit_l1_slsqp???
    """
    from cvxopt import solvers, matrix

    if callback:
        print "Callback will be ignored with l1"
    if hess: 
        print "Hessian not used with l1 methods"

    ### Extract arguments
    fargs += (f,score) 
    # K is total number of covariates, possibly including a leading constant.
    K = len(start_params)  
    fargs += (K,)
    # The regularization parameter
    alpha = np.array(kwargs['alpha']).ravel(order='F')
    fargs += (alpha,)
    # The start point
    x0 = np.append(start_params, np.fabs(start_params))
    x0 = matrix(x0, (2*K, 1))
    # Wrap up functions to be used by cvxopt
    f_0 = lambda x : matrix(func(np.array(x), *fargs))
    Df = lambda x : matrix(fprime(np.array(x), *fargs), (1, 2*K))
    def H(x,z):
        zh_x = np.array(z[0]) * hess(np.array(x))
        zero_mat = np.zeros(zh_x.shape)
        A = np.concatenate((zh_x, zero_mat), axis=1)
        B = np.concatenate((zero_mat, zero_mat), axis=1)
        zh_x_ext = np.concatenate((A,B), axis=0)
        return matrix(zh_x_ext, (2*K, 2*K))
    G = matrix(-1*fprime_ieqcons(x0, *fargs))
    h = matrix(0.0, (2*K, 1))
    ## Define the optimization function
    def F(x=None, z=None):
        if x is None: 
            return 0, x0
        elif z is None: 
            return f_0(x), Df(x)
        else:
            return f_0(x), Df(x), H(x,z)

    ## Convert optimization settings to cvxopt form
    solvers.options['show_progress'] = retall
    solvers.options['maxiters'] = maxiter
    if 'abstol' in kwargs:
        solvers.options['abstol'] = kwargs['abstol']
    if 'reltol' in kwargs:
        solvers.options['reltol'] = kwargs['reltol']
    if 'feastol' in kwargs:
        solvers.options['feastol'] = kwargs['feastol']
    if 'refinement' in kwargs:
        solvers.options['refinement'] = kwargs['refinement']

    ### Call the optimization
    results = solvers.cp(F, G, h)

    ### Post-process 
    #QA_results(x, params, K, acc) 
    #if kwargs.get('trim_params'):
    #    results = trim_params(results, full_output, K, func, fargs, acc, alpha)

    ### Pack up return values for statsmodels optimizers
    if full_output:
        x = np.array(results['x'])
        params = x[:K]
        # TODO How is fopt used?  Is it expected to be neg-log-like?
        # TODO Possibly get rid of gopt, hopt altogether?
        args = fargs[:-4]
        fopt = f(params, *args)
        gopt = score(params)
        hopt = hess(params)
        iterations = float('nan')
        converged = 'True' if results['status'] == 'optimal' else results['status']
        retvals = {'fopt':fopt, 'converged':converged, 'iterations':iterations, 
                'gopt':gopt, 'hopt':hopt}
    else:
        x = np.array(results['x'])
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
    return np.append(score(params), alpha * np.ones(K))

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
    args = fargs[:-4]
    f, score, K, alpha = fargs[-4:]
    params = x[:K]
    nonconst_params = x[:K]
    u = x[K:]

    I = np.eye(K)
    A = np.concatenate((I,I), axis=1)
    B = np.concatenate((-I,I), axis=1)
    C = np.concatenate((A,B), axis=0)

    return C
