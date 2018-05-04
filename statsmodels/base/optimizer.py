"""
Functions that are general enough to use for any model fitting. The idea is
to untie these from LikelihoodModel so that they may be re-used generally.
"""
from __future__ import print_function

import numpy as np
from scipy import optimize

def _check_method(method, methods):
    if method not in methods:
        message = "Unknown fit method %s" % method
        raise ValueError(message)


class Optimizer(object):
    def _fit(self, objective, gradient, start_params, fargs, kwargs,
             hessian=None, method='newton', maxiter=100, full_output=True,
             disp=True, callback=None, retall=False):
        """
        Fit function for any model with an objective function.

        Parameters
        ----------
        start_params : array-like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str {'newton','nm','bfgs','powell','cg','ncg','basinhopping',
            'minimize'}
            Method can be 'newton' for Newton-Raphson, 'nm' for Nelder-Mead,
            'bfgs' for Broyden-Fletcher-Goldfarb-Shanno, 'powell' for modified
            Powell's method, 'cg' for conjugate gradient, 'ncg' for Newton-
            conjugate gradient, 'basinhopping' for global basin-hopping
            solver, if available or a generic 'minimize' which is a wrapper for
            scipy.optimize.minimize. `method` determines which solver from
            scipy.optimize is used. The explicit arguments in `fit` are passed
            to the solver, with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports..
        maxiter : int
            The maximum number of iterations to perform.
        full_output : bool
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool
            Set to True to print convergence messages.
        fargs : tuple
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args)
        callback : callable callback(xk)
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.

        Returns
        -------
        xopt : array
            The solution to the objective function
        retvals : dict, None
            If `full_output` is True then this is a dictionary which holds
            information returned from the solver used. If it is False, this is
            None.
        optim_settings : dict
            A dictionary that contains the parameters passed to the solver.

        Notes
        -----
        The 'basinhopping' solver ignores `maxiter`, `retall`, `full_output`
        explicit arguments.

        Optional arguments for the solvers (available in Results.mle_settings)::

            'newton'
                tol : float
                    Relative error in params acceptable for convergence.
            'nm' -- Nelder Mead
                xtol : float
                    Relative error in params acceptable for convergence
                ftol : float
                    Relative error in loglike(params) acceptable for
                    convergence
                maxfun : int
                    Maximum number of function evaluations to make.
            'bfgs'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
            'cg'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon : float
                    If fprime is approximated, use this value for the step
                    size. Can be scalar or vector.  Only relevant if
                    Likelihoodmodel.score is None.
            'ncg'
                fhess_p : callable f'(x,*args)
                    Function which computes the Hessian of f times an arbitrary
                    vector, p.  Should only be supplied if
                    LikelihoodModel.hessian is None.
                avextol : float
                    Stop when the average relative error in the minimizer
                    falls below this amount.
                epsilon : float or ndarray
                    If fhess is approximated, use this value for the step size.
                    Only relevant if Likelihoodmodel.hessian is None.
            'powell'
                xtol : float
                    Line-search error tolerance
                ftol : float
                    Relative error in loglike(params) for acceptable for
                    convergence.
                maxfun : int
                    Maximum number of function evaluations to make.
                start_direc : ndarray
                    Initial direction set.
            'basinhopping'
                niter : integer
                    The number of basin hopping iterations.
                niter_success : integer
                    Stop the run if the global minimum candidate remains the
                    same for this number of iterations.
                T : float
                    The "temperature" parameter for the accept or reject
                    criterion. Higher "temperatures" mean that larger jumps
                    in function value will be accepted. For best results
                    `T` should be comparable to the separation (in function
                    value) between local minima.
                stepsize : float
                    Initial step size for use in the random displacement.
                interval : integer
                    The interval for how often to update the `stepsize`.
                minimizer : dict
                    Extra keyword arguments to be passed to the minimizer
                    `scipy.optimize.minimize()`, for example 'method' - the
                    minimization method (e.g. 'L-BFGS-B'), or 'tol' - the
                    tolerance for termination. Other arguments are mapped from
                    explicit argument of `fit`:
                    - `args` <- `fargs`
                    - `jac` <- `score`
                    - `hess` <- `hess`
            'minimize'
                min_method : str, optional
                    Name of minimization method to use.
                    Any method specific arguments can be passed directly.
                    For a list of methods and their arguments, see
                    documentation of `scipy.optimize.minimize`.
                    If no method is specified, then BFGS is used.
        """
        #TODO: generalize the regularization stuff
        # Extract kwargs specific to fit_regularized calling fit
        extra_fit_funcs = kwargs.setdefault('extra_fit_funcs', dict())

        methods = ['newton', 'nm', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg',
                'basinhopping', 'minimize']
        methods += extra_fit_funcs.keys()
        method = method.lower()
        _check_method(method, methods)

        fit_funcs = {
            'newton': _fit_newton,
            'nm': _fit_nm,  # Nelder-Mead
            'bfgs': _fit_bfgs,
            'lbfgs': _fit_lbfgs,
            'cg': _fit_cg,
            'ncg': _fit_ncg,
            'powell': _fit_powell,
            'basinhopping': _fit_basinhopping,
            'minimize': _fit_minimize # wrapper for scipy.optimize.minimize
        }

        #NOTE: fit_regularized checks the methods for these but it should be
        #      moved up probably
        if extra_fit_funcs:
            fit_funcs.update(extra_fit_funcs)

        func = fit_funcs[method]
        xopt, retvals = func(objective, gradient, start_params, fargs, kwargs,
                            disp=disp, maxiter=maxiter, callback=callback,
                            retall=retall, full_output=full_output,
                            hess=hessian)

        optim_settings = {'optimizer': method, 'start_params': start_params,
                        'maxiter': maxiter, 'full_output': full_output,
                        'disp': disp, 'fargs': fargs, 'callback': callback,
                        'retall': retall}
        optim_settings.update(kwargs)
        # set as attributes or return?
        return xopt, retvals, optim_settings

    def _fit_constrained(self, params):
        """
        TODO: how to add constraints?

        Something like
        sm.add_constraint(Model, func)

        or

        model_instance.add_constraint(func)
        model_instance.add_constraint("x1 + x2 = 2")
        result = model_instance.fit()
        """
        pass


    def _fit_regularized(self, params):
        #TODO: code won't necessarily be general here. 3 options.
        # 1) setup for scipy.optimize.fmin_sqlsqp
        # 2) setup for cvxopt
        # 3) setup for openopt
        pass

########################################
# Helper functions to fit


def _fit_minimize(f, score, start_params, fargs, kwargs, disp=True,
                        maxiter=100, callback=None, retall=False,
                        full_output=True, hess=None):
    kwargs.setdefault('min_method', 'BFGS')

    # prepare options dict for minimize
    filter_opts = ['extra_fit_funcs', 'niter', 'min_method', 'tol']
    options = dict((k,v) for k,v in kwargs.items() if k not in filter_opts)
    options['disp']    = disp
    options['maxiter'] = maxiter

    # Use Hessian/Jacobian only if they're required by the method
    no_hess = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'COBYLA', 'SLSQP']
    no_jac  = ['Nelder-Mead', 'Powell', 'COBYLA']
    if kwargs['min_method'] in no_hess: hess = None
    if kwargs['min_method'] in no_jac: score = None

    res = optimize.minimize(f, start_params, args=fargs, method=kwargs['min_method'],
                            jac=score, hess=hess, callback=callback, options=options)

    xopt    = res.x
    retvals = None
    if full_output:
        nit = getattr(res, 'nit', np.nan) # scipy 0.14 compat
        retvals = {'fopt': res.fun, 'iterations': nit,
                   'fcalls': res.nfev, 'warnflag': res.status,
                   'converged': res.success}
        if retall:
            retvals.update({'allvecs': res.values()})

    return xopt, retvals


def _fit_newton(f, score, start_params, fargs, kwargs, disp=True,
                    maxiter=100, callback=None, retall=False,
                    full_output=True, hess=None, ridge_factor=1e-10):
    tol = kwargs.setdefault('tol', 1e-8)
    iterations = 0
    oldparams = np.inf
    newparams = np.asarray(start_params)
    if retall:
        history = [oldparams, newparams]
    while (iterations < maxiter and np.any(np.abs(newparams -
            oldparams) > tol)):
        H = np.asarray(hess(newparams))
        # regularize Hessian, not clear what ridge factor should be
        # keyword option with absolute default 1e-10, see #1847
        if not np.all(ridge_factor == 0):
            H[np.diag_indices(H.shape[0])] += ridge_factor
        oldparams = newparams
        newparams = oldparams - np.dot(np.linalg.inv(H),
                score(oldparams))
        if retall:
            history.append(newparams)
        if callback is not None:
            callback(newparams)
        iterations += 1
    fval = f(newparams, *fargs)  # this is the negative likelihood
    if iterations == maxiter:
        warnflag = 1
        if disp:
            print("Warning: Maximum number of iterations has been "
                   "exceeded.")
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
    else:
        warnflag = 0
        if disp:
            print("Optimization terminated successfully.")
            print("         Current function value: %f" % fval)
            print("         Iterations %d" % iterations)
    if full_output:
        (xopt, fopt, niter,
         gopt, hopt) = (newparams, f(newparams, *fargs),
                        iterations, score(newparams),
                        hess(newparams))
        converged = not warnflag
        retvals = {'fopt': fopt, 'iterations': niter, 'score': gopt,
                   'Hessian': hopt, 'warnflag': warnflag,
                   'converged': converged}
        if retall:
            retvals.update({'allvecs': history})

    else:
        xopt = newparams
        retvals = None

    return xopt, retvals


def _fit_bfgs(f, score, start_params, fargs, kwargs, disp=True,
                    maxiter=100, callback=None, retall=False,
                    full_output=True, hess=None):
    gtol = kwargs.setdefault('gtol', 1.0000000000000001e-05)
    norm = kwargs.setdefault('norm', np.Inf)
    epsilon = kwargs.setdefault('epsilon', 1.4901161193847656e-08)
    retvals = optimize.fmin_bfgs(f, start_params, score, args=fargs,
                                 gtol=gtol, norm=norm, epsilon=epsilon,
                                 maxiter=maxiter, full_output=full_output,
                                 disp=disp, retall=retall, callback=callback)
    if full_output:
        if not retall:
            xopt, fopt, gopt, Hinv, fcalls, gcalls, warnflag = retvals
        else:
            (xopt, fopt, gopt, Hinv, fcalls,
             gcalls, warnflag, allvecs) = retvals
        converged = not warnflag
        retvals = {'fopt': fopt, 'gopt': gopt, 'Hinv': Hinv,
                'fcalls': fcalls, 'gcalls': gcalls, 'warnflag':
                warnflag, 'converged': converged}
        if retall:
            retvals.update({'allvecs': allvecs})
    else:
        xopt = retvals
        retvals = None

    return xopt, retvals


def _fit_lbfgs(f, score, start_params, fargs, kwargs, disp=True,
                   maxiter=100, callback=None, retall=False,
                   full_output=True, hess=None):
    """
    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.

    Notes
    -----
    Within the mle part of statsmodels, the log likelihood function and
    its gradient with respect to the parameters do not have notationally
    consistent sign.
    """

    # Use unconstrained optimization by default.
    bounds = kwargs.setdefault('bounds', [(None, None)] * len(start_params))
    kwargs.setdefault('iprint', 0)

    # Pass the following keyword argument names through to fmin_l_bfgs_b
    # if they are present in kwargs, otherwise use the fmin_l_bfgs_b
    # default values.
    names = ('m', 'pgtol', 'factr', 'maxfun', 'epsilon', 'approx_grad')
    extra_kwargs = dict((x, kwargs[x]) for x in names if x in kwargs)

    # Extract values for the options related to the gradient.
    approx_grad = kwargs.get('approx_grad', False)
    loglike_and_score = kwargs.get('loglike_and_score', None)
    epsilon = kwargs.get('epsilon', None)

    # The approx_grad flag has superpowers nullifying the score function arg.
    if approx_grad:
        score = None

    # Choose among three options for dealing with the gradient (the gradient
    # of a log likelihood function with respect to its parameters
    # is more specifically called the score in statistics terminology).
    # The first option is to use the finite-differences
    # approximation that is built into the fmin_l_bfgs_b optimizer.
    # The second option is to use the provided score function.
    # The third option is to use the score component of a provided
    # function that simultaneously evaluates the log likelihood and score.
    if epsilon and not approx_grad:
        raise ValueError('a finite-differences epsilon was provided '
                         'even though we are not using approx_grad')
    if approx_grad and loglike_and_score:
        raise ValueError('gradient approximation was requested '
                         'even though an analytic loglike_and_score function '
                         'was given')
    if loglike_and_score:
        func = lambda p, *a : tuple(-x for x in loglike_and_score(p, *a))
    elif score:
        func = f
        extra_kwargs['fprime'] = score
    elif approx_grad:
        func = f

    retvals = optimize.fmin_l_bfgs_b(func, start_params, maxiter=maxiter,
                                     callback=callback, args=fargs,
                                     bounds=bounds, disp=disp,
                                     **extra_kwargs)

    if full_output:
        xopt, fopt, d = retvals
        # The warnflag is
        # 0 if converged
        # 1 if too many function evaluations or too many iterations
        # 2 if stopped for another reason, given in d['task']
        warnflag = d['warnflag']
        converged = (warnflag == 0)
        gopt = d['grad']
        fcalls = d['funcalls']
        iterations = d['nit']
        retvals = {'fopt': fopt, 'gopt': gopt, 'fcalls': fcalls,
                   'warnflag': warnflag, 'converged': converged,
                   'iterations': iterations}
    else:
        xopt = retvals[0]
        retvals = None

    return xopt, retvals


def _fit_nm(f, score, start_params, fargs, kwargs, disp=True,
                maxiter=100, callback=None, retall=False,
                full_output=True, hess=None):
    xtol = kwargs.setdefault('xtol', 0.0001)
    ftol = kwargs.setdefault('ftol', 0.0001)
    maxfun = kwargs.setdefault('maxfun', None)
    retvals = optimize.fmin(f, start_params, args=fargs, xtol=xtol,
                            ftol=ftol, maxiter=maxiter, maxfun=maxfun,
                            full_output=full_output, disp=disp, retall=retall,
                            callback=callback)
    if full_output:
        if not retall:
            xopt, fopt, niter, fcalls, warnflag = retvals
        else:
            xopt, fopt, niter, fcalls, warnflag, allvecs = retvals
        converged = not warnflag
        retvals = {'fopt': fopt, 'iterations': niter,
                   'fcalls': fcalls, 'warnflag': warnflag,
                   'converged': converged}
        if retall:
            retvals.update({'allvecs': allvecs})
    else:
        xopt = retvals
        retvals = None

    return xopt, retvals


def _fit_cg(f, score, start_params, fargs, kwargs, disp=True,
                maxiter=100, callback=None, retall=False,
                full_output=True, hess=None):
    gtol = kwargs.setdefault('gtol', 1.0000000000000001e-05)
    norm = kwargs.setdefault('norm', np.Inf)
    epsilon = kwargs.setdefault('epsilon', 1.4901161193847656e-08)
    retvals = optimize.fmin_cg(f, start_params, score, gtol=gtol, norm=norm,
                               epsilon=epsilon, maxiter=maxiter,
                               full_output=full_output, disp=disp,
                               retall=retall, callback=callback)
    if full_output:
        if not retall:
            xopt, fopt, fcalls, gcalls, warnflag = retvals
        else:
            xopt, fopt, fcalls, gcalls, warnflag, allvecs = retvals
        converged = not warnflag
        retvals = {'fopt': fopt, 'fcalls': fcalls, 'gcalls': gcalls,
                   'warnflag': warnflag, 'converged': converged}
        if retall:
            retvals.update({'allvecs': allvecs})

    else:
        xopt = retvals
        retvals = None

    return xopt, retvals


def _fit_ncg(f, score, start_params, fargs, kwargs, disp=True,
                 maxiter=100, callback=None, retall=False,
                 full_output=True, hess=None):
    fhess_p = kwargs.setdefault('fhess_p', None)
    avextol = kwargs.setdefault('avextol', 1.0000000000000001e-05)
    epsilon = kwargs.setdefault('epsilon', 1.4901161193847656e-08)
    retvals = optimize.fmin_ncg(f, start_params, score, fhess_p=fhess_p,
                                fhess=hess, args=fargs, avextol=avextol,
                                epsilon=epsilon, maxiter=maxiter,
                                full_output=full_output, disp=disp,
                                retall=retall, callback=callback)
    if full_output:
        if not retall:
            xopt, fopt, fcalls, gcalls, hcalls, warnflag = retvals
        else:
            xopt, fopt, fcalls, gcalls, hcalls, warnflag, allvecs =\
                retvals
        converged = not warnflag
        retvals = {'fopt': fopt, 'fcalls': fcalls, 'gcalls': gcalls,
                   'hcalls': hcalls, 'warnflag': warnflag,
                   'converged': converged}
        if retall:
            retvals.update({'allvecs': allvecs})
    else:
        xopt = retvals
        retvals = None

    return xopt, retvals


def _fit_powell(f, score, start_params, fargs, kwargs, disp=True,
                    maxiter=100, callback=None, retall=False,
                    full_output=True, hess=None):
    xtol = kwargs.setdefault('xtol', 0.0001)
    ftol = kwargs.setdefault('ftol', 0.0001)
    maxfun = kwargs.setdefault('maxfun', None)
    start_direc = kwargs.setdefault('start_direc', None)
    retvals = optimize.fmin_powell(f, start_params, args=fargs, xtol=xtol,
                                   ftol=ftol, maxiter=maxiter, maxfun=maxfun,
                                   full_output=full_output, disp=disp,
                                   retall=retall, callback=callback,
                                   direc=start_direc)
    if full_output:
        if not retall:
            xopt, fopt, direc, niter, fcalls, warnflag = retvals
        else:
            xopt, fopt, direc, niter, fcalls, warnflag, allvecs =\
                retvals
        converged = not warnflag
        retvals = {'fopt': fopt, 'direc': direc, 'iterations': niter,
                   'fcalls': fcalls, 'warnflag': warnflag,
                   'converged': converged}
        if retall:
            retvals.update({'allvecs': allvecs})
    else:
        xopt = retvals
        retvals = None

    return xopt, retvals


def _fit_basinhopping(f, score, start_params, fargs, kwargs, disp=True,
                          maxiter=100, callback=None, retall=False,
                          full_output=True, hess=None):
    if not 'basinhopping' in vars(optimize):
        msg = 'basinhopping solver is not available, use e.g. bfgs instead!'
        raise ValueError(msg)

    from copy import copy
    kwargs = copy(kwargs)
    niter = kwargs.setdefault('niter', 100)
    niter_success = kwargs.setdefault('niter_success', None)
    T = kwargs.setdefault('T', 1.0)
    stepsize = kwargs.setdefault('stepsize', 0.5)
    interval = kwargs.setdefault('interval', 50)
    minimizer_kwargs = kwargs.get('minimizer', {})
    minimizer_kwargs['args'] = fargs
    minimizer_kwargs['jac'] = score
    method = minimizer_kwargs.get('method', None)
    if method and method != 'L-BFGS-B': # l_bfgs_b doesn't take a hessian
        minimizer_kwargs['hess'] = hess

    retvals = optimize.basinhopping(f, start_params,
                                    minimizer_kwargs=minimizer_kwargs,
                                    niter=niter, niter_success=niter_success,
                                    T=T, stepsize=stepsize, disp=disp,
                                    callback=callback, interval=interval)
    if full_output:
        xopt, fopt, niter, fcalls = map(lambda x : getattr(retvals, x),
                                        ['x', 'fun', 'nit', 'nfev'])
        converged = 'completed successfully' in retvals.message[0]
        retvals = {'fopt': fopt, 'iterations': niter,
                   'fcalls': fcalls, 'converged': converged}

    else:
        xopt = retvals.x
        retvals = None

    return xopt, retvals
