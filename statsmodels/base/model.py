import numpy as np
from scipy import optimize, stats
from statsmodels.base.data import handle_data
from statsmodels.tools.tools import recipr, nan_dot
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.decorators import (resettable_cache,
                                                  cache_readonly)
import statsmodels.base.wrapper as wrap
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.formula import handle_formula_data


_model_params_doc = """\
    Parameters
    ----------
    endog : array-like
        1-d endogenous response variable. The dependent variable.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An interecept is not included by default
        and should be added by the user. See `statsmodels.tools.add_constant`."""

_missing_param_doc = """missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none.'
        """
_extra_param_doc = """hasconst : None or bool
        Indicates whether the RHS includes a user-supplied constant. If True,
        a constant is not checked for and k_constant is set to 1 and all
        result statistics are calculated as if a constant is present. If
        False, a constant is not checked for and k_constant is set to 0.
        """

class Model(object):
    __doc__ = """
    A (predictive) statistical model. Intended to be subclassed not used.

    %(params_doc)s
    %(extra_params_doc)s

    Notes
    -----
    `endog` and `exog` are references to any data provided.  So if the data is
    already stored in numpy arrays and it is changed then `endog` and `exog`
    will change as well.
    """ % {'params_doc' : _model_params_doc,
            'extra_params_doc' : _missing_param_doc + _extra_param_doc}
    def __init__(self, endog, exog=None, **kwargs):
        missing = kwargs.pop('missing', 'none')
        hasconst = kwargs.pop('hasconst', None)
        self.data = handle_data(endog, exog, missing, hasconst, **kwargs)
        self.k_constant = self.data.k_constant
        self.exog = self.data.exog
        self.endog = self.data.endog
        # kwargs arrays could have changed, easier to just attach here
        for key in kwargs:
            # pop so we don't start keeping all these twice or references
            setattr(self, key, self.data.__dict__.pop(key))
        self._data_attr = []
        self._data_attr.extend(['exog', 'endog', 'data.exog', 'data.endog',
                                'data.orig_endog', 'data.orig_exog'])

    @classmethod
    def from_formula(cls, formula, data, subset=None, *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array-like
            The data for the model. See Notes.
        subset : array-like
            An array-like object of booleans, integers, or index values that
            indicate the subset of df to use in the model. Assumes df is a
            `pandas.DataFrame`
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model.

        Returns
        -------
        model : Model instance

        Notes
        ------
        data must define __getitem__ with the keys in the formula terms
        args and kwargs are passed on to the model instantiation. E.g.,
        a numpy structured or rec array, a dictionary, or a pandas DataFrame.
        """
        #TODO: provide a docs template for args/kwargs from child models
        #TODO: subset could use syntax. issue #469.
        if subset is not None:
            data= data.ix[subset]
        endog, exog = handle_formula_data(data, None, formula)
        mod = cls(endog, exog, *args, **kwargs)
        mod.formula = formula

        # since we got a dataframe, attach the original
        mod.data.frame = data
        return mod


    @property
    def endog_names(self):
        return self.data.ynames

    @property
    def exog_names(self):
        return self.data.xnames

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, params, exog=None, *args, **kwargs):
        """
        After a model has been fit predict returns the fitted values.

        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError


class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

    def __init__(self, endog, exog=None, **kwargs):
        super(LikelihoodModel, self).__init__(endog, exog, **kwargs)
        self.initialize()

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        pass

    # TODO: if the intent is to re-initialize the model with new data then this
    # method needs to take inputs...

    def loglike(self, params):
        """
        Log-likelihood of model.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.
        """
        raise NotImplementedError

    def information(self, params):
        """
        Fisher information matrix of model

        Returns -Hessian of loglike evaluated at params.
        """
        raise NotImplementedError

    def hessian(self, params):
        """
        The Hessian matrix of the model
        """
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=True, disp=True, fargs=(), callback=None,
            retall=False, **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array-like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str {'newton','nm','bfgs','powell','cg','ncg','basinhopping'}
            Method can be 'newton' for Newton-Raphson, 'nm' for Nelder-Mead,
            'bfgs' for Broyden-Fletcher-Goldfarb-Shanno, 'powell' for modified
            Powell's method, 'cg' for conjugate gradient, 'ncg' for Newton-
            conjugate gradient or 'basinhopping' for global basin-hopping
            solver, if available. `method` determines which solver from
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
        """
        # Extract kwargs specific to fit_regularized calling fit
        extra_fit_funcs = kwargs.setdefault('extra_fit_funcs', dict())
        cov_params_func = kwargs.setdefault('cov_params_func', None)

        Hinv = None  # JP error if full_output=0, Hinv not defined
        methods = ['newton', 'nm', 'bfgs', 'powell', 'cg', 'ncg',
                   'basinhopping']
        methods += extra_fit_funcs.keys()
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            elif self.exog is not None:
                # fails for shape (K,)?
                start_params = [0] * self.exog.shape[1]
            else:
                raise ValueError("If exog is None, then start_params should "
                                 "be specified")

        if method.lower() not in methods:
            message = "Unknown fit method %s" % method
            raise ValueError(message)
        method = method.lower()

        # TODO: separate args from nonarg taking score and hessian, ie.,
        # user-supplied and numerically evaluated estimate frprime doesn't take
        # args in most (any?) of the optimize function

        nobs = self.endog.shape[0]
        f = lambda params, *args: -self.loglike(params, *args) / nobs
        score = lambda params: -self.score(params) / nobs
        try:
            hess = lambda params: -self.hessian(params) / nobs
        except:
            hess = None

        fit_funcs = {
            'newton': _fit_mle_newton,
            'nm': _fit_mle_nm,  # Nelder-Mead
            'bfgs': _fit_mle_bfgs,
            'cg': _fit_mle_cg,
            'ncg': _fit_mle_ncg,
            'powell': _fit_mle_powell,
            'basinhopping': _fit_mle_basinhopping,
        }
        if extra_fit_funcs:
            fit_funcs.update(extra_fit_funcs)

        if method == 'newton':
            score = lambda params: self.score(params) / nobs
            hess = lambda params: self.hessian(params) / nobs
            #TODO: why are score and hess positive?

        func = fit_funcs[method]
        xopt, retvals = func(f, score, start_params, fargs, kwargs,
                             disp=disp, maxiter=maxiter, callback=callback,
                             retall=retall, full_output=full_output,
                             hess=hess)

        if not full_output: # xopt should be None and retvals is argmin
            xopt = retvals

        elif cov_params_func:
            Hinv = cov_params_func(self, xopt, retvals)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-retvals['Hessian']) / nobs
        else:
            try:
                Hinv = np.linalg.inv(-1 * self.hessian(xopt))
            except:
                #might want custom warning ResultsWarning? NumericalWarning?
                from warnings import warn
                warndoc = ('Inverting hessian failed, no bse or '
                           'cov_params available')
                warn(warndoc, Warning)
                Hinv = None

        #TODO: add Hessian approximation and change the above if needed
        mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1.)

        #TODO: hardcode scale?
        if isinstance(retvals, dict):
            mlefit.mle_retvals = retvals
        optim_settings = {'optimizer': method, 'start_params': start_params,
                          'maxiter': maxiter, 'full_output': full_output,
                          'disp': disp, 'fargs': fargs, 'callback': callback,
                          'retall': retall}
        optim_settings.update(kwargs)
        mlefit.mle_settings = optim_settings
        return mlefit


def _fit_mle_newton(f, score, start_params, fargs, kwargs, disp=True,
                    maxiter=100, callback=None, retall=False,
                    full_output=True, hess=None):
    tol = kwargs.setdefault('tol', 1e-8)
    iterations = 0
    oldparams = np.inf
    newparams = np.asarray(start_params)
    if retall:
        history = [oldparams, newparams]
    while (iterations < maxiter and np.any(np.abs(newparams -
            oldparams) > tol)):
        H = hess(newparams)
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
            print ("Warning: Maximum number of iterations has been "
                   "exceeded.")
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % iterations
    else:
        warnflag = 0
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations %d" % iterations
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
        retvals = newparams
        xopt = None

    return xopt, retvals



def _fit_mle_bfgs(f, score, start_params, fargs, kwargs, disp=True,
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
        xopt = None

    return xopt, retvals


def _fit_mle_nm(f, score, start_params, fargs, kwargs, disp=True,
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
        xopt = None

    return xopt, retvals


def _fit_mle_cg(f, score, start_params, fargs, kwargs, disp=True,
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
        xopt = None

    return xopt, retvals


def _fit_mle_ncg(f, score, start_params, fargs, kwargs, disp=True,
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
        xopt = None

    return xopt, retvals


def _fit_mle_powell(f, score, start_params, fargs, kwargs, disp=True,
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
        xopt = None

    return xopt, retvals

def _fit_mle_basinhopping(f, score, start_params, fargs, kwargs, disp=True,
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
    minimizer_kwargs['hess'] = hess

    res = optimize.basinhopping(f, start_params,
                                minimizer_kwargs=minimizer_kwargs,
                                niter=niter, niter_success=niter_success,
                                T=T, stepsize=stepsize, disp=disp,
                                callback=callback, interval=interval)
    if full_output:
        xopt, fopt, niter, fcalls = res.x, res.fun, res.nit, res.nfev
        converged = 'completed successfully' in res.message[0]
        retvals = {'fopt': fopt, 'iterations': niter,
                   'fcalls': fcalls, 'converged': converged}

    else:
        xopt = None

    return xopt, retvals

#TODO: the below is unfinished
class GenericLikelihoodModel(LikelihoodModel):
    """
    Allows the fitting of any likelihood function via maximum likelihood.

    A subclass needs to specify at least the log-likelihood
    If the log-likelihood is specified for each observation, then results that
    require the Jacobian will be available. (The other case is not tested yet.)

    Notes
    -----
    Optimization methods that require only a likelihood function are 'nm' and
    'powell'

    Optimization methods that require a likelihood function and a
    score/gradient are 'bfgs', 'cg', and 'ncg'. A function to compute the
    Hessian is optional for 'ncg'.

    Optimization method that require a likelihood function, a score/gradient,
    and a Hessian is 'newton'

    If they are not overwritten by a subclass, then numerical gradient,
    Jacobian and Hessian of the log-likelihood are caclulated by numerical
    forward differentiation. This might results in some cases in precision
    problems, and the Hessian might not be positive definite. Even if the
    Hessian is not positive definite the covariance matrix of the parameter
    estimates based on the outer product of the Jacobian might still be valid.


    Examples
    --------
    see also subclasses in directory miscmodels

    import statsmodels.api as sm
    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog)
    # in this dir
    from model import GenericLikelihoodModel
    probit_mod = sm.Probit(data.endog, data.exog)
    probit_res = probit_mod.fit()
    loglike = probit_mod.loglike
    score = probit_mod.score
    mod = GenericLikelihoodModel(data.endog, data.exog, loglike, score)
    res = mod.fit(method="nm", maxiter = 500)
    import numpy as np
    np.allclose(res.params, probit_res.params)

    """
    def __init__(self, endog, exog=None, loglike=None, score=None,
                 hessian=None, missing='none', extra_params_names=None, **kwds):
    # let them be none in case user wants to use inheritance
        if not loglike is None:
            self.loglike = loglike
        if not score is None:
            self.score = score
        if not hessian is None:
            self.hessian = hessian
        self.confint_dist = stats.norm

        self.__dict__.update(kwds)

        # TODO: data structures?

        #TODO temporary solution, force approx normal
        #self.df_model = 9999
        #somewhere: CacheWriteWarning: The attribute 'df_model' cannot be overwritten
        super(GenericLikelihoodModel, self).__init__(endog, exog, missing=missing)

        # this won't work for ru2nmnl, maybe np.ndim of a dict?
        if exog is not None:
            #try:
            self.nparams = (exog.shape[1] if np.ndim(exog) == 2 else 1)

        if extra_params_names is not None:
            self._set_extra_params_names(extra_params_names)

    def _set_extra_params_names(self, extra_params_names):
        # check param_names
        if extra_params_names is not None:
            if self.exog is not None:
                self.exog_names.extend(extra_params_names)
            else:
                self.data.xnames = extra_params_names

        self.nparams = len(self.exog_names)


    #this is redundant and not used when subclassing
    def initialize(self):
        if not self.score:  # right now score is not optional
            self.score = approx_fprime
            if not self.hessian:
                pass
        else:   # can use approx_hess_p if we have a gradient
            if not self.hessian:
                pass
        #Initialize is called by
        #statsmodels.model.LikelihoodModel.__init__
        #and should contain any preprocessing that needs to be done for a model.
        from statsmodels.tools import tools
        if self.exog is not None:
            self.df_model = float(tools.rank(self.exog) - 1)  # assumes constant
            self.df_resid = float(self.exog.shape[0] - tools.rank(self.exog))
        else:
            self.df_model = np.nan
            self.df_resid = np.nan
        super(GenericLikelihoodModel, self).initialize()

    def expandparams(self, params):
        '''
        expand to full parameter array when some parameters are fixed

        Parameters
        ----------
        params : array
            reduced parameter array

        Returns
        -------
        paramsfull : array
            expanded parameter array where fixed parameters are included

        Notes
        -----
        Calling this requires that self.fixed_params and self.fixed_paramsmask
        are defined.

        *developer notes:*

        This can be used in the log-likelihood to ...

        this could also be replaced by a more general parameter
        transformation.

        '''
        paramsfull = self.fixed_params.copy()
        paramsfull[self.fixed_paramsmask] = params
        return paramsfull

    def reduceparams(self, params):
        return params[self.fixed_paramsmask]

    def loglike(self, params):
        return self.loglikeobs(params).sum(0)

    def nloglike(self, params):
        return -self.loglikeobs(params).sum(0)

    def loglikeobs(self, params):
        return -self.nloglikeobs(params)

    def score(self, params):
        '''
        Gradient of log-likelihood evaluated at params
        '''
        kwds = {}
        kwds.setdefault('centered', True)
        return approx_fprime(params, self.loglike, **kwds).ravel()

    def jac(self, params, **kwds):
        '''
        Jacobian/Gradient of log-likelihood evaluated at params for each
        observation.
        '''
        #kwds.setdefault('epsilon', 1e-4)
        kwds.setdefault('centered', True)
        return approx_fprime(params, self.loglikeobs, **kwds)

    def hessian(self, params):
        '''
        Hessian of log-likelihood evaluated at params
        '''
        from statsmodels.tools.numdiff import approx_hess
        # need options for hess (epsilon)
        return approx_hess(params, self.loglike)

    def fit(self, start_params=None, method='nm', maxiter=500, full_output=1,
            disp=1, callback=None, retall=0, **kwargs):
        """
        Fit the model using maximum likelihood.

        The rest of the docstring is from
        statsmodels.LikelihoodModel.fit
        """
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = 0.1 * np.ones(self.nparams)

        fit_method = super(GenericLikelihoodModel, self).fit
        mlefit = fit_method(start_params=start_params,
                            method=method, maxiter=maxiter,
                            full_output=full_output,
                            disp=disp, callback=callback, **kwargs)
        genericmlefit = GenericLikelihoodModelResults(self, mlefit)

        #amend param names
        exog_names = [] if (self.exog_names is None) else self.exog_names
        k_miss = len(exog_names) - len(mlefit.params)
        if not k_miss == 0:
            if k_miss < 0:
                self._set_extra_params_names(
                                         ['par%d' % i for i in range(-k_miss)])
            else:
                # I don't want to raise after we have already fit()
                import warnings
                warnings.warn('more exog_names than parameters', UserWarning)

        return genericmlefit
    #fit.__doc__ += LikelihoodModel.fit.__doc__


class Results(object):
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        the previously specified model instance
    params : array
        parameter estimates from the fit model
    """
    def __init__(self, model, params, **kwd):
        self.__dict__.update(kwd)
        self.initialize(model, params, **kwd)
        self._data_attr = []

    def initialize(self, model, params, **kwd):
        self.params = params
        self.model = model
        if hasattr(model, 'k_constant'):
            self.k_constant = model.k_constant

    def predict(self, exog=None, transform=True, *args, **kwargs):
        """
        Call self.model.predict with self.params as the first argument.

        Parameters
        ----------
        exog : array-like, optional
            The values for which you want to predict.
        transform : bool, optional
            If the model was fit via a formula, do you want to pass
            exog through the formula. Default is True. E.g., if you fit
            a model y ~ log(x1) + log(x2), and transform is True, then
            you can pass a data structure that contains x1 and x2 in
            their original form. Otherwise, you'd need to log the data
            first.

        Returns
        -------
        See self.model.predict
        """
        if transform and hasattr(self.model, 'formula') and exog is not None:
            from patsy import dmatrix
            exog = dmatrix(self.model.data.orig_exog.design_info.builder,
                    exog)
        return self.model.predict(self.params, exog, *args, **kwargs)


#TODO: public method?
class LikelihoodModelResults(Results):
    """
    Class to contain results from likelihood models

    Parameters
    -----------
    model : LikelihoodModel instance or subclass instance
        LikelihoodModelResults holds a reference to the model that is fit.
    params : 1d array_like
        parameter estimates from estimated model
    normalized_cov_params : 2d array
       Normalized (before scaling) covariance of params. (dot(X.T,X))**-1
    scale : float
        For (some subset of models) scale will typically be the
        mean square error from the estimated model (sigma^2)

    Returns
    -------
    **Attributes**
    mle_retvals : dict
        Contains the values returned from the chosen optimization method if
        full_output is True during the fit.  Available only if the model
        is fit by maximum likelihood.  See notes below for the output from
        the different methods.
    mle_settings : dict
        Contains the arguments passed to the chosen optimization method.
        Available if the model is fit by maximum likelihood.  See
        LikelihoodModel.fit for more information.
    model : model instance
        LikelihoodResults contains a reference to the model that is fit.
    params : ndarray
        The parameters estimated for the model.
    scale : float
        The scaling factor of the model given during instantiation.
    tvalues : array
        The t-values of the standard errors.


    Notes
    --------
    The covariance of params is given by scale times normalized_cov_params.

    Return values by solver if full_ouput is True during fit:

        'newton'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            score : ndarray
                The score vector at the optimum.
            Hessian : ndarray
                The Hessian at the optimum.
            warnflag : int
                1 if maxiter is exceeded. 0 if successful convergence.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'nm'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            warnflag : int
                1: Maximum number of function evaluations made.
                2: Maximum number of iterations reached.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'bfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            Hinv : ndarray
                value of the inverse Hessian matrix at minimum.  Note
                that this is just an approximation and will often be
                different from the value of the analytic Hessian.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient
                and/or function calls are not changing.
            converged : bool
                True: converged.  False: did not converge.
            allvecs : list
                Results at each iteration.
        'powell'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            direc : ndarray
                Current direction set.
            iterations : int
                Number of iterations performed.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                1: Maximum number of function evaluations. 2: Maximum number
                of iterations.
            converged : bool
                True : converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'cg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient and/
                or function calls not changing.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'ncg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            hcalls : int
                Number of calls to hessian.
            warnflag : int
                1: Maximum number of iterations exceeded.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        """
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(LikelihoodModelResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale

    def normalized_cov_params(self):
        raise NotImplementedError

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        return self.params / self.bse

    @cache_readonly
    def pvalues(self):
        return stats.norm.sf(np.abs(self.tvalues)) * 2

    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None,
            other=None):
        """
        Returns the variance/covariance matrix.

        The variance/covariance matrix can be of a linear contrast
        of the estimates of params or all params multiplied by scale which
        will usually be an estimate of sigma^2.  Scale is assumed to be
        a scalar.

        Parameters
        ----------
        r_matrix : array-like
            Can be 1d, or 2d.  Can be used alone or with other.
        column :  array-like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        other : array-like, optional
            Can be used when r_matrix is specified.

        Returns
        -------
        (The below are assumed to be in matrix notation.)

        cov : ndarray

        If no argument is specified returns the covariance matrix of a model
        (scale)*(X.T X)^(-1)

        If contrast is specified it pre and post-multiplies as follows
        (scale) * r_matrix (X.T X)^(-1) r_matrix.T

        If contrast and other are specified returns
        (scale) * r_matrix (X.T X)^(-1) other.T

        If column is specified returns
        (scale) * (X.T X)^(-1)[column,column] if column is 0d

        OR

        (scale) * (X.T X)^(-1)[column][:,column] if column is 1d

        """
        if (hasattr(self, 'mle_settings') and
            self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            dot_fun = nan_dot
        else:
            dot_fun = np.dot

        if cov_p is None and self.normalized_cov_params is None:
            raise ValueError('need covariance of parameters for computing '
                             '(unnormalized) covariances')
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError('Column should be specified without other '
                             'arguments.')
        if other is not None and r_matrix is None:
            raise ValueError('other can only be specified with r_matrix')

        if cov_p is None:
            if scale is None:
                scale = self.scale
            cov_p = self.normalized_cov_params * scale

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return cov_p[column, column]
            else:
                #return cov_p[column][:, column]
                return cov_p[column[:, None], column]
        elif r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():
                raise ValueError("r_matrix should be 1d or 2d")
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = dot_fun(r_matrix, dot_fun(cov_p, np.transpose(other)))
            return tmp
        else:  #if r_matrix is None and column is None:
            return cov_p

    #TODO: make sure this works as needed for GLMs
    def t_test(self, r_matrix, q_matrix=None, cov_p=None, scale=None):
        """
        Compute a t-test for a joint linear hypothesis of the form Rb = q

        Parameters
        ----------
        r_matrix : array-like, str, tuple
            - array : If an array is given, a p x k 2d array or length k 1d
              array specifying the linear restrictions.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), since q_matrix is
              deprecated.
        q_matrix : array-like or scalar, optional
            This is deprecated. See `r_matrix` and the examples for more
            information on new usage. Can be either a scalar or a length p
            row vector. If omitted and r_matrix is an array, `q_matrix` is
            assumed to be a conformable array of zeros.
        cov_p : array-like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> r = np.zeros_like(results.params)
        >>> r[5:] = [1,-1]
        >>> print r
        [ 0.  0.  0.  0.  0.  1. -1.]

        r tests that the coefficients on the 5th and 6th independent
        variable are the same.

        >>>T_Test = results.t_test(r)
        >>>print T_test
        <T contrast: effect=-1829.2025687192481, sd=455.39079425193762,
        t=-4.0167754636411717, p=0.0015163772380899498, df_denom=9>
        >>> T_test.effect
        -1829.2025687192481
        >>> T_test.sd
        455.39079425193762
        >>> T_test.tvalue
        -4.0167754636411717
        >>> T_test.pvalue
        0.0015163772380899498

        Alternatively, you can specify the hypothesis tests using a string

        >>> dta = sm.datasets.longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = 'GNPDEFL = GNP, UNEMP = 2, YEAR/1829 = 1'
        >>> t_test = results.t_test(hypotheses)
        >>> print t_test

        See also
        ---------
        tvalues : individual t statistics
        f_test : for F tests
        patsy.DesignInfo.linear_constraint
        """
        from patsy import DesignInfo
        if q_matrix is not None:
            from warnings import warn
            warn("The `q_matrix` keyword is deprecated and will be removed "
                 "in 0.6.0. See the documentation for the new API",
                 FutureWarning)
            r_matrix = (r_matrix, q_matrix)
        LC = DesignInfo(self.model.exog_names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants
        num_ttests = r_matrix.shape[0]
        num_params = r_matrix.shape[1]

        if cov_p is None and self.normalized_cov_params is None:
            raise ValueError('Need covariance of parameters for computing '
                             'T statistics')
        if num_params != self.params.shape[0]:
            raise ValueError('r_matrix and params are not aligned')
        if q_matrix is None:
            q_matrix = np.zeros(num_ttests)
        else:
            q_matrix = np.asarray(q_matrix)
            q_matrix = q_matrix.squeeze()
        if q_matrix.size > 1:
            if q_matrix.shape[0] != num_ttests:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")

        _t = _sd = None

        _effect = np.dot(r_matrix, self.params)
        # nan_dot multiplies with the convention nan * 0 = 0

        # Perform the test
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(self.cov_params(
                r_matrix=r_matrix, cov_p=cov_p)))
        else:
            _sd = np.sqrt(self.cov_params(r_matrix=r_matrix, cov_p=cov_p))
        _t = (_effect - q_matrix) * recipr(_sd)
        return ContrastResults(effect=_effect, t=_t, sd=_sd,
                               df_denom=self.model.df_resid)


    #TODO: untested for GLMs?
    def f_test(self, r_matrix, q_matrix=None, cov_p=None, scale=1.0,
               invcov=None):
        """
        Compute an F-test for a joint linear hypothesis.

        Parameters
        ----------
        r_matrix : array-like, str, or tuple
            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), since q_matrix is
              deprecated.
        q_matrix : array-like
            This is deprecated. See `r_matrix` and the examples for more
            information on new usage. Can be either a scalar or a length p
            row vector. If omitted and r_matrix is an array, `q_matrix` is
            assumed to be a conformable array of zeros.
        cov_p : array-like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            Default is 1.0 for no scaling.
        invcov : array-like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[1:,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>> print results.f_test(A)
        <F contrast: F=330.28533923463488, p=4.98403052872e-10,
         df_denom=9, df_num=6>

        Compare this to

        >>> results.F
        330.2853392346658
        >>> results.F_p
        4.98403096572e-10

        >>> B = np.array(([0,0,1,-1,0,0,0],[0,0,0,0,0,1,-1]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print results.f_test(B)
        <F contrast: F=9.740461873303655, p=0.00560528853174, df_denom=9,
         df_num=2>

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.datasets import longley
        >>> from statsmodels.formula.api import ols
        >>> dta = longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
        >>> f_test = results.f_test(hypotheses)
        >>> print f_test

        See also
        --------
        statsmodels.contrasts
        statsmodels.model.t_test
        patsy.DesignInfo.linear_constraint

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        from patsy import DesignInfo
        if q_matrix is not None:
            from warnings import warn
            warn("The `q_matrix` keyword is deprecated and will be removed "
                 "in 0.6.0. See the documentation for the new API",
                 FutureWarning)
            r_matrix = (r_matrix, q_matrix)
        LC = DesignInfo(self.model.exog_names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants

        if (self.normalized_cov_params is None and cov_p is None and
            invcov is None):
            raise ValueError('need covariance of parameters for computing '
                             'F statistics')

        cparams = np.dot(r_matrix, self.params[:, None])
        J = float(r_matrix.shape[0])  # number of restrictions
        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_p).max():
                raise ValueError("r_matrix performs f_test for using "
                    "dimensions that are asymptotically non-normal")
            invcov = np.linalg.inv(cov_p)

        if (hasattr(self, 'mle_settings') and
            self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            F = nan_dot(nan_dot(Rbq.T, invcov), Rbq) / J
        else:
            F = np.dot(np.dot(Rbq.T, invcov), Rbq) / J
        return ContrastResults(F=F, df_denom=self.model.df_resid,
                    df_num=invcov.shape[0])

    def conf_int(self, alpha=.05, cols=None, method='default'):
        """
        Returns the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
            `cols` specifies which confidence intervals to return
        method : string
            Not Implemented Yet
            Method to estimate the confidence_interval.
            "Default" : uses self.bse which is based on inverse Hessian for MLE
            "jhj" :
            "jac" :
            "boot-bse"
            "boot_quant"
            "profile"


        Returns
        --------
        conf_int : array
            Each row contains [lower, upper] confidence interval

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.conf_int()
        array([[-5496529.48322745, -1467987.78596704],
               [    -177.02903529,      207.15277984],
               [      -0.1115811 ,        0.03994274],
               [      -3.12506664,       -0.91539297],
               [      -1.5179487 ,       -0.54850503],
               [      -0.56251721,        0.460309  ],
               [     798.7875153 ,     2859.51541392]])


        >>> results.conf_int(cols=(2,3))
        array([[-0.1115811 ,  0.03994274],
               [-3.12506664, -0.91539297]])

        Notes
        -----
        The confidence interval is based on the standard normal distribution.
        Models wish to use a different distribution should overwrite this
        method.
        """
        bse = self.bse
        dist = stats.norm
        q = dist.ppf(1 - alpha / 2)

        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = self.params[cols] - q * bse[cols]
            upper = self.params[cols] + q * bse[cols]
        return np.asarray(zip(lower, upper))

    def save(self, fname, remove_data=False):
        '''
        save a pickle of this instance

        Parameters
        ----------
        fname : string or filehandle
            fname can be a string to a file path or filename, or a filehandle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.

        Notes
        -----
        If remove_data is true and the model result does not implement a
        remove_data method then this will raise an exception.

        '''

        from statsmodels.iolib.smpickle import save_pickle

        if remove_data:
            self.remove_data()

        save_pickle(self, fname)

    @classmethod
    def load(cls, fname):
        '''
        load a pickle, (class method)

        Parameters
        ----------
        fname : string or filehandle
            fname can be a string to a file path or filename, or a filehandle.

        Returns
        -------
        unpickled instance

        '''

        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)

    def remove_data(self):
        '''remove data arrays, all nobs arrays from result and model

        This reduces the size of the instance, so it can be pickled with less
        memory. Currently tested for use with predict from an unpickled
        results and model instance.

        .. warning:: Since data and some intermediate results have been removed
           calculating new statistics that require them will raise exceptions.
           The exception will occur the first time an attribute is accessed that
           has been set to None.

        Not fully tested for time series models, tsa, and might delete too much
        for prediction or not all that would be possible.

        The list of arrays to delete is maintained as an attribute of the
        result and model instance, except for cached values. These lists could
        be changed before calling remove_data.

        '''
        def wipe(obj, att):
            #get to last element in attribute path
            p = att.split('.')
            att_ = p.pop(-1)
            try:
                obj_ = reduce(getattr, [obj] + p)

                #print repr(obj), repr(att)
                #print hasattr(obj_, att_)
                if hasattr(obj_, att_):
                    #print 'removing3', att_
                    setattr(obj_, att_, None)
            except AttributeError:
                pass

        model_attr = ['model.'+ i for i in self.model._data_attr]
        for att in self._data_attr + model_attr:
            #print 'removing', att
            wipe(self, att)

        data_in_cache = getattr(self, 'data_in_cache', [])
        data_in_cache += ['fittedvalues', 'resid', 'wresid']
        for key in data_in_cache:
            try:
                self._cache[key] = None
            except (AttributeError, KeyError):
                pass


class LikelihoodResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
        'bse': 'columns',
        'pvalues': 'columns',
        'tvalues': 'columns',
        'resid': 'rows',
        'fittedvalues': 'rows',
        'normalized_cov_params': 'cov',
    }

    _wrap_attrs = _attrs
    _wrap_methods = {
        'cov_params': 'cov',
        'conf_int': 'columns'
    }

wrap.populate_wrapper(LikelihoodResultsWrapper,
                      LikelihoodModelResults)


class ResultMixin(object):

    @cache_readonly
    def df_modelwc(self):
        # collect different ways of defining the number of parameters, used for
        # aic, bic
        if hasattr(self, 'df_model'):
            if hasattr(self, 'hasconst'):
                hasconst = self.hasconst
            else:
                # default assumption
                hasconst = 1
            return self.df_model + hasconst
        else:
            return self.params.size

    @cache_readonly
    def aic(self):
        return -2 * self.llf + 2 * (self.df_modelwc)

    @cache_readonly
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * (self.df_modelwc)

    @cache_readonly
    def jacv(self):
        '''cached Jacobian of log-likelihood
        '''
        return self.model.jac(self.params)

    @cache_readonly
    def hessv(self):
        '''cached Hessian of log-likelihood
        '''
        return self.model.hessian(self.params)

    @cache_readonly
    def covjac(self):
        '''
        covariance of parameters based on outer product of jacobian of
        log-likelihood

        '''
##        if not hasattr(self, '_results'):
##            raise ValueError('need to call fit first')
##            #self.fit()
##        self.jacv = jacv = self.jac(self._results.params)
        jacv = self.jacv
        return np.linalg.inv(np.dot(jacv.T, jacv))

    @cache_readonly
    def covjhj(self):
        '''covariance of parameters based on HJJH

        dot product of Hessian, Jacobian, Jacobian, Hessian of likelihood

        name should be covhjh
        '''
        jacv = self.jacv
##        hessv = self.hessv
##        hessinv = np.linalg.inv(hessv)
##        self.hessinv = hessinv
        hessinv = self.cov_params()
        return np.dot(hessinv, np.dot(np.dot(jacv.T, jacv), hessinv))

    @cache_readonly
    def bsejhj(self):
        '''standard deviation of parameter estimates based on covHJH
        '''
        return np.sqrt(np.diag(self.covjhj))

    @cache_readonly
    def bsejac(self):
        '''standard deviation of parameter estimates based on covjac
        '''
        return np.sqrt(np.diag(self.covjac))

    def bootstrap(self, nrep=100, method='nm', disp=0, store=1):
        '''simple bootstrap to get mean and variance of estimator

        see notes

        Parameters
        ----------
        nrep : int
            number of bootstrap replications
        method : str
            optimization method to use
        disp : bool
            If true, then optimization prints results
        store : bool
            If true, then parameter estimates for all bootstrap iterations
            are attached in self.bootstrap_results

        Returns
        -------
        mean : array
            mean of parameter estimates over bootstrap replications
        std : array
            standard deviation of parameter estimates over bootstrap
            replications

        Notes
        -----
        This was mainly written to compare estimators of the standard errors of
        the parameter estimates.  It uses independent random sampling from the
        original endog and exog, and therefore is only correct if observations
        are independently distributed.

        This will be moved to apply only to models with independently
        distributed observations.
        '''
        results = []
        print self.model.__class__
        hascloneattr = True if hasattr(self, 'cloneattr') else False
        for i in xrange(nrep):
            rvsind = np.random.randint(self.nobs - 1, size=self.nobs)
            #this needs to set startparam and get other defining attributes
            #need a clone method on model
            fitmod = self.model.__class__(self.endog[rvsind],
                                          self.exog[rvsind, :])
            if hascloneattr:
                for attr in self.model.cloneattr:
                    setattr(fitmod, attr, getattr(self.model, attr))

            fitres = fitmod.fit(method=method, disp=disp)
            results.append(fitres.params)
        results = np.array(results)
        if store:
            self.bootstrap_results = results
        return results.mean(0), results.std(0), results

    def get_nlfun(self, fun):
        #I think this is supposed to get the delta method that is currently
        #in miscmodels count (as part of Poisson example)
        pass


class GenericLikelihoodModelResults(LikelihoodModelResults, ResultMixin):
    """
    A results class for the discrete dependent variable models.

    ..Warning :

    The following description has not been updated to this version/class.
    Where are AIC, BIC, ....? docstring looks like copy from discretemod

    Parameters
    ----------
    model : A DiscreteModel instance
    mlefit : instance of LikelihoodResults
        This contains the numerical optimization results as returned by
        LikelihoodModel.fit(), in a superclass of GnericLikelihoodModels


    Returns
    -------
    *Attributes*

    Warning most of these are not available yet

    aic : float
        Akaike information criterion.  -2*(`llf` - p) where p is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. -2*`llf` + ln(`nobs`)*p where p is the
        number of regressors including the intercept.
    bse : array
        The standard errors of the coefficients.
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    fitted_values : array
        Linear predictor XB.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; -2*(`llnull` - `llf`)
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. 1 - (`llf`/`llnull`)

    """

    def __init__(self, model, mlefit):
#        super(DiscreteResults, self).__init__(model, params,
#                np.linalg.inv(-hessian), scale=1.)
        self.model = model
        self.endog = model.endog
        self.exog = model.exog
        self.nobs = model.endog.shape[0]

        # TODO: possibly move to model.fit()
        #       and outsource together with patching names
        if hasattr(model, 'df_model'):
            self.df_model = model.df_model
        else:
            self.df_model = len(mlefit.params)
            # retrofitting the model, used in t_test TODO: check design
            self.model.df_model = self.df_model

        if hasattr(model, 'df_resid'):
            self.df_resid = model.df_resid
        else:
            self.df_resid = self.endog.shape[0] - self.df_model
            # retrofitting the model, used in t_test TODO: check design
            self.model.df_resid = self.df_resid

        self._cache = resettable_cache()
        self.__dict__.update(mlefit.__dict__)

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Maximum Likelihood']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None), #[self.df_resid]), #TODO: spelling
                    ('Df Model:', None), #[self.df_model])
                    ]

        top_right = [#('R-squared:', ["%#8.3f" % self.rsquared]),
                     #('Adj. R-squared:', ["%#8.3f" % self.rsquared_adj]),
                     #('F-statistic:', ["%#8.4g" % self.fvalue] ),
                     #('Prob (F-statistic):', ["%#6.3g" % self.f_pvalue]),
                     ('Log-Likelihood:', None), #["%#6.4g" % self.llf]),
                     ('AIC:', ["%#8.4g" % self.aic]),
                     ('BIC:', ["%#8.4g" % self.bic])
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Results"

        #create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                          yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                             use_t=False)

        return smry
