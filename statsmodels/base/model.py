import numpy as np
from scipy import optimize, stats
from statsmodels.base.data import handle_data
from statsmodels.tools.tools import recipr
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.decorators import (resettable_cache,
                                                  cache_readonly)
import statsmodels.base.wrapper as wrap
from statsmodels.sandbox.regression.numdiff import approx_fprime1


class Model(object):
    """
    A (predictive) statistical model. The class Model itself is not to be used.

    Model lays out the methods expected of any subclass.

    Parameters
    ----------
    endog : array-like
        Endogenous response variable.
    exog : array-like
        Exogenous design.

    Notes
    -----
    `endog` and `exog` are references to any data provided.  So if the data is
    already stored in numpy arrays and it is changed then `endog` and `exog`
    will change as well.
    """

    def __init__(self, endog, exog=None):
        self._data = handle_data(endog, exog)
        self.exog = self._data.exog
        self.endog = self._data.endog

    @property
    def endog_names(self):
        return self._data.ynames

    @property
    def exog_names(self):
        return self._data.xnames

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

    def __init__(self, endog, exog=None):
        super(LikelihoodModel, self).__init__(endog, exog)
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
            full_output=True, disp=True, fargs=(), callback=None, retall=False,
            **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array-like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str {'newton','nm','bfgs','powell','cg', or 'ncg'}
            Method can be 'newton' for Newton-Raphson, 'nm' for Nelder-Mead,
            'bfgs' for Broyden-Fletcher-Goldfarb-Shanno, 'powell' for modified
            Powell's method, 'cg' for conjugate gradient, or 'ncg' for Newton-
            conjugate gradient. `method` determines which solver from
            scipy.optimize is used.  The explicit arguments in `fit` are passed
            to the solver.  Each solver has several optional arguments that are
            not the same across solvers.  See the notes section below (or
            scipy.optimize) for the available arguments.
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
        Optional arguments for the solvers (available in Results.mle_settings):

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
                """
        Hinv = None  # JP error if full_output=0, Hinv not defined
        methods = ['newton', 'nm', 'bfgs', 'powell', 'cg', 'ncg']
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
            raise ValueError("Unknown fit method %s" % method)
        method = method.lower()

        # TODO: separate args from nonarg taking score and hessian, ie.,
        # user-supplied and numerically evaluated estimate frprime doesn't take
        # args in most (any?) of the optimize function

        f = lambda params, *args: -self.loglike(params, *args)
        score = lambda params: -self.score(params)
        try:
            hess = lambda params: -self.hessian(params)
        except:
            hess = None

        fit_funcs = {
            'newton': _fit_mle_newton,
            'nm': _fit_mle_nm,  # Nelder-Mead
            'bfgs': _fit_mle_bfgs,
            'cg': _fit_mle_cg,
            'ncg': _fit_mle_ncg,
            'powell': _fit_mle_powell
        }

        if method == 'newton':
            score = lambda params: self.score(params)
            hess = lambda params: self.hessian(params)

        func = fit_funcs[method]
        xopt, retvals = func(f, score, start_params, fargs, kwargs,
                             disp=disp, maxiter=maxiter, callback=callback,
                             retall=retall, full_output=full_output,
                             hess=hess)

        if not full_output:
            xopt = retvals

        # NOTE: better just to use the Analytic Hessian here, as approximation
        # isn't great
#        if method == 'bfgs' and full_output:
#            Hinv = retvals.setdefault('Hinv', 0)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-retvals['Hessian'])
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
        return newparams

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
                 hessian=None):
    # let them be none in case user wants to use inheritance
        if loglike:
            self.loglike = loglike
        if score:
            self.score = score
        if hessian:
            self.hessian = hessian
        self.confint_dist = stats.norm

        # TODO: data structures?

        # this won't work for ru2nmnl, maybe np.ndim of a dict?
        if exog is not None:
            #try:
            self.nparams = self.df_model = (exog.shape[1]
                                            if np.ndim(exog) == 2 else 1)
        super(GenericLikelihoodModel, self).__init__(endog, exog)

    #this is redundant and not used when subclassing
    def initialize(self):
        if not self.score:  # right now score is not optional
            self.score = approx_fprime1
            if not self.hessian:
                pass
        else:   # can use approx_hess_p if we have a gradient
            if not self.hessian:
                pass

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
        return approx_fprime1(params, self.loglike, epsilon=1e-4).ravel()

    def jac(self, params, **kwds):
        '''
        Jacobian/Gradient of log-likelihood evaluated at params for each
        observation.
        '''
        kwds.setdefault('epsilon', 1e-4)
        return approx_fprime1(params, self.loglikeobs, **kwds)

    def hessian(self, params):
        '''
        Hessian of log-likelihood evaluated at params
        '''
        from statsmodels.sandbox.regression.numdiff import approx_hess
        # need options for hess (epsilon)
        return approx_hess(params, self.loglike)[0]

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
        return genericmlefit
    #fit.__doc__ += LikelihoodModel.fit.__doc__

    #------------------------------
    #TODO: the following have been moved to the result mixin class
    #      check if anything is still using them from here

    @cache_readonly
    def jacv(self):
        if not hasattr(self, '_results'):
            raise ValueError('need to call fit first')
        return self.jac(self._results.params)

    @cache_readonly
    def hessv(self):
        if not hasattr(self, '_results'):
            raise ValueError('need to call fit first')
        return self.hessian(self._results.params)

    # the following could be moved to results
    @cache_readonly
    def covjac(self):
        '''
        covariance of parameters based on loglike outer product of jacobian
        '''
##        if not hasattr(self, '_results'):
##            raise ValueError('need to call fit first')
##            #self.fit()
##        self.jacv = jacv = self.jac(self._results.params)
        jacv = self.jacv
        return np.linalg.inv(np.dot(jacv.T, jacv))

    @cache_readonly
    def covjhj(self):
        jacv = self.jacv
##        hessv = self.hessv
##        hessinv = np.linalg.inv(hessv)
##        self.hessinv = hessinv
        hessinv = self._results.cov_params()
        return np.dot(hessinv, np.dot(np.dot(jacv.T, jacv), hessinv))

    @cache_readonly
    def bsejhj(self):
        return np.sqrt(np.diag(self.covjhj))

    @cache_readonly
    def bsejac(self):
        return np.sqrt(np.diag(self.covjac))


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

    def initialize(self, model, params, **kwd):
        self.params = params
        self.model = model

    def predict(self, exog=None, *args, **kwargs):
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

    #JP: add methods that are valid generically higher up in class hierarchy
    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params()))

    def t(self, column=None):
        """
        deprecated: Return the t-statistic for a given parameter estimate.

        FutureWarning: use attribute tvalues instead, t will be removed
        in the next release

        Parameters
        ----------
        column : array-like
            The columns for which you would like the t-value.
            Note that this uses Python's indexing conventions.

        See also
        ---------
        Use t_test for more complicated t-statistics.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.tvalues
        array([ 0.17737603, -1.06951632, -4.13642736, -4.82198531, -0.22605114,
        4.01588981, -3.91080292])
        >>> results.tvalues[[1,2,4]]
        array([-1.06951632, -4.13642736, -0.22605114])
        >>> import numpy as np
        >>> results.tvalues[np.array([1,2,4]]
        array([-1.06951632, -4.13642736, -0.22605114])

        """
        import warnings
        warnings.warn("`t` will be removed in the next release, use attribute"
                      "`tvalues` instead", FutureWarning)

        if self.normalized_cov_params is None:
            raise ValueError('need covariance of parameters for computing T '
                             'statistics')

        if column is None:
            column = range(self.params.shape[0])

        column = np.asarray(column)
        _params = self.params[column]
        _cov = self.cov_params(column=column)
        if _cov.ndim == 2:
            _cov = np.diag(_cov)
        _t = _params * recipr(np.sqrt(_cov))
    # repicr drops precision for MNLogit?
        _t = _params / np.sqrt(_cov)
        return _t

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
            tmp = np.dot(r_matrix, np.dot(cov_p, np.transpose(other)))
            return tmp
        else:  #if r_matrix is None and column is None:
            return cov_p

    #TODO: make sure this works as needed for GLMs
    def t_test(self, r_matrix, q_matrix=None, cov_p=None, scale=None):
        """
        Compute a tcontrast/t-test for a row vector array of the form Rb = q

        where R is r_matrix, b = the parameter vector, and q is q_matrix.

        Parameters
        ----------
        r_matrix : array-like
            A length p row vector specifying the linear restrictions.
        q_matrix : array-like or scalar, optional
            Either a scalar or a length p row vector.
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
        >>> r[4:6] = [1,-1]
        >>> print r
        [ 0.  0.  0.  0.  1. -1.  0.]

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
        >>> T_test.t
        -4.0167754636411717
        >>> T_test.p
        0.0015163772380899498

        See also
        ---------
        t : method to get simpler t values
        f_test : for f tests

        """
        r_matrix = np.atleast_2d(np.asarray(r_matrix))
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
        if q_matrix.size > 1:
            if q_matrix.shape[0] != num_ttests:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")

        _t = _sd = None

        _effect = np.dot(r_matrix, self.params)
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(self.cov_params(r_matrix=r_matrix,
                                                  cov_p=cov_p)))
        else:
            _sd = np.sqrt(self.cov_params(r_matrix=r_matrix, cov_p=cov_p))
        _t = (_effect - q_matrix) * recipr(_sd)
        return ContrastResults(effect=_effect, t=_t, sd=_sd,
                               df_denom=self.model.df_resid)

    #TODO: untested for GLMs?
    def f_test(self, r_matrix, q_matrix=None, cov_p=None, scale=1.0,
               invcov=None):
        """
        Compute an Fcontrast/F-test for a contrast matrix.

        Here, matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        Parameters
        ----------
        r_matrix : array-like
            q x p array where q is the number of restrictions to test and
            p is the number of regressors in the full model fit.
            If q is 1 then f_test(r_matrix).fvalue is equivalent to
            the square of t_test(r_matrix).t
        q_matrix : array-like
            q x 1 array, that represents the sum of each linear restriction.
            Default is all zeros for each restriction.
        scale : float, optional
            Default is 1.0 for no scaling.
        invcov : array-like, optional
            A qxq matrix to specify an inverse covariance
            matrix based on a restrictions matrix.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[:-1,:]

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

        >>> B = np.array(([0,1,-1,0,0,0,0],[0,0,0,0,1,-1,0]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print results.f_test(B)
        <F contrast: F=9.740461873303655, p=0.00560528853174, df_denom=9,
         df_num=2>

        See also
        --------
        statsmodels.contrasts
        statsmodels.model.t_test

        """
        r_matrix = np.asarray(r_matrix)
        r_matrix = np.atleast_2d(r_matrix)

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
            invcov = np.linalg.inv(self.cov_params(r_matrix=r_matrix,
                                                   cov_p=cov_p))
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
        array([[ -1.77029035e+02,   2.07152780e+02],
        [ -1.11581102e-01,   3.99427438e-02],
        [ -3.12506664e+00,  -9.15392966e-01],
        [ -1.51794870e+00,  -5.48505034e-01],
        [ -5.62517214e-01,   4.60309003e-01],
        [  7.98787515e+02,   2.85951541e+03],
        [ -5.49652948e+06,  -1.46798779e+06]])

        >>> results.conf_int(cols=(1,2))
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

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)


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
        #self.df_model = model.df_model
        #self.df_resid = model.df_resid
        self.endog = model.endog
        self.exog = model.exog
        self.nobs = model.endog.shape[0]
        self._cache = resettable_cache()
        self.__dict__.update(mlefit.__dict__)
