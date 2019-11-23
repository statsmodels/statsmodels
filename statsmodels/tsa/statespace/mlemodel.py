# -*- coding: utf-8 -*-
"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from statsmodels.compat.python import long

from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
                                       approx_fprime_cs, approx_fprime)
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic

import statsmodels.base.wrapper as wrap

import statsmodels.genmod._prediction as pred
from statsmodels.genmod.families.links import identity

import statsmodels.tsa.base.tsa_model as tsbase

from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU

if bytes != str:
    # PY3
    unicode = str


def _handle_args(names, defaults, *args, **kwargs):
    output_args = []
    # We need to handle positional arguments in two ways, in case this was
    # called by a Scipy optimization routine
    if len(args) > 0:
        # the fit() method will pass a dictionary
        if isinstance(args[0], dict):
            flags = args[0]
        # otherwise, a user may have just used positional arguments...
        else:
            flags = dict(zip(names, args))
        for i in range(len(names)):
            output_args.append(flags.get(names[i], defaults[i]))

        for name, value in flags.items():
            if name in kwargs:
                raise TypeError("loglike() got multiple values for keyword"
                                " argument '%s'" % name)
    else:
        for i in range(len(names)):
            output_args.append(kwargs.pop(names[i], defaults[i]))

    return tuple(output_args) + (kwargs,)


class MLEModel(tsbase.TimeSeriesModel):
    r"""
    State space model for maximum likelihood estimation

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k. Default is no
        exogenous regressors.
    dates : array-like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    ssm : statsmodels.tsa.statespace.kalman_filter.KalmanFilter
        Underlying state space representation.

    Notes
    -----
    This class wraps the state space model with Kalman filtering to add in
    functionality for maximum likelihood estimation. In particular, it adds
    the concept of updating the state space representation based on a defined
    set of parameters, through the `update` method or `updater` attribute (see
    below for more details on which to use when), and it adds a `fit` method
    which uses a numerical optimizer to select the parameters that maximize
    the likelihood of the model.

    The `start_params` `update` method must be overridden in the
    child class (and the `transform` and `untransform` methods, if needed).

    See Also
    --------
    MLEResults
    statsmodels.tsa.statespace.kalman_filter.KalmanFilter
    statsmodels.tsa.statespace.representation.Representation
    """

    def __init__(self, endog, k_states, exog=None, dates=None, freq=None,
                 **kwargs):
        # Initialize the model base
        super(MLEModel, self).__init__(endog=endog, exog=exog,
                                       dates=dates, freq=freq,
                                       missing='none')

        # Store kwargs to recreate model
        self._init_kwargs = kwargs

        # Prepared the endog array: C-ordered, shape=(nobs x k_endog)
        self.endog, self.exog = self.prepare_data()

        # Dimensions
        self.nobs = self.endog.shape[0]
        self.k_states = k_states

        # Initialize the state-space representation
        self.initialize_statespace(**kwargs)

    def prepare_data(self):
        """
        Prepare data for use in the state space representation
        """
        endog = np.array(self.data.orig_endog, order='C')
        exog = self.data.orig_exog
        if exog is not None:
            exog = np.array(exog)

        # Base class may allow 1-dim data, whereas we need 2-dim
        if endog.ndim == 1:
            endog.shape = (endog.shape[0], 1)  # this will be C-contiguous

        return endog, exog

    def initialize_statespace(self, **kwargs):
        """
        Initialize the state space representation

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the state space class
            constructor.

        """
        # (Now self.endog is C-ordered and in long format (nobs x k_endog). To
        # get F-ordered and in wide format just need to transpose)
        endog = self.endog.T

        # Instantiate the state space object
        self.ssm = SimulationSmoother(endog.shape[0], self.k_states,
                                      nobs=endog.shape[1], **kwargs)
        # Bind the data to the model
        self.ssm.bind(endog)

        # Other dimensions, now that `ssm` is available
        self.k_endog = self.ssm.k_endog

    def __setitem__(self, key, value):
        return self.ssm.__setitem__(key, value)

    def __getitem__(self, key):
        return self.ssm.__getitem__(key)

    def set_filter_method(self, filter_method=None, **kwargs):
        """
        Set the filtering method

        The filtering method controls aspects of which Kalman filtering
        approach will be used.

        Parameters
        ----------
        filter_method : integer, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_filter_method(filter_method, **kwargs)

    def set_inversion_method(self, inversion_method=None, **kwargs):
        """
        Set the inversion method

        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.

        Parameters
        ----------
        inversion_method : integer, optional
            Bitmask value to set the inversion method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the inversion method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_inversion_method(inversion_method, **kwargs)

    def set_stability_method(self, stability_method=None, **kwargs):
        """
        Set the numerical stability method

        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.

        Parameters
        ----------
        stability_method : integer, optional
            Bitmask value to set the stability method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the stability method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_stability_method(stability_method, **kwargs)

    def set_conserve_memory(self, conserve_memory=None, **kwargs):
        """
        Set the memory conservation method

        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.

        Parameters
        ----------
        conserve_memory : integer, optional
            Bitmask value to set the memory conservation method to. See notes
            for details.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanFilter` class for details.
        """
        self.ssm.set_conserve_memory(conserve_memory, **kwargs)

    def set_smoother_output(self, smoother_output=None, **kwargs):
        """
        Set the smoother output

        The smoother can produce several types of results. The smoother output
        variable controls which are calculated and returned.

        Parameters
        ----------
        smoother_output : integer, optional
            Bitmask value to set the smoother output to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the smoother output by
            setting individual boolean flags.

        Notes
        -----
        This method is rarely used. See the corresponding function in the
        `KalmanSmoother` class for details.
        """
        self.ssm.set_smoother_output(smoother_output, **kwargs)

    def initialize_known(self, initial_state, initial_state_cov):
        """Initialize known"""
        self.ssm.initialize_known(initial_state, initial_state_cov)

    def initialize_approximate_diffuse(self, variance=None):
        """Initialize approximate diffuse"""
        self.ssm.initialize_approximate_diffuse(variance)

    def initialize_stationary(self):
        """Initialize stationary"""
        self.ssm.initialize_stationary()

    @property
    def initialization(self):
        return self.ssm.initialization

    @initialization.setter
    def initialization(self, value):
        self.ssm.initialization = value

    @property
    def initial_variance(self):
        return self.ssm.initial_variance

    @initial_variance.setter
    def initial_variance(self, value):
        self.ssm.initial_variance = value

    @property
    def loglikelihood_burn(self):
        return self.ssm.loglikelihood_burn

    @loglikelihood_burn.setter
    def loglikelihood_burn(self, value):
        self.ssm.loglikelihood_burn = value

    @property
    def tolerance(self):
        return self.ssm.tolerance

    @tolerance.setter
    def tolerance(self, value):
        self.ssm.tolerance = value

    def fit(self, start_params=None, transformed=True,
            cov_type='opg', cov_kwds=None, method='lbfgs', maxiter=50,
            full_output=1, disp=5, callback=None, return_params=False,
            optim_score=None, optim_complex_step=None, optim_hessian=None,
            flags=None, **kwargs):
        """
        Fits the model by maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : boolean, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presense of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : boolean, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : boolean, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : boolean, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : boolean, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : boolean, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        optim_score : {'harvey', 'approx'} or None, optional
            The method by which the score vector is calculated. 'harvey' uses
            the method from Harvey (1989), 'approx' uses either finite
            difference or complex step differentiation depending upon the
            value of `optim_complex_step`, and None uses the built-in gradient
            approximation of the optimizer. Default is None. This keyword is
            only relevant if the optimization method uses the score.
        optim_complex_step : bool, optional
            Whether or not to use complex step differentiation when
            approximating the score; if False, finite difference approximation
            is used. Default is True. This keyword is only relevant if
            `optim_score` is set to 'harvey' or 'approx'.
        optim_hessian : {'opg','oim','approx'}, optional
            The method by which the Hessian is numerically approximated. 'opg'
            uses outer product of gradients, 'oim' uses the information
            matrix formula from Harvey (1989), and 'approx' uses numerical
            approximation. This keyword is only relevant if the
            optimization method uses the Hessian matrix.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        MLEResults

        See Also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        MLEResults
        """
        if start_params is None:
            start_params = self.start_params
            transformed = True

        # Update the score method
        if optim_score is None and method == 'lbfgs':
            kwargs.setdefault('approx_grad', True)
            kwargs.setdefault('epsilon', 1e-5)
        elif optim_score is None:
            optim_score = 'approx'

        # Check for complex step differentiation
        if optim_complex_step is None:
            optim_complex_step = not self.ssm._complex_endog
        elif optim_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        # Unconstrain the starting parameters
        if transformed:
            start_params = self.untransform_params(np.array(start_params))

        # Maximum likelihood estimation
        if flags is None:
            flags = {}
        flags.update({
            'transformed': False,
            'score_method': optim_score,
            'approx_complex_step': optim_complex_step
        })
        if optim_hessian is not None:
            flags['hessian_method'] = optim_hessian
        fargs = (flags,)
        mlefit = super(MLEModel, self).fit(start_params, method=method,
                                           fargs=fargs,
                                           maxiter=maxiter,
                                           full_output=full_output,
                                           disp=disp, callback=callback,
                                           skip_hessian=True, **kwargs)

        # Just return the fitted parameters if requested
        if return_params:
            return self.transform_params(mlefit.params)
        # Otherwise construct the results class if desired
        else:
            res = self.smooth(mlefit.params, transformed=False,
                              cov_type=cov_type, cov_kwds=cov_kwds)

            res.mlefit = mlefit
            res.mle_retvals = mlefit.mle_retvals
            res.mle_settings = mlefit.mle_settings

            return res

    @property
    def _res_classes(self):
        return {'fit': (MLEResults, MLEResultsWrapper)}

    def _wrap_results(self, params, result, return_raw, cov_type=None,
                      cov_kwds=None, results_class=None, wrapper_class=None):
        if not return_raw:
            # Wrap in a results object
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds

            if results_class is None:
                results_class = self._res_classes['fit'][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes['fit'][1]

            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def filter(self, params, transformed=True, complex_step=False,
               cov_type=None, cov_kwds=None, return_ssm=False,
               results_class=None, results_wrapper_class=None, **kwargs):
        """
        Kalman filtering

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : boolean,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
        self.update(params, transformed=True, complex_step=complex_step)

        # Save the parameter names
        self.data.param_names = self.param_names

        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        # Get the state space output
        result = self.ssm.filter(complex_step=complex_step, **kwargs)

        # Wrap in a results object
        return self._wrap_results(params, result, return_ssm, cov_type,
                                  cov_kwds, results_class,
                                  results_wrapper_class)

    def smooth(self, params, transformed=True, complex_step=False,
               cov_type=None, cov_kwds=None, return_ssm=False,
               results_class=None, results_wrapper_class=None, **kwargs):
        """
        Kalman smoothing

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : boolean,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
        self.update(params, transformed=True, complex_step=complex_step)

        # Save the parameter names
        self.data.param_names = self.param_names

        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        # Get the state space output
        result = self.ssm.smooth(complex_step=complex_step, **kwargs)

        # Wrap in a results object
        return self._wrap_results(params, result, return_ssm, cov_type,
                                  cov_kwds, results_class,
                                  results_wrapper_class)

    _loglike_param_names = ['transformed', 'complex_step']
    _loglike_param_defaults = [True, False]

    def loglike(self, params, *args, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.

        See Also
        --------
        update : modifies the internal state of the state space model to
                 reflect new params
        """
        transformed, complex_step, kwargs = _handle_args(
            MLEModel._loglike_param_names, MLEModel._loglike_param_defaults,
            *args, **kwargs)

        if not transformed:
            params = self.transform_params(params)

        self.update(params, transformed=True, complex_step=complex_step)

        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)

        # Koopman, Shephard, and Doornik recommend maximizing the average
        # likelihood to avoid scale issues, but the averaging is done
        # automatically in the base model `fit` method
        return loglike

    def loglikeobs(self, params, transformed=True, complex_step=False,
                   **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this is done automatically by the base Model fit method.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.

        See Also
        --------
        update : modifies the internal state of the Model to reflect new params
        """
        if not transformed:
            params = self.transform_params(params)

        # If we're using complex-step differentiation, then we can't use
        # Cholesky factorization
        if complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        self.update(params, transformed=True, complex_step=complex_step)

        return self.ssm.loglikeobs(complex_step=complex_step, **kwargs)

    def simulation_smoother(self, simulation_output=None, **kwargs):
        r"""
        Retrieve a simulation smoother for the state space model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        **kwargs
            Additional keyword arguments, used to set the simulation output.
            See `set_simulation_output` for more details.

        Returns
        -------
        SimulationSmoothResults
        """
        return self.ssm.simulation_smoother(
            simulation_output=simulation_output, **kwargs)

    def _forecasts_error_partial_derivatives(self, params, transformed=True,
                                             approx_complex_step=None,
                                             approx_centered=False,
                                             res=None, **kwargs):
        params = np.array(params, ndmin=1)

        # We can't use complex-step differentiation with non-transformed
        # parameters
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the observed_information_matrix"
                             " with untransformed parameters.")

        # If we're using complex-step differentiation, then we can't use
        # Cholesky factorization
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU

        # Get values at the params themselves
        if res is None:
            self.update(params, transformed=transformed,
                        complex_step=approx_complex_step)
            res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)

        # Setup
        n = len(params)

        # Compute partial derivatives w.r.t. forecast error and forecast
        # error covariance
        partials_forecasts_error = (
            np.zeros((self.k_endog, self.nobs, n))
        )
        partials_forecasts_error_cov = (
            np.zeros((self.k_endog, self.k_endog, self.nobs, n))
        )
        if approx_complex_step:
            epsilon = _get_epsilon(params, 2, None, n)
            increments = np.identity(n) * 1j * epsilon

            for i, ih in enumerate(increments):
                self.update(params + ih, transformed=transformed,
                            complex_step=True)
                _res = self.ssm.filter(complex_step=True, **kwargs)

                partials_forecasts_error[:, :, i] = (
                    _res.forecasts_error.imag / epsilon[i]
                )

                partials_forecasts_error_cov[:, :, :, i] = (
                    _res.forecasts_error_cov.imag / epsilon[i]
                )
        elif not approx_centered:
            epsilon = _get_epsilon(params, 2, None, n)
            ei = np.zeros((n,), float)
            for i in range(n):
                ei[i] = epsilon[i]
                self.update(params + ei, transformed=transformed,
                            complex_step=False)
                _res = self.ssm.filter(complex_step=False, **kwargs)

                partials_forecasts_error[:, :, i] = (
                    _res.forecasts_error - res.forecasts_error) / epsilon[i]

                partials_forecasts_error_cov[:, :, :, i] = (
                    _res.forecasts_error_cov -
                    res.forecasts_error_cov) / epsilon[i]
                ei[i] = 0.0
        else:
            epsilon = _get_epsilon(params, 3, None, n) / 2.
            ei = np.zeros((n,), float)
            for i in range(n):
                ei[i] = epsilon[i]

                self.update(params + ei, transformed=transformed,
                            complex_step=False)
                _res1 = self.ssm.filter(complex_step=False, **kwargs)

                self.update(params - ei, transformed=transformed,
                            complex_step=False)
                _res2 = self.ssm.filter(complex_step=False, **kwargs)

                partials_forecasts_error[:, :, i] = (
                    (_res1.forecasts_error - _res2.forecasts_error) /
                    (2 * epsilon[i]))

                partials_forecasts_error_cov[:, :, :, i] = (
                    (_res1.forecasts_error_cov - _res2.forecasts_error_cov) /
                    (2 * epsilon[i]))

                ei[i] = 0.0

        return partials_forecasts_error, partials_forecasts_error_cov

    def observed_information_matrix(self, params, transformed=True,
                                    approx_complex_step=None,
                                    approx_centered=False, **kwargs):
        """
        Observed information matrix

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        This method is from Harvey (1989), which shows that the information
        matrix only depends on terms from the gradient. This implementation is
        partially analytic and partially numeric approximation, therefore,
        because it uses the analytic formula for the information matrix, with
        numerically computed elements of the gradient.

        References
        ----------
        Harvey, Andrew C. 1990.
        Forecasting, Structural Time Series Models and the Kalman Filter.
        Cambridge University Press.

        """
        params = np.array(params, ndmin=1)

        # Setup
        n = len(params)

        # We can't use complex-step differentiation with non-transformed
        # parameters
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the observed_information_matrix"
                             " with untransformed parameters.")

        # Get values at the params themselves
        self.update(params, transformed=transformed,
                    complex_step=approx_complex_step)
        # If we're using complex-step differentiation, then we can't use
        # Cholesky factorization
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)
        dtype = self.ssm.dtype

        # Save this for inversion later
        inv_forecasts_error_cov = res.forecasts_error_cov.copy()

        partials_forecasts_error, partials_forecasts_error_cov = (
            self._forecasts_error_partial_derivatives(
                params, transformed=transformed,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, res=res, **kwargs))

        # Compute the information matrix
        tmp = np.zeros((self.k_endog, self.k_endog, self.nobs, n), dtype=dtype)

        information_matrix = np.zeros((n, n), dtype=dtype)
        for t in range(self.ssm.loglikelihood_burn, self.nobs):
            inv_forecasts_error_cov[:, :, t] = (
                np.linalg.inv(res.forecasts_error_cov[:, :, t])
            )
            for i in range(n):
                tmp[:, :, t, i] = np.dot(
                    inv_forecasts_error_cov[:, :, t],
                    partials_forecasts_error_cov[:, :, t, i]
                )
            for i in range(n):
                for j in range(n):
                    information_matrix[i, j] += (
                        0.5 * np.trace(np.dot(tmp[:, :, t, i],
                                              tmp[:, :, t, j]))
                    )
                    information_matrix[i, j] += np.inner(
                        partials_forecasts_error[:, t, i],
                        np.dot(inv_forecasts_error_cov[:, :, t],
                               partials_forecasts_error[:, t, j])
                    )
        return information_matrix / (self.nobs - self.ssm.loglikelihood_burn)

    def opg_information_matrix(self, params, transformed=True,
                               approx_complex_step=None, **kwargs):
        """
        Outer product of gradients information matrix

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional arguments to the `loglikeobs` method.

        References
        ----------
        Berndt, Ernst R., Bronwyn Hall, Robert Hall, and Jerry Hausman. 1974.
        Estimation and Inference in Nonlinear Structural Models.
        NBER Chapters. National Bureau of Economic Research, Inc.

        """
        # We can't use complex-step differentiation with non-transformed
        # parameters
        if approx_complex_step is None:
            approx_complex_step = transformed
        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the observed_information_matrix"
                             " with untransformed parameters.")

        score_obs = self.score_obs(params, transformed=transformed,
                                   approx_complex_step=approx_complex_step,
                                   **kwargs).transpose()
        return (
            np.inner(score_obs, score_obs) /
            (self.nobs - self.ssm.loglikelihood_burn)
        )

    def _score_complex_step(self, params, **kwargs):
        # the default epsilon can be too small
        # inversion_method = INVERT_UNIVARIATE | SOLVE_LU
        epsilon = _get_epsilon(params, 2., None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        return approx_fprime_cs(params, self.loglike, epsilon=epsilon,
                                kwargs=kwargs)

    def _score_finite_difference(self, params, approx_centered=False,
                                 **kwargs):
        kwargs['transformed'] = True
        return approx_fprime(params, self.loglike, kwargs=kwargs,
                             centered=approx_centered)

    def _score_harvey(self, params, approx_complex_step=True, **kwargs):
        score_obs = self._score_obs_harvey(
            params, approx_complex_step=approx_complex_step, **kwargs)
        return np.sum(score_obs, axis=0)

    def _score_obs_harvey(self, params, approx_complex_step=True,
                          approx_centered=False, **kwargs):
        """
        Score

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        This method is from Harvey (1989), section 3.4.5

        References
        ----------
        Harvey, Andrew C. 1990.
        Forecasting, Structural Time Series Models and the Kalman Filter.
        Cambridge University Press.

        """
        params = np.array(params, ndmin=1)
        n = len(params)

        # Get values at the params themselves
        self.update(params, transformed=True, complex_step=approx_complex_step)
        if approx_complex_step:
            kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
        res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)

        # Get forecasts error partials
        partials_forecasts_error, partials_forecasts_error_cov = (
            self._forecasts_error_partial_derivatives(
                params, transformed=True,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, res=res, **kwargs))

        # Compute partial derivatives w.r.t. likelihood function
        partials = np.zeros((self.nobs, n))
        k_endog = self.k_endog
        for t in range(self.nobs):
            inv_forecasts_error_cov = np.linalg.inv(
                    res.forecasts_error_cov[:, :, t])

            for i in range(n):
                partials[t, i] += np.trace(np.dot(
                    np.dot(inv_forecasts_error_cov,
                           partials_forecasts_error_cov[:, :, t, i]),
                    (np.eye(k_endog) -
                     np.dot(inv_forecasts_error_cov,
                            np.outer(res.forecasts_error[:, t],
                                     res.forecasts_error[:, t])))))
                # 2 * dv / di * F^{-1} v_t
                # where x = F^{-1} v_t or F x = v
                partials[t, i] += 2 * np.dot(
                    partials_forecasts_error[:, t, i],
                    np.dot(inv_forecasts_error_cov, res.forecasts_error[:, t]))

        return -partials / 2.

    _score_param_names = ['transformed', 'score_method',
                          'approx_complex_step', 'approx_centered']
    _score_param_defaults = [True, 'approx', None, False]

    def score(self, params, *args, **kwargs):
        """
        Compute the score function at params.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        args
            Additional positional arguments to the `loglike` method.
        kwargs
            Additional keyword arguments to the `loglike` method.

        Returns
        -------
        score : array
            Score, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglike` method.

        Both args and kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        params = np.array(params, ndmin=1)

        transformed, method, approx_complex_step, approx_centered, kwargs = (
            _handle_args(MLEModel._score_param_names,
                         MLEModel._score_param_defaults, *args, **kwargs))
        # For fit() calls, the method is called 'score_method' (to distinguish
        # it from the method used for fit) but generally in kwargs the method
        # will just be called 'method'
        if 'method' in kwargs:
            method = kwargs.pop('method')

        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        if not transformed:
            transform_score = self.transform_jacobian(params)
            params = self.transform_params(params)

        if method == 'harvey':
            score = self._score_harvey(
                params, approx_complex_step=approx_complex_step, **kwargs)
        elif method == 'approx' and approx_complex_step:
            score = self._score_complex_step(params, **kwargs)
        elif method == 'approx':
            score = self._score_finite_difference(
                params, approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid score method.')

        if not transformed:
            score = np.dot(transform_score, score)

        return score

    def score_obs(self, params, method='approx', transformed=True,
                  approx_complex_step=None, approx_centered=False, **kwargs):
        """
        Compute the score per observation, evaluated at params

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        kwargs
            Additional arguments to the `loglike` method.

        Returns
        -------
        score : array
            Score per observation, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglikeobs` method.
        """
        params = np.array(params, ndmin=1)

        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the score at each observation"
                             " with untransformed parameters.")

        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        if method == 'harvey':
            score = self._score_obs_harvey(
                params, transformed=transformed,
                approx_complex_step=approx_complex_step, **kwargs)
        elif method == 'approx' and approx_complex_step:
            # the default epsilon can be too small
            epsilon = _get_epsilon(params, 2., None, len(params))
            kwargs['complex_step'] = True
            kwargs['transformed'] = True
            score = approx_fprime_cs(params, self.loglikeobs, epsilon=epsilon,
                                     kwargs=kwargs)
        elif method == 'approx':
            kwargs['transformed'] = transformed
            score = approx_fprime(params, self.loglikeobs, kwargs=kwargs,
                                  centered=approx_centered)
        else:
            raise NotImplementedError('Invalid scoreobs method.')

        return score

    _hessian_param_names = ['transformed', 'hessian_method',
                            'approx_complex_step', 'approx_centered']
    _hessian_param_defaults = [True, 'approx', None, False]

    def hessian(self, params, *args, **kwargs):
        r"""
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the hessian.
        args
            Additional positional arguments to the `loglike` method.
        kwargs
            Additional keyword arguments to the `loglike` method.

        Returns
        -------
        hessian : array
            Hessian matrix evaluated at `params`

        Notes
        -----
        This is a numerical approximation.

        Both args and kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        transformed, method, approx_complex_step, approx_centered, kwargs = (
            _handle_args(MLEModel._hessian_param_names,
                         MLEModel._hessian_param_defaults,
                         *args, **kwargs))
        # For fit() calls, the method is called 'hessian_method' (to
        # distinguish it from the method used for fit) but generally in kwargs
        # the method will just be called 'method'
        if 'method' in kwargs:
            method = kwargs.pop('method')

        if not transformed and approx_complex_step:
            raise ValueError("Cannot use complex-step approximations to"
                             " calculate the hessian with untransformed"
                             " parameters.")

        if approx_complex_step is None:
            approx_complex_step = not self.ssm._complex_endog
        if approx_complex_step and self.ssm._complex_endog:
            raise ValueError('Cannot use complex step derivatives when data'
                             ' or parameters are complex.')

        if method == 'oim':
            hessian = self._hessian_oim(
                params, transformed=transformed,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, **kwargs)
        elif method == 'opg':
            hessian = self._hessian_opg(
                params, transformed=transformed,
                approx_complex_step=approx_complex_step,
                approx_centered=approx_centered, **kwargs)
        elif method == 'approx' and approx_complex_step:
            hessian = self._hessian_complex_step(
                params, transformed=transformed, **kwargs)
        elif method == 'approx':
            hessian = self._hessian_finite_difference(
                params, transformed=transformed,
                approx_centered=approx_centered, **kwargs)
        else:
            raise NotImplementedError('Invalid Hessian calculation method.')
        return hessian

    def _hessian_oim(self, params, **kwargs):
        """
        Hessian matrix computed using the Harvey (1989) information matrix
        """
        return -self.observed_information_matrix(params, **kwargs)

    def _hessian_opg(self, params, **kwargs):
        """
        Hessian matrix computed using the outer product of gradients
        information matrix
        """
        return -self.opg_information_matrix(params, **kwargs)

    def _hessian_finite_difference(self, params, approx_centered=False,
                                   **kwargs):
        params = np.array(params, ndmin=1)

        warnings.warn('Calculation of the Hessian using finite differences'
                      ' is usually subject to substantial approximation'
                      ' errors.', PrecisionWarning)

        if not approx_centered:
            epsilon = _get_epsilon(params, 3, None, len(params))
        else:
            epsilon = _get_epsilon(params, 4, None, len(params)) / 2
        hessian = approx_fprime(params, self._score_finite_difference,
                                epsilon=epsilon, kwargs=kwargs,
                                centered=approx_centered)

        return hessian / (self.nobs - self.ssm.loglikelihood_burn)

    def _hessian_complex_step(self, params, **kwargs):
        """
        Hessian matrix computed by second-order complex-step differentiation
        on the `loglike` function.
        """
        # the default epsilon can be too small
        epsilon = _get_epsilon(params, 3., None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        hessian = approx_hess_cs(
            params, self.loglike, epsilon=epsilon, kwargs=kwargs)

        return hessian / (self.nobs - self.ssm.loglikelihood_burn)

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        if hasattr(self, '_start_params'):
            return self._start_params
        else:
            raise NotImplementedError

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        if hasattr(self, '_param_names'):
            return self._param_names
        else:
            try:
                names = ['param.%d' % i for i in range(len(self.start_params))]
            except NotImplementedError:
                names = []
            return names

    def transform_jacobian(self, unconstrained, approx_centered=False):
        """
        Jacobian matrix for the parameter transformation function

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Returns
        -------
        jacobian : array
            Jacobian matrix of the transformation, evaluated at `unconstrained`

        Notes
        -----
        This is a numerical approximation using finite differences. Note that
        in general complex step methods cannot be used because it is not
        guaranteed that the `transform_params` method is a real function (e.g.
        if Cholesky decomposition is used).

        See Also
        --------
        transform_params
        """
        return approx_fprime(unconstrained, self.transform_params,
                             centered=approx_centered)

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evalation.

        Notes
        -----
        This is a noop in the base class, subclasses should override where
        appropriate.
        """
        return np.array(unconstrained, ndmin=1)

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evalution, to be
            transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Notes
        -----
        This is a noop in the base class, subclasses should override where
        appropriate.
        """
        return np.array(constrained, ndmin=1)

    def update(self, params, transformed=True, complex_step=False):
        """
        Update the parameters of the model

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : boolean, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True.

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        Since Model is a base class, this method should be overridden by
        subclasses to perform actual updating steps.
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        return params

    def simulate(self, params, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None):
        r"""
        Simulate a new time series following the state space model

        Parameters
        ----------
        params : array_like
            Array of model parameters.
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the state vector at time zero, which should
            be shaped (`k_states` x 1), where `k_states` is the same as in the
            state space model. If unspecified, but the model has been
            initialized, then that initialization is used. If unspecified and
            the model has not been initialized, then a vector of zeros is used.
            Note that this is not included in the returned `simulated_states`
            array.

        Returns
        -------
        simulated_obs : array
            An (nsimulations x k_endog) array of simulated observations.
        """
        self.update(params)

        simulated_obs, simulated_states = self.ssm.simulate(
            nsimulations, measurement_shocks, state_shocks, initial_state)

        # Simulated obs is (nobs x k_endog); don't want to squeeze in
        # case of nsimulations = 1
        if simulated_obs.shape[1] == 1:
            simulated_obs = simulated_obs[:, 0]
        return simulated_obs

    def impulse_responses(self, params, steps=1, impulse=0,
                          orthogonalized=False, cumulative=False, **kwargs):
        """
        Impulse response function

        Parameters
        ----------
        params : array_like
            Array of model parameters.
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 1. Note that the initial impulse is not counted as a
            step, so if `steps=1`, the output will have 2 entries.
        impulse : int or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1`. Alternatively, a custom impulse vector may be
            provided; must be shaped `k_posdef x 1`.
        orthogonalized : boolean, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : boolean, optional
            Whether or not to return cumulative impulse responses. Default is
            False.
        **kwargs
            If the model is time-varying and `steps` is greater than the number
            of observations, any of the state space representation matrices
            that are time-varying must have updated values provided for the
            out-of-sample steps.
            For example, if `design` is a time-varying component, `nobs` is 10,
            and `steps` is 15, a (`k_endog` x `k_states` x 5) matrix must be
            provided with the new design matrix values.

        Returns
        -------
        impulse_responses : array
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. A (steps + 1 x k_endog) array.

        Notes
        -----
        Intercepts in the measurement and state equation are ignored when
        calculating impulse responses.

        """
        self.update(params)
        irfs = self.ssm.impulse_responses(
            steps, impulse, orthogonalized, cumulative, **kwargs)

        # IRF is (nobs x k_endog); don't want to squeeze in case of steps = 1
        if irfs.shape[1] == 1:
            irfs = irfs[:, 0]

        return irfs

    @classmethod
    def from_formula(cls, formula, data, subset=None):
        """
        Not implemented for state space models
        """
        raise NotImplementedError


class MLEResults(tsbase.TimeSeriesModelResults):
    r"""
    Class to hold results from fitting a state space model.

    Parameters
    ----------
    model : MLEModel instance
        The fitted model instance
    params : array
        Fitted parameters
    filter_results : KalmanFilter instance
        The underlying state space model and Kalman filter output

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : KalmanFilter instance
        The underlying state space model and Kalman filter output
    nobs : float
        The number of observations used to fit the model.
    params : array
        The parameters of the model.
    scale : float
        This is currently set to 1.0 unless the model uses concentrated
        filtering.

    See Also
    --------
    MLEModel
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.representation.FrozenRepresentation
    """
    def __init__(self, model, params, results, cov_type='opg',
                 cov_kwds=None, **kwargs):
        self.data = model.data
        scale = results.scale

        tsbase.TimeSeriesModelResults.__init__(self, model, params,
                                               normalized_cov_params=None,
                                               scale=scale)

        # Save the state space representation output
        self.filter_results = results
        if isinstance(results, SmootherResults):
            self.smoother_results = results
        else:
            self.smoother_results = None

        # Dimensions
        self.nobs = self.filter_results.nobs
        self.nobs_diffuse = self.filter_results.nobs_diffuse
        if self.nobs_diffuse > 0 and self.loglikelihood_burn > 0:
            warnings.warn('Care should be used when applying a loglikelihood'
                          ' burn to a model with exact diffuse initialization.'
                          ' Some results objects, e.g. degrees of freedom,'
                          ' expect only one of the two to be set.')
        # This only excludes explicitly burned (usually approximate diffuse)
        # periods but does not exclude approximate diffuse periods. This is
        # because the loglikelihood remains valid for the initial periods in
        # the exact diffuse case (see DK, 2012, section 7.2) and so also do
        # e.g. information criteria (see DK, 2012, section 7.4) and the score
        # vector (see DK, 2012, section 7.3.3, equation 7.15).
        # However, other objects should be excluded in the diffuse periods
        # (e.g. the diffuse forecast errors, so in some cases a different
        # nobs_effective will have to be computed and used)
        self.nobs_effective = self.nobs - self.loglikelihood_burn

        P = self.filter_results.initial_diffuse_state_cov
        self.k_diffuse_states = 0 if P is None else np.sum(np.diagonal(P) == 1)

        # Degrees of freedom (see DK 2012, section 7.4)
        self.df_model = (self.params.size + self.k_diffuse_states +
                         self.filter_results.filter_concentrated)
        self.df_resid = self.nobs_effective - self.df_model

        # Setup covariance matrix notes dictionary
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = cov_type

        # Setup the cache
        self._cache = {}

        # Handle covariance matrix calculation
        if cov_kwds is None:
            cov_kwds = {}
        self._cov_approx_complex_step = (
            cov_kwds.pop('approx_complex_step', True))
        self._cov_approx_centered = cov_kwds.pop('approx_centered', False)
        try:
            self._rank = None
            self._get_robustcov_results(cov_type=cov_type, use_self=True,
                                        **cov_kwds)
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds['cov_type'] = (
                'Covariance matrix could not be calculated: singular.'
                ' information matrix.')
        self.model.update(self.params)

        # References of filter and smoother output
        extra_arrays = [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']
        for name in extra_arrays:
            setattr(self, name, getattr(self.filter_results, name, None))

        # Handle removing data
        self._data_attr_model = getattr(self, '_data_attr_model', [])
        self._data_attr_model.extend(['ssm'])
        self._data_attr.extend(extra_arrays)
        self._data_attr.extend(['filter_results', 'smoother_results'])
        self.data_in_cache = getattr(self, 'data_in_cache', [])
        self.data_in_cache.extend([])

    def _get_robustcov_results(self, cov_type='opg', **kwargs):
        """
        Create new results instance with specified covariance estimator as
        default

        Note: creating new results instance currently not supported.

        Parameters
        ----------
        cov_type : string
            the type of covariance matrix estimator to use. See Notes below
        kwargs : depends on cov_type
            Required or optional arguments for covariance calculation.
            See Notes below.

        Returns
        -------
        results : results instance
            This method creates a new results instance with the requested
            covariance as the default covariance of the parameters.
            Inferential statistics like p-values and hypothesis tests will be
            based on this covariance matrix.

        Notes
        -----
        The following covariance types and required or optional arguments are
        currently available:

        - 'opg' for the outer product of gradient estimator
        - 'oim' for the observed information matrix estimator, calculated
          using the method of Harvey (1989)
        - 'approx' for the observed information matrix estimator,
          calculated using a numerical approximation of the Hessian matrix.
          Uses complex step approximation by default, or uses finite
          differences if `approx_complex_step=False` in the `cov_kwds`
          dictionary.
        - 'robust' for an approximate (quasi-maximum likelihood) covariance
          matrix that may be valid even in the presense of some
          misspecifications. Intermediate calculations use the 'oim'
          method.
        - 'robust_approx' is the same as 'robust' except that the
          intermediate calculations use the 'approx' method.
        - 'none' for no covariance matrix calculation.
        """
        from statsmodels.base.covtype import descriptions

        use_self = kwargs.pop('use_self', False)
        if use_self:
            res = self
        else:
            raise NotImplementedError
            res = self.__class__(
                self.model, self.params,
                normalized_cov_params=self.normalized_cov_params,
                scale=self.scale)

        # Set the new covariance type
        res.cov_type = cov_type
        res.cov_kwds = {}

        # Calculate the new covariance matrix
        approx_complex_step = self._cov_approx_complex_step
        if approx_complex_step:
            approx_type_str = 'complex-step'
        elif self._cov_approx_centered:
            approx_type_str = 'centered finite differences'
        else:
            approx_type_str = 'finite differences'

        k_params = len(self.params)
        if k_params == 0:
            res.cov_params_default = np.zeros((0, 0))
            res._rank = 0
            res.cov_kwds['description'] = 'No parameters estimated.'
        elif cov_type == 'custom':
            res.cov_type = kwargs['custom_cov_type']
            res.cov_params_default = kwargs['custom_cov_params']
            res.cov_kwds['description'] = kwargs['custom_description']
            res._rank = np.linalg.matrix_rank(res.cov_params_default)
        elif cov_type == 'none':
            res.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            res._rank = np.nan
            res.cov_kwds['description'] = descriptions['none']
        elif self.cov_type == 'approx':
            res.cov_params_default = res.cov_params_approx
            res.cov_kwds['description'] = descriptions['approx'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'oim':
            res.cov_params_default = res.cov_params_oim
            res.cov_kwds['description'] = descriptions['OIM'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'opg':
            res.cov_params_default = res.cov_params_opg
            res.cov_kwds['description'] = descriptions['OPG'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'robust' or self.cov_type == 'robust_oim':
            res.cov_params_default = res.cov_params_robust_oim
            res.cov_kwds['description'] = descriptions['robust-OIM'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'robust_approx':
            res.cov_params_default = res.cov_params_robust_approx
            res.cov_kwds['description'] = descriptions['robust-approx'].format(
                                                approx_type=approx_type_str)
        else:
            raise NotImplementedError('Invalid covariance matrix type.')

        return res

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        # return -2 * self.llf + 2 * self.df_model
        return aic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        # return (-2 * self.llf +
        #         self.df_model * np.log(self.nobs_effective))
        return bic(self.llf, self.nobs_effective, self.df_model)

    def _cov_params_approx(self, approx_complex_step=True,
                           approx_centered=False):
        evaluated_hessian = self.nobs_effective * self.model.hessian(
            params=self.params, transformed=True, method='approx',
            approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)
        # TODO: Case with "not approx_complex_step" is not hit in
        # tests as of 2017-05-19

        (neg_cov, singular_values) = pinv_extended(evaluated_hessian)

        self.model.update(self.params)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_approx(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian approximated by complex step or finite differences methods.
        """
        return self._cov_params_approx(self._cov_approx_complex_step,
                                       self._cov_approx_centered)

    def _cov_params_oim(self, approx_complex_step=True, approx_centered=False):
        evaluated_hessian = self.nobs_effective * self.model.hessian(
            self.params, hessian_method='oim', transformed=True,
            approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)

        (neg_cov, singular_values) = pinv_extended(evaluated_hessian)

        self.model.update(self.params)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_oim(self):
        """
        (array) The variance / covariance matrix. Computed using the method
        from Harvey (1989).
        """
        return self._cov_params_oim(self._cov_approx_complex_step,
                                    self._cov_approx_centered)

    def _cov_params_opg(self, approx_complex_step=True, approx_centered=False):
        evaluated_hessian = self.nobs_effective * self.model._hessian_opg(
            self.params, transformed=True,
            approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)

        (neg_cov, singular_values) = pinv_extended(evaluated_hessian)

        self.model.update(self.params)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_opg(self):
        """
        (array) The variance / covariance matrix. Computed using the outer
        product of gradients method.
        """
        return self._cov_params_opg(self._cov_approx_complex_step,
                                    self._cov_approx_centered)

    @cache_readonly
    def cov_params_robust(self):
        """
        (array) The QMLE variance / covariance matrix. Alias for
        `cov_params_robust_oim`
        """
        return self.cov_params_robust_oim

    def _cov_params_robust_oim(self, approx_complex_step=True,
                               approx_centered=False):
        cov_opg = self._cov_params_opg(approx_complex_step=approx_complex_step,
                                       approx_centered=approx_centered)

        evaluated_hessian = self.nobs_effective * self.model.hessian(
            self.params, hessian_method='oim', transformed=True,
            approx_complex_step=approx_complex_step,
            approx_centered=approx_centered)

        cov_params, singular_values = pinv_extended(
            np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian)
        )

        self.model.update(self.params)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return cov_params

    @cache_readonly
    def cov_params_robust_oim(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        method from Harvey (1989) as the evaluated hessian.
        """
        return self._cov_params_robust_oim(self._cov_approx_complex_step,
                                           self._cov_approx_centered)

    def _cov_params_robust_approx(self, approx_complex_step=True,
                                  approx_centered=False):
        cov_opg = self._cov_params_opg(approx_complex_step=approx_complex_step,
                                       approx_centered=approx_centered)

        evaluated_hessian = self.nobs_effective * self.model.hessian(
            self.params, transformed=True, method='approx',
            approx_complex_step=approx_complex_step)
        # TODO: Case with "not approx_complex_step" is not
        # hit in tests as of 2017-05-19

        (cov_params, singular_values) = pinv_extended(
            np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian)
        )

        self.model.update(self.params)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return cov_params

    @cache_readonly
    def cov_params_robust_approx(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        numerical Hessian as the evaluated hessian.
        """
        return self._cov_params_robust_approx(self._cov_approx_complex_step,
                                              self._cov_approx_centered)

    def info_criteria(self, criteria, method='standard'):
        r"""
        Information criteria

        Parameters
        ----------
        criteria : {'aic', 'bic', 'hqic'}
            The information criteria to compute.
        method : {'standard', 'lutkepohl'}
            The method for information criteria computation. Default is
            'standard' method; 'lutkepohl' computes the information criteria
            as in Lütkepohl (2007). See Notes for formulas.

        Notes
        -----
        The `'standard'` formulas are:

        .. math::

            AIC & = -2 \log L(Y_n | \hat \psi) + 2 k \\
            BIC & = -2 \log L(Y_n | \hat \psi) + k \log n \\
            HQIC & = -2 \log L(Y_n | \hat \psi) + 2 k \log \log n \\

        where :math:`\hat \psi` are the maximum likelihood estimates of the
        parameters, :math:`n` is the number of observations, and `k` is the
        number of estimated parameters.

        Note that the `'standard'` formulas are returned from the `aic`, `bic`,
        and `hqic` results attributes.

        The `'lutkepohl'` formuals are (Lütkepohl, 2010):

        .. math::

            AIC_L & = \log | Q | + \frac{2 k}{n} \\
            BIC_L & = \log | Q | + \frac{k \log n}{n} \\
            HQIC_L & = \log | Q | + \frac{2 k \log \log n}{n} \\

        where :math:`Q` is the state covariance matrix. Note that the Lütkepohl
        definitions do not apply to all state space models, and should be used
        with care outside of SARIMAX and VARMAX models.

        References
        ----------
        .. [*] Lütkepohl, Helmut. 2007. *New Introduction to Multiple Time*
           *Series Analysis.* Berlin: Springer.
        """
        criteria = criteria.lower()
        method = method.lower()

        if method == 'standard':
            out = getattr(self, criteria)
        elif method == 'lutkepohl':
            if self.filter_results.state_cov.shape[-1] > 1:
                raise ValueError('Cannot compute Lütkepohl statistics for'
                                 ' models with time-varying state covariance'
                                 ' matrix.')

            cov = self.filter_results.state_cov[:, :, 0]
            if criteria == 'aic':
                out = np.squeeze(np.linalg.slogdet(cov)[1] +
                                 2 * self.df_model / self.nobs_effective)
            elif criteria == 'bic':
                out = np.squeeze(np.linalg.slogdet(cov)[1] +
                                 self.df_model * np.log(self.nobs_effective) /
                                 self.nobs_effective)
            elif criteria == 'hqic':
                out = np.squeeze(np.linalg.slogdet(cov)[1] +
                                 2 * self.df_model *
                                 np.log(np.log(self.nobs_effective)) /
                                 self.nobs_effective)
            else:
                raise ValueError('Invalid information criteria')

        else:
            raise ValueError('Invalid information criteria computation method')

        return out

    @cache_readonly
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; don't want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        fittedvalues = self.forecasts
        if fittedvalues.shape[0] == 1:
            fittedvalues = fittedvalues[0, :]
        else:
            fittedvalues = fittedvalues.T
        return fittedvalues

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        # return (-2 * self.llf +
        #         2 * np.log(np.log(self.nobs_effective)) * self.df_model)
        return hqic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def llf_obs(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.filter_results.llf_obs

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.llf_obs[self.filter_results.loglikelihood_burn:].sum()

    @cache_readonly
    def loglikelihood_burn(self):
        """
        (float) The number of observations during which the likelihood is not
        evaluated.
        """
        return self.filter_results.loglikelihood_burn

    @cache_readonly
    def pvalues(self):
        """
        (array) The p-values associated with the z-statistics of the
        coefficients. Note that the coefficients are assumed to have a Normal
        distribution.
        """
        return norm.sf(np.abs(self.zvalues)) * 2

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; don't want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        resid = self.forecasts_error
        if resid.shape[0] == 1:
            resid = resid[0, :]
        else:
            resid = resid.T
        return resid

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse

    def test_normality(self, method):
        """
        Test for normality of standardized residuals.

        Null hypothesis is normality.

        Parameters
        ----------
        method : string {'jarquebera'} or None
            The statistical test for normality. Must be 'jarquebera' for
            Jarque-Bera normality test. If None, an attempt is made to select
            an appropriate test.

        Notes
        -----
        Let `d` = max(loglikelihood_burn, nobs_diffuse); this test is
        calculated ignoring the first `d` residuals.

        In the case of missing data, the maintained hypothesis is that the
        data are missing completely at random. This test is then run on the
        standardized residuals excluding those corresponding to missing
        observations.

        See Also
        --------
        statsmodels.stats.stattools.jarque_bera

        """
        if method is None:
            method = 'jarquebera'

        if method == 'jarquebera':
            from statsmodels.stats.stattools import jarque_bera
            d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            output = []
            for i in range(self.model.k_endog):
                resid = self.filter_results.standardized_forecasts_error[i, d:]
                mask = ~np.isnan(resid)
                output.append(jarque_bera(resid[mask]))
        else:
            raise NotImplementedError('Invalid normality test method.')

        return np.array(output)

    def test_heteroskedasticity(self, method, alternative='two-sided',
                                use_f=True):
        r"""
        Test for heteroskedasticity of standardized residuals

        Tests whether the sum-of-squares in the first third of the sample is
        significantly different than the sum-of-squares in the last third
        of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
        is of no heteroskedasticity.

        Parameters
        ----------
        method : string {'breakvar'} or None
            The statistical test for heteroskedasticity. Must be 'breakvar'
            for test of a break in the variance. If None, an attempt is
            made to select an appropriate test.
        alternative : string, 'increasing', 'decreasing' or 'two-sided'
            This specifies the alternative for the p-value calculation. Default
            is two-sided.
        use_f : boolean, optional
            Whether or not to compare against the asymptotic distribution
            (chi-squared) or the approximate small-sample distribution (F).
            Default is True (i.e. default is to compare against an F
            distribution).

        Returns
        -------
        output : array
            An array with `(test_statistic, pvalue)` for each endogenous
            variable. The array is then sized `(k_endog, 2)`. If the method is
            called as `het = res.test_heteroskedasticity()`, then `het[0]` is
            an array of size 2 corresponding to the first endogenous variable,
            where `het[0][0]` is the test statistic, and `het[0][1]` is the
            p-value.

        Notes
        -----
        The null hypothesis is of no heteroskedasticity. That means different
        things depending on which alternative is selected:

        - Increasing: Null hypothesis is that the variance is not increasing
          throughout the sample; that the sum-of-squares in the later
          subsample is *not* greater than the sum-of-squares in the earlier
          subsample.
        - Decreasing: Null hypothesis is that the variance is not decreasing
          throughout the sample; that the sum-of-squares in the earlier
          subsample is *not* greater than the sum-of-squares in the later
          subsample.
        - Two-sided: Null hypothesis is that the variance is not changing
          throughout the sample. Both that the sum-of-squares in the earlier
          subsample is not greater than the sum-of-squares in the later
          subsample *and* that the sum-of-squares in the later subsample is
          not greater than the sum-of-squares in the earlier subsample.

        For :math:`h = [T/3]`, the test statistic is:

        .. math::

            H(h) = \sum_{t=T-h+1}^T  \tilde v_t^2
            \Bigg / \sum_{t=d+1}^{d+1+h} \tilde v_t^2

        where :math:`d` = max(loglikelihood_burn, nobs_diffuse)` (usually
        corresponding to diffuse initialization under either the approximate
        or exact approach).

        This statistic can be tested against an :math:`F(h,h)` distribution.
        Alternatively, :math:`h H(h)` is asymptotically distributed according
        to :math:`\chi_h^2`; this second test can be applied by passing
        `asymptotic=True` as an argument.

        See section 5.4 of [1]_ for the above formula and discussion, as well
        as additional details.

        TODO

        - Allow specification of :math:`h`

        References
        ----------
        .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
               *Models and the Kalman Filter.* Cambridge University Press.
        """
        if method is None:
            method = 'breakvar'

        if method == 'breakvar':
            # Store some values
            squared_resid = self.filter_results.standardized_forecasts_error**2
            d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            # This differs from self.nobs_effective because here we want to
            # exclude exact diffuse periods, whereas self.nobs_effective only
            # excludes explicitly burned (usually approximate diffuse) periods.
            nobs_effective = self.nobs - d

            test_statistics = []
            p_values = []
            for i in range(self.model.k_endog):
                h = int(np.round(nobs_effective / 3))
                numer_resid = squared_resid[i, -h:]
                numer_resid = numer_resid[~np.isnan(numer_resid)]
                numer_dof = len(numer_resid)

                denom_resid = squared_resid[i, d:d+h]
                denom_resid = denom_resid[~np.isnan(denom_resid)]
                denom_dof = len(denom_resid)

                if numer_dof < 2:
                    warnings.warn('Early subset of data for variable %d'
                                  '  has too few non-missing observations to'
                                  ' calculate test statistic.' % i)
                    numer_resid = np.nan
                if denom_dof < 2:
                    warnings.warn('Later subset of data for variable %d'
                                  '  has too few non-missing observations to'
                                  ' calculate test statistic.' % i)
                    denom_resid = np.nan

                test_statistic = np.sum(numer_resid) / np.sum(denom_resid)

                # Setup functions to calculate the p-values
                if use_f:
                    from scipy.stats import f
                    pval_lower = lambda test_statistics: f.cdf(  # noqa:E731
                        test_statistics, numer_dof, denom_dof)
                    pval_upper = lambda test_statistics: f.sf(  # noqa:E731
                        test_statistics, numer_dof, denom_dof)
                else:
                    from scipy.stats import chi2
                    pval_lower = lambda test_statistics: chi2.cdf(  # noqa:E731
                        numer_dof * test_statistics, denom_dof)
                    pval_upper = lambda test_statistics: chi2.sf(  # noqa:E731
                        numer_dof * test_statistics, denom_dof)

                # Calculate the one- or two-sided p-values
                alternative = alternative.lower()
                if alternative in ['i', 'inc', 'increasing']:
                    p_value = pval_upper(test_statistic)
                elif alternative in ['d', 'dec', 'decreasing']:
                    test_statistic = 1. / test_statistic
                    p_value = pval_upper(test_statistic)
                elif alternative in ['2', '2-sided', 'two-sided']:
                    p_value = 2 * np.minimum(
                        pval_lower(test_statistic),
                        pval_upper(test_statistic)
                    )
                else:
                    raise ValueError('Invalid alternative.')

                test_statistics.append(test_statistic)
                p_values.append(p_value)

            output = np.c_[test_statistics, p_values]
        else:
            raise NotImplementedError('Invalid heteroskedasticity test'
                                      ' method.')

        return output

    def test_serial_correlation(self, method, lags=None):
        """
        Ljung-box test for no serial correlation of standardized residuals

        Null hypothesis is no serial correlation.

        Parameters
        ----------
        method : string {'ljungbox','boxpierece'} or None
            The statistical test for serial correlation. If None, an attempt is
            made to select an appropriate test.
        lags : None, int or array_like
            If lags is an integer then this is taken to be the largest lag
            that is included, the test result is reported for all smaller lag
            length.
            If lags is a list or array, then all lags are included up to the
            largest lag in the list, however only the tests for the lags in the
            list are reported.
            If lags is None, then the default maxlag is 12*(nobs/100)^{1/4}

        Returns
        -------
        output : array
            An array with `(test_statistic, pvalue)` for each endogenous
            variable and each lag. The array is then sized
            `(k_endog, 2, lags)`. If the method is called as
            `ljungbox = res.test_serial_correlation()`, then `ljungbox[i]`
            holds the results of the Ljung-Box test (as would be returned by
            `statsmodels.stats.diagnostic.acorr_ljungbox`) for the `i` th
            endogenous variable.

        Notes
        -----
        Let `d` = max(loglikelihood_burn, nobs_diffuse); this test is
        calculated ignoring the first `d` residuals.

        Output is nan for any endogenous variable which has missing values.

        See Also
        --------
        statsmodels.stats.diagnostic.acorr_ljungbox

        """
        if method is None:
            method = 'ljungbox'

        if method == 'ljungbox' or method == 'boxpierce':
            from statsmodels.stats.diagnostic import acorr_ljungbox
            d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            # This differs from self.nobs_effective because here we want to
            # exclude exact diffuse periods, whereas self.nobs_effective only
            # excludes explicitly burned (usually approximate diffuse) periods.
            nobs_effective = self.nobs - d
            output = []

            # Default lags for acorr_ljungbox is 40, but may not always have
            # that many observations
            if lags is None:
                lags = min(40, nobs_effective - 1)

            for i in range(self.model.k_endog):
                results = acorr_ljungbox(
                    self.filter_results.standardized_forecasts_error[i][d:],
                    lags=lags, boxpierce=(method == 'boxpierce'))
                if method == 'ljungbox':
                    output.append(results[0:2])
                else:
                    output.append(results[2:])

            output = np.c_[output]
        else:
            raise NotImplementedError('Invalid serial correlation test'
                                      ' method.')
        return output

    def get_prediction(self, start=None, end=None, dynamic=False,
                       index=None, **kwargs):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : boolean, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        if start is None:
            start = self.model._index[0]

        # Handle start, end, dynamic
        start, end, out_of_sample, prediction_index = (
            self.model._get_prediction_index(start, end, index))

        # Handle `dynamic`
        if isinstance(dynamic, (bytes, unicode)):
            dynamic, _, _ = self.model._get_index_loc(dynamic)

        # Perform the prediction
        # This is a (k_endog x npredictions) array; don't want to squeeze in
        # case of npredictions = 1
        prediction_results = self.filter_results.predict(
            start, end + out_of_sample + 1, dynamic, **kwargs)

        # Return a new mlemodel.PredictionResults object
        return PredictionResultsWrapper(PredictionResults(
            self, prediction_results, row_labels=prediction_index))

    def get_forecast(self, steps=1, **kwargs):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        if isinstance(steps, (int, long)):
            end = self.nobs + steps - 1
        else:
            end = steps
        return self.get_prediction(start=self.nobs, end=end, **kwargs)

    def predict(self, start=None, end=None, dynamic=False, **kwargs):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : boolean, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        # Perform the prediction
        prediction_results = self.get_prediction(start, end, dynamic, **kwargs)
        return prediction_results.predicted_mean

    def forecast(self, steps=1, **kwargs):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        if isinstance(steps, (int, long)):
            end = self.nobs + steps - 1
        else:
            end = steps
        return self.predict(start=self.nobs, end=end, **kwargs)

    def simulate(self, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None):
        r"""
        Simulate a new time series following the state space model

        Parameters
        ----------
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the state vector at time zero, which should
            be shaped (`k_states` x 1), where `k_states` is the same as in the
            state space model. If unspecified, but the model has been
            initialized, then that initialization is used. If unspecified and
            the model has not been initialized, then a vector of zeros is used.
            Note that this is not included in the returned `simulated_states`
            array.

        Returns
        -------
        simulated_obs : array
            An (nsimulations x k_endog) array of simulated observations.
        """
        scale = self.scale if self.filter_results.filter_concentrated else None
        with self.model.ssm.fixed_scale(scale):
            sim = self.model.simulate(self.params, nsimulations,
                                      measurement_shocks, state_shocks,
                                      initial_state)
        return sim

    def impulse_responses(self, steps=1, impulse=0, orthogonalized=False,
                          cumulative=False, **kwargs):
        """
        Impulse response function

        Parameters
        ----------
        steps : int, optional
            The number of steps for which impulse responses are calculated.
            Default is 1. Note that the initial impulse is not counted as a
            step, so if `steps=1`, the output will have 2 entries.
        impulse : int or array_like
            If an integer, the state innovation to pulse; must be between 0
            and `k_posdef-1`. Alternatively, a custom impulse vector may be
            provided; must be shaped `k_posdef x 1`.
        orthogonalized : boolean, optional
            Whether or not to perform impulse using orthogonalized innovations.
            Note that this will also affect custum `impulse` vectors. Default
            is False.
        cumulative : boolean, optional
            Whether or not to return cumulative impulse responses. Default is
            False.
        **kwargs
            If the model is time-varying and `steps` is greater than the number
            of observations, any of the state space representation matrices
            that are time-varying must have updated values provided for the
            out-of-sample steps.
            For example, if `design` is a time-varying component, `nobs` is 10,
            and `steps` is 15, a (`k_endog` x `k_states` x 5) matrix must be
            provided with the new design matrix values.

        Returns
        -------
        impulse_responses : array
            Responses for each endogenous variable due to the impulse
            given by the `impulse` argument. A (steps + 1 x k_endog) array.

        Notes
        -----
        Intercepts in the measurement and state equation are ignored when
        calculating impulse responses.

        """
        scale = self.scale if self.filter_results.filter_concentrated else None
        with self.model.ssm.fixed_scale(scale):
            irfs = self.model.impulse_responses(self.params, steps, impulse,
                                                orthogonalized, cumulative,
                                                **kwargs)
        return irfs

    def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None):
        """
        Diagnostic plots for standardized residuals of one endogenous variable

        Parameters
        ----------
        variable : integer, optional
            Index of the endogenous variable for which the diagnostic plots
            should be created. Default is 0.
        lags : integer, optional
            Number of lags to include in the correlogram. Default is 10.
        fig : Matplotlib Figure instance, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the 2x2 grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Produces a 2x2 plot grid with the following plots (ordered clockwise
        from top left):

        1. Standardized residuals over time
        2. Histogram plus estimated density of standardized residulas, along
           with a Normal(0,1) density plotted for reference.
        3. Normal Q-Q plot, with Normal reference line.
        4. Correlogram

        See Also
        --------
        statsmodels.graphics.gofplots.qqplot
        statsmodels.graphics.tsaplots.plot_acf
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        # Eliminate residuals associated with burned or diffuse likelihoods
        d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
        resid = self.filter_results.standardized_forecasts_error[variable, d:]

        # Top-left: residuals vs time
        ax = fig.add_subplot(221)
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            x = self.data.dates[d:]._mpl_repr()
        else:
            x = np.arange(len(resid))
        ax.plot(x, resid)
        ax.hlines(0, x[0], x[-1], alpha=0.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title('Standardized residual')

        # Top-right: histogram, Gaussian kernel density, Normal density
        # Can only do histogram and Gaussian kernel density on the non-null
        # elements
        resid_nonmissing = resid[~(np.isnan(resid))]
        ax = fig.add_subplot(222)

        # gh5792: Remove  except after support for matplotlib>2.1 required
        try:
            ax.hist(resid_nonmissing, density=True, label='Hist')
        except AttributeError:
            ax.hist(resid_nonmissing, normed=True, label='Hist')

        from scipy.stats import gaussian_kde, norm
        kde = gaussian_kde(resid_nonmissing)
        xlim = (-1.96*2, 1.96*2)
        x = np.linspace(xlim[0], xlim[1])
        ax.plot(x, kde(x), label='KDE')
        ax.plot(x, norm.pdf(x), label='N(0,1)')
        ax.set_xlim(xlim)
        ax.legend()
        ax.set_title('Histogram plus estimated density')

        # Bottom-left: QQ plot
        ax = fig.add_subplot(223)
        from statsmodels.graphics.gofplots import qqplot
        qqplot(resid_nonmissing, line='s', ax=ax)
        ax.set_title('Normal Q-Q')

        # Bottom-right: Correlogram
        ax = fig.add_subplot(224)
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(resid, ax=ax, lags=lags)
        ax.set_title('Correlogram')

        ax.set_ylim(-1, 1)

        return fig

    def summary(self, alpha=.05, start=None, title=None, model_name=None,
                display_params=True):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        model_name : string
            The name of the model used. Default is to use model class name.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary

        # Model specification results
        model = self.model
        if title is None:
            title = 'Statespace Model Results'

        if start is None:
            start = 0
        if self.model._index_dates:
            ix = self.model._index
            d = ix[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = ix[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.nobs)]

        # Standardize the model name as a list of str
        if model_name is None:
            model_name = model.__class__.__name__

        # Diagnostic tests results
        try:
            het = self.test_heteroskedasticity(method='breakvar')
        except Exception:  # FIXME: catch something specific
            het = np.array([[np.nan]*2])
        try:
            lb = self.test_serial_correlation(method='ljungbox')
        except Exception:  # FIXME: catch something specific
            lb = np.array([[np.nan]*2]).reshape(1, 2, 1)
        try:
            jb = self.test_normality(method='jarquebera')
        except Exception:  # FIXME: catch something specific
            jb = np.array([[np.nan]*4])

        # Create the tables
        if not isinstance(model_name, list):
            model_name = [model_name]

        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [
            ('Date:', None),
            ('Time:', None),
            ('Sample:', [sample[0]]),
            ('', [sample[1]])
        ]

        top_right = [
            ('No. Observations:', [self.nobs]),
            ('Log Likelihood', ["%#5.3f" % self.llf]),
        ]
        if hasattr(self, 'rsquared'):
            top_right.append(('R-squared:', ["%#8.3f" % self.rsquared]))
        top_right += [
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('HQIC', ["%#5.3f" % self.hqic])]
        if (self.filter_results is not None and
                self.filter_results.filter_concentrated):
            top_right.append(('Scale', ["%#5.3f" % self.scale]))

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        format_str = lambda array: [  # noqa:E731
            ', '.join(['{0:.2f}'.format(i) for i in array])
        ]
        diagn_left = [('Ljung-Box (Q):', format_str(lb[:, 0, -1])),
                      ('Prob(Q):', format_str(lb[:, 1, -1])),
                      ('Heteroskedasticity (H):', format_str(het[:, 0])),
                      ('Prob(H) (two-sided):', format_str(het[:, 1]))
                      ]

        diagn_right = [('Jarque-Bera (JB):', format_str(jb[:, 0])),
                       ('Prob(JB):', format_str(jb[:, 1])),
                       ('Skew:', format_str(jb[:, 2])),
                       ('Kurtosis:', format_str(jb[:, 3]))
                       ]

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)
        if len(self.params) > 0 and display_params:
            summary.add_table_params(self, alpha=alpha,
                                     xname=self.data.param_names, use_t=False)
        summary.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                                title="")

        # Add warnings/notes, added to text format only
        etext = []
        if hasattr(self, 'cov_type') and 'description' in self.cov_kwds:
            etext.append(self.cov_kwds['description'])
        if self._rank < len(self.params):
            etext.append("Covariance matrix is singular or near-singular,"
                         " with condition number %6.3g. Standard errors may be"
                         " unstable." % np.linalg.cond(self.cov_params()))

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Warnings:")
            summary.add_extra_txt(etext)

        return summary


class MLEResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'zvalues': 'columns',
        'cov_params_approx': 'cov',
        'cov_params_default': 'cov',
        'cov_params_oim': 'cov',
        'cov_params_opg': 'cov',
        'cov_params_robust': 'cov',
        'cov_params_robust_approx': 'cov',
        'cov_params_robust_oim': 'cov',
    }
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {
        'forecast': 'dates',
        'simulate': 'ynames',
        'impulse_responses': 'ynames'
    }
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(MLEResultsWrapper, MLEResults)  # noqa:E305


class PredictionResults(pred.PredictionResults):
    """

    Parameters
    ----------
    prediction_results : kalman_filter.PredictionResults instance
        Results object from prediction after fitting or filtering a state space
        model.
    row_labels : iterable
        Row labels for the predicted data.

    Attributes
    ----------

    """
    def __init__(self, model, prediction_results, row_labels=None):
        if model.model.k_endog == 1:
            endog = pd.Series(prediction_results.endog[:, 0],
                              name=model.model.endog_names)
        else:
            endog = pd.DataFrame(prediction_results.endog.T,
                                 columns=model.model.endog_names)
        self.model = Bunch(data=model.data.__class__(
            endog=endog, predict_dates=row_labels))
        self.prediction_results = prediction_results

        # Get required values
        predicted_mean = self.prediction_results.forecasts
        if predicted_mean.shape[0] == 1:
            predicted_mean = predicted_mean[0, :]
        else:
            predicted_mean = predicted_mean.transpose()

        var_pred_mean = self.prediction_results.forecasts_error_cov
        if var_pred_mean.shape[0] == 1:
            var_pred_mean = var_pred_mean[0, 0, :]
        else:
            var_pred_mean = var_pred_mean.transpose()

        # Initialize
        super(PredictionResults, self).__init__(predicted_mean, var_pred_mean,
                                                dist='norm',
                                                row_labels=row_labels,
                                                link=identity())

    @property
    def se_mean(self):
        if self.var_pred_mean.ndim == 1:
            se_mean = np.sqrt(self.var_pred_mean)
        else:
            se_mean = np.sqrt(self.var_pred_mean.T.diagonal())
        return se_mean

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        # TODO: this performs metadata wrapping, and that should be handled
        #       by attach_* methods. However, they don't currently support
        #       this use case.
        conf_int = super(PredictionResults, self).conf_int(
            method, alpha, **kwds)

        # Create a dataframe
        if self.row_labels is not None:
            conf_int = pd.DataFrame(conf_int, index=self.row_labels)

            # Attach the endog names
            ynames = self.model.data.ynames
            if not type(ynames) == list:
                ynames = [ynames]
            names = (['lower %s' % name for name in ynames] +
                     ['upper %s' % name for name in ynames])
            conf_int.columns = names

        return conf_int

    def summary_frame(self, endog=0, what='all', alpha=0.05):
        # TODO: finish and cleanup
        # import pandas as pd
        from collections import OrderedDict
        # ci_obs = self.conf_int(alpha=alpha, obs=True) # need to split
        ci_mean = np.asarray(self.conf_int(alpha=alpha))
        to_include = OrderedDict()
        if self.predicted_mean.ndim == 1:
            yname = self.model.data.ynames
            to_include['mean'] = self.predicted_mean
            to_include['mean_se'] = self.se_mean
            k_endog = 1
        else:
            yname = self.model.data.ynames[endog]
            to_include['mean'] = self.predicted_mean[:, endog]
            to_include['mean_se'] = self.se_mean[:, endog]
            k_endog = self.predicted_mean.shape[1]
        to_include['mean_ci_lower'] = ci_mean[:, endog]
        to_include['mean_ci_upper'] = ci_mean[:, k_endog + endog]

        # OrderedDict doesn't work to preserve sequence
        # pandas dict doesn't handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        res.columns.name = yname
        return res


class PredictionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'predicted_mean': 'dates',
        'se_mean': 'dates',
        't_values': 'dates',
    }
    _wrap_attrs = wrap.union_dicts(_attrs)

    _methods = {}
    _wrap_methods = wrap.union_dicts(_methods)
wrap.populate_wrapper(PredictionResultsWrapper, PredictionResults)  # noqa:E305
