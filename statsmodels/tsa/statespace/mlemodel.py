"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from scipy.stats import norm

from .kalman_smoother import KalmanSmoother, SmootherResults
from .kalman_filter import (
    KalmanFilter, FilterResults, PredictionResults, INVERT_UNIVARIATE, SOLVE_LU
)
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.base.wrapper as wrap
from statsmodels.tools.numdiff import (
    _get_epsilon, approx_hess_cs, approx_fprime_cs
)
from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.tools import Bunch
import statsmodels.genmod._prediction as pred
from statsmodels.genmod.families.links import identity


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
    ssm : KalmanFilter
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

    optim_hessian = 'cs'

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
        endog = np.array(self.data.orig_endog)
        exog = self.data.orig_exog
        if exog is not None:
            exog = np.array(exog)

        # Base class may allow 1-dim data, whereas we need 2-dim
        if endog.ndim == 1:
            endog.shape = (endog.shape[0], 1)  # this will be C-contiguous

        # Base classes data may be either C-ordered or F-ordered - we want it
        # to be C-ordered since it will also be in shape (nobs, k_endog), and
        # then we can just transpose it.
        if not endog.flags['C_CONTIGUOUS']:
            # TODO this breaks the reference link between the model endog
            # variable and the original object - do we need a warn('')?
            # This will happen often with Pandas DataFrames, which are often
            # Fortran-ordered and in the long format
            endog = np.ascontiguousarray(endog)

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
        self.ssm = KalmanSmoother(endog.shape[0], self.k_states, **kwargs)
        # Bind the data to the model
        self.ssm.bind(endog)

        # Other dimensions, now that `ssm` is available
        self.k_endog = self.ssm.k_endog

    def fit(self, start_params=None, transformed=True, cov_type='opg',
            cov_kwds=None, method='lbfgs', maxiter=50, full_output=1,
            disp=5, callback=None, return_params=False,
            optim_hessian=None, **kwargs):
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
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'cs' for the observed information matrix estimator, calculated
              using a numerical (complex step) approximation of the Hessian
              matrix.
            - 'delta' for the observed information matrix estimator, calculated
              using a numerical (complex step) approximation of the Hessian
              along with the delta method (method of propagation of errors)
              applied to the parameter transformation function
              `transform_params`.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presense of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_cs' is the same as 'robust' except that the intermediate
              calculations use the 'cs' method.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
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
        optim_hessian : {'opg','oim','cs'}, optional
            The method by which the Hessian is numerically approximated. 'opg'
            uses outer product of gradients, 'oim' uses the information
            matrix formula from Harvey (1989), and 'cs' uses second-order
            complex step differentiation. This keyword is only relevant if the
            optimization method uses the Hessian matrix.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        MLEResults

        See also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        MLEResults
        """

        if start_params is None:
            start_params = self.start_params
            transformed = True

        # Update the hessian method
        if optim_hessian is not None:
            self.optim_hessian = optim_hessian

        # Unconstrain the starting parameters
        if transformed:
            start_params = self.untransform_params(np.array(start_params))

        if method == 'lbfgs' or method == 'bfgs':
            # kwargs.setdefault('pgtol', 1e-8)
            # kwargs.setdefault('factr', 1e2)
            # kwargs.setdefault('m', 12)
            kwargs.setdefault('approx_grad', True)
            kwargs.setdefault('epsilon', 1e-5)

        # Maximum likelihood estimation
        fargs = (False,)  # (sets transformed=False)
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
            res = self.filter(mlefit.params, transformed=False,
                              cov_type=cov_type, cov_kwds=cov_kwds)
            res.mlefit = mlefit
            res.mle_retvals = mlefit.mle_retvals
            res.mle_settings = mlefit.mle_settings

            return res

    def filter(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_ssm=False, **kwargs):
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
        self.update(params, transformed=True)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Get the state space output
        result = self.ssm.filter(**kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = MLEResultsWrapper(
                MLEResults(self, params, result, **result_kwargs)
            )

        return result

    def smooth(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_ssm=False, **kwargs):
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
        self.update(params, transformed=True)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Get the state space output
        result = self.ssm.smooth(**kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = MLEResultsWrapper(
                MLEResults(self, params, result, **result_kwargs)
            )

        return result

    def loglike(self, params, transformed=True, **kwargs):
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
        update : modifies the internal state of the state space model to
                 reflect new params
        """
        if not transformed:
            params = self.transform_params(params)
        self.update(params, transformed=True)

        loglike = self.ssm.loglike(**kwargs)

        # Koopman, Shephard, and Doornik recommend maximizing the average
        # likelihood to avoid scale issues, but the averaging is done
        # automatically in the base model `fit` method
        return loglike

    def loglikeobs(self, params, transformed=True, **kwargs):
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
        self.update(params, transformed=True)

        return self.ssm.loglikeobs(**kwargs)

    def observed_information_matrix(self, params, **kwargs):
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
        epsilon = _get_epsilon(params, 1, None, n)
        increments = np.identity(n) * 1j * epsilon

        # Get values at the params themselves
        self.update(params)
        res = self.ssm.filter(**kwargs)
        dtype = self.ssm.dtype
        # Save this for inversion later
        inv_forecasts_error_cov = res.forecasts_error_cov.copy()

        # Compute partial derivatives
        partials_forecasts_error = (
            np.zeros((self.k_endog, self.nobs, n))
        )
        partials_forecasts_error_cov = (
            np.zeros((self.k_endog, self.k_endog, self.nobs, n))
        )
        for i, ih in enumerate(increments):
            self.update(params + ih)
            res = self.ssm.filter(**kwargs)

            partials_forecasts_error[:, :, i] = (
                res.forecasts_error.imag / epsilon[i]
            )

            partials_forecasts_error_cov[:, :, :, i] = (
                res.forecasts_error_cov.imag / epsilon[i]
            )

        # Compute the information matrix
        tmp = np.zeros((self.k_endog, self.k_endog, self.nobs, n), dtype=dtype)

        information_matrix = np.zeros((n, n), dtype=dtype)
        for t in range(self.ssm.loglikelihood_burn, self.nobs):
            inv_forecasts_error_cov[:, :, t] = (
                np.linalg.inv(inv_forecasts_error_cov[:, :, t])
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
                        np.dot(inv_forecasts_error_cov[:,:,t],
                               partials_forecasts_error[:, t, j])
                    )
        return information_matrix / (self.nobs - self.ssm.loglikelihood_burn)

    def opg_information_matrix(self, params, **kwargs):
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
        score_obs = self.score_obs(params, **kwargs).transpose()
        return (
            np.inner(score_obs, score_obs) /
            (self.nobs - self.ssm.loglikelihood_burn)
        )

    def score(self, params, *args, **kwargs):
        """
        Compute the score function at params.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        *args, **kwargs
            Additional arguments to the `loglike` method.

        Returns
        ----------
        score : array
            Score, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglike` method.

        Both \*args and \*\*kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        \*args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        params = np.array(params, ndmin=1)

        transformed = (
            args[0] if len(args) > 0 else kwargs.get('transformed', False)
        )

        score = approx_fprime_cs(params, self.loglike, kwargs={
            'transformed': transformed
        })

        return score

    def score_obs(self, params, **kwargs):
        """
        Compute the score per observation, evaluated at params
 
        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score.
        *args, **kwargs
            Additional arguments to the `loglike` method.

        Returns
        ----------
        score : array (nobs, k_vars)
            Score per observation, evaluated at `params`.

        Notes
        -----
        This is a numerical approximation, calculated using first-order complex
        step differentiation on the `loglikeobs` method.
        """
        params = np.array(params, ndmin=1)

        self.update(params)
        return approx_fprime_cs(params, self.loglikeobs, kwargs=kwargs)

    def hessian(self, params, *args, **kwargs):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the hessian.
        *args, **kwargs
            Additional arguments to the `loglike` method.

        Returns
        -------
        hessian : array
            Hessian matrix evaluated at `params`

        Notes
        -----
        This is a numerical approximation.

        Both \*args and \*\*kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        \*args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        if self.optim_hessian == 'cs':
            hessian = self._hessian_cs(params, *args, **kwargs)
        elif self.optim_hessian == 'oim':
            hessian = self._hessian_oim(params)
        elif self.optim_hessian == 'opg':
            hessian = self._hessian_opg(params)
        else:
            raise NotImplementedError('Invalid Hessian calculation method.')
        return hessian

    def _hessian_oim(self, params):
        """
        Hessian matrix computed using the Harvey (1989) information matrix
        """
        return -self.observed_information_matrix(params)

    def _hessian_opg(self, params):
        """
        Hessian matrix computed using the outer product of gradients
        information matrix
        """
        return -self.opg_information_matrix(params)

    def _hessian_cs(self, params, *args, **kwargs):
        """
        Hessian matrix computed by second-order complex-step differentiation
        on the `loglike` function.
        """

        transformed = (
            args[0] if len(args) > 0 else kwargs.get('transformed', False)
        )

        f = lambda params, **kwargs: self.loglike(params, **kwargs) / self.nobs
        hessian = approx_hess_cs(params, f, kwargs={
            'transformed': transformed
        })

        return hessian

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

    def transform_jacobian(self, unconstrained):
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
        This is a numerical approximation.

        See Also
        --------
        transform_params
        """
        return approx_fprime_cs(unconstrained, self.transform_params)

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

    def update(self, params, transformed=True):
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
        """
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

        # Simulated obs is (k_endog x nobs); don't want to squeeze in
        # case of npredictions = 1
        if simulated_obs.shape[0] == 1:
            simulated_obs = simulated_obs[0,:]
        else:
            simulated_obs = simulated_obs.T
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
        return self.ssm.impulse_responses(
            steps, impulse, orthogonalized, cumulative, **kwargs)

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
        This is currently set to 1.0 and not used by the model or its results.

    See Also
    --------
    MLEModel
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.representation.FrozenRepresentation
    """
    def __init__(self, model, params, results, cov_type='opg',
                 cov_kwds=None, **kwargs):
        self.data = model.data

        tsbase.TimeSeriesModelResults.__init__(self, model, params,
                                               normalized_cov_params=None,
                                               scale=1.)

        # Save the state space representation output
        self.filter_results = results
        if isinstance(results, SmootherResults):
            self.smoother_results = results
        else:
            self.smoother_results = None

        # Dimensions
        self.nobs = model.nobs

        # Setup covariance matrix notes dictionary
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = cov_type

        # Setup the cache
        self._cache = resettable_cache()

        # Handle covariance matrix calculation
        if cov_kwds is None:
                cov_kwds = {}
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
        - 'cs' for the observed information matrix estimator, calculated
          using a numerical (complex step) approximation of the Hessian
          matrix.
        - 'delta' for the observed information matrix estimator, calculated
          using a numerical (complex step) approximation of the Hessian along
          with the delta method (method of propagation of errors)
          applied to the parameter transformation function `transform_params`.
        - 'robust' for an approximate (quasi-maximum likelihood) covariance
          matrix that may be valid even in the presense of some
          misspecifications. Intermediate calculations use the 'oim'
          method.
        - 'robust_cs' is the same as 'robust' except that the intermediate
          calculations use the 'cs' method.
        """

        import statsmodels.stats.sandwich_covariance as sw

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
        if len(self.params) == 0:
            res.cov_params_default = np.zeros((0,0))
            res._rank = 0
            res.cov_kwds['cov_type'] = (
                'No parameters estimated.')
        elif self.cov_type == 'cs':
            res.cov_params_default = res.cov_params_cs
            res.cov_kwds['description'] = (
                'Covariance matrix calculated using numerical (complex-step)'
                ' differentiation.')
        elif self.cov_type == 'delta':
            res.cov_params_default = res.cov_params_delta
            res.cov_kwds['description'] = (
                'Covariance matrix calculated using numerical differentiation'
                ' and the delta method (method of propagation of errors)'
                ' applied to the parameter transformation function.')
        elif self.cov_type == 'oim':
            res.cov_params_default = res.cov_params_oim
            res.cov_kwds['description'] = (
                'Covariance matrix calculated using the observed information'
                ' matrix described in Harvey (1989).')
        elif self.cov_type == 'opg':
            res.cov_params_default = res.cov_params_opg
            res.cov_kwds['description'] = (
                'Covariance matrix calculated using the outer product of'
                ' gradients.'
            )
        elif self.cov_type == 'robust' or self.cov_type == 'robust_oim':
            res.cov_params_default = res.cov_params_robust_oim
            res.cov_kwds['description'] = (
                'Quasi-maximum likelihood covariance matrix used for'
                ' robustness to some misspecifications; calculated using the'
                ' observed information matrix described in Harvey (1989).')
        elif self.cov_type == 'robust_cs':
            res.cov_params_default = res.cov_params_robust_cs
            res.cov_kwds['description'] = (
                'Quasi-maximum likelihood covariance matrix used for'
                ' robustness to some misspecifications; calculated using'
                ' numerical (complex-step) differentiation.')
        else:
            raise NotImplementedError('Invalid covariance matrix type.')

        return res

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        # return -2*self.llf + 2*self.params.shape[0]
        return aic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        # return -2*self.llf + self.params.shape[0]*np.log(self.nobs)
        return bic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def cov_params_cs(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian computed without using parameter transformations.
        """
        nobs = (self.model.nobs - self.filter_results.loglikelihood_burn)
        # When using complex-step methods, cannot rely on Cholesky inversion
        # because variance parameters will then have a complex component which
        # which implies non-positive-definiteness.
        inversion_method = INVERT_UNIVARIATE | SOLVE_LU
        evaluated_hessian = self.model._hessian_cs(
            self.params, transformed=True, inversion_method=inversion_method
        )
        self.model.update(self.params)

        neg_cov, singular_values = pinv_extended(nobs * evaluated_hessian)

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return -neg_cov

    @cache_readonly
    def cov_params_delta(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian computed using parameter transformations and the Delta method
        (method of propagation of errors).
        """
        nobs = (self.model.nobs - self.filter_results.loglikelihood_burn)

        unconstrained = self.model.untransform_params(self.params)
        jacobian = self.model.transform_jacobian(unconstrained)
        neg_cov, singular_values = pinv_extended(
            nobs * self.model._hessian_cs(unconstrained, transformed=False)
        )

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return np.dot(np.dot(jacobian, -neg_cov), jacobian.transpose())

    @cache_readonly
    def cov_params_oim(self):
        """
        (array) The variance / covariance matrix. Computed using the method
        from Harvey (1989).
        """
        nobs = (self.model.nobs - self.filter_results.loglikelihood_burn)
        cov_params, singular_values = pinv_extended(
            nobs * self.model.observed_information_matrix(self.params)
        )

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return cov_params

    @cache_readonly
    def cov_params_opg(self):
        """
        (array) The variance / covariance matrix. Computed using the outer
        product of gradients method.
        """
        nobs = (self.model.nobs - self.filter_results.loglikelihood_burn)
        # When using complex-step methods, cannot rely on Cholesky inversion
        # because variance parameters will then have a complex component which
        # which implies non-positive-definiteness.
        inversion_method = INVERT_UNIVARIATE | SOLVE_LU
        cov_params, singular_values = pinv_extended(
            nobs *
            self.model.opg_information_matrix(
                self.params, inversion_method=inversion_method
            )
        )

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return cov_params

    @cache_readonly
    def cov_params_robust(self):
        """
        (array) The QMLE variance / covariance matrix. Alias for
        `cov_params_robust_oim`
        """
        return self.cov_params_robust_oim

    @cache_readonly
    def cov_params_robust_oim(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        method from Harvey (1989) as the evaluated hessian.
        """
        nobs = (self.model.nobs - self.filter_results.loglikelihood_burn)
        cov_opg = self.cov_params_opg
        evaluated_hessian = (
            nobs * self.model.observed_information_matrix(self.params)
        )
        cov_params, singular_values = pinv_extended(
            np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian)
        )

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return cov_params

    @cache_readonly
    def cov_params_robust_cs(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        numerical Hessian computed without using parameter transformations as
        the evaluated hessian.
        """
        nobs = (self.model.nobs - self.filter_results.loglikelihood_burn)
        cov_opg = self.cov_params_opg
        # When using complex-step methods, cannot rely on Cholesky inversion
        # because variance parameters will then have a complex component which
        # which implies non-positive-definiteness.
        inversion_method = INVERT_UNIVARIATE | SOLVE_LU
        evaluated_hessian = (
            nobs *
            self.model._hessian_cs(self.params, transformed=True,
                                   inversion_method=inversion_method)
        )
        cov_params, singular_values = pinv_extended(
            np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian)
        )

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return cov_params

    @cache_readonly
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; don't want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        fittedvalues = self.filter_results.forecasts
        if fittedvalues.shape[0] == 1:
            fittedvalues = fittedvalues[0,:]
        else:
            fittedvalues = fittedvalues.T
        return fittedvalues

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        # return -2*self.llf + 2*np.log(np.log(self.nobs))*self.params.shape[0]
        return hqic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def llf_obs(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglikeobs(self.params)

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
        resid = self.filter_results.forecasts_error
        if resid.shape[0] == 1:
            resid = resid[0,:]
        else:
            resid = resid.T
        return resid

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse

    def test_normality(self):
        """
        Jarque-Bera test for normaility of standardized residuals.

        Null hypothesis is normality.

        Notes
        -----
        If the first `d` loglikelihood values were burned (i.e. in the
        specified model, `loglikelihood_burn=d`), then this test is calculated
        ignoring the first `d` residuals.

        See Also
        --------
        statsmodels.stats.stattools.jarque_bera

        """
        from statsmodels.stats.stattools import jarque_bera
        d = self.loglikelihood_burn
        return np.array(jarque_bera(
            self.filter_results.standardized_forecasts_error[:, d:],
            axis=1
        )).transpose()

    def test_heteroskedasticity(self, alternative='two-sided',
                                asymptotic=False):
        """
        Sum-of-squares test for heteroskedasticity of standardized residuals

        Tests whether the sum-of-squares in the first third of the sample is
        significantly different than the sum-of-squares in the last third
        of the sample. Analogous to a Goldfeld-Quandt test.

        Parameters
        ----------
        alternative : string, 'increasing', 'decreasing' or 'two-sided'
            This specifies the alternative for the p-value calculation. Default
            is two-sided.
        asymptotic : boolean, optional
            Whether or not to compare against the asymptotic distribution
            (chi-squared) or the approximate small-sample distribution (F).
            Default is False (i.e. default is to compare against an F
            distribution).

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
            \left / \sum_{t=d+1}^{d+1+h} \tilde v_t^2 \right .

        where :math:`d` is the number of periods in which the loglikelihood was
        burned in the parent model (usually corresponding to diffuse
        initialization).

        This statistic can be tested against an :math:`F(h,h)` distribution.
        Alternatively, :math:`h H(h)` is asymptotically distributed according
        to :math:`\chi_h^2`; this second test can be applied by passing
        `asymptotic=True` as an argument.

        See section 5.4 of [1]_ for the above formula and discussion, as well
        as additional details.

        TODO

        - Allow specification of :math:`h`

        Returns
        -------
        output : array
            An array with `(test_statistic, pvalue)` for each endogenous
            variable. The array is then sized `(k_endog, 2)`. If the method is
            called as `het = res.test_heteroskedasticity()`, then `het[0]` is
            an array of size 2 corresponding to the first endogenous variable,
            where `het[0][0]` is the test statistic, and `het[0][1]` is the
            p-value.

        References
        ----------
        .. [1] Harvey, Andrew C. 1990.
           Forecasting, Structural Time Series Models and the Kalman Filter.
           Cambridge University Press.

        """
        # Store some values
        squared_resid = self.filter_results.standardized_forecasts_error**2
        d = self.loglikelihood_burn
        h = np.round((self.nobs - d) / 3)

        # Calculate the test statistics for each endogenous variable
        test_statistics = np.array([
            np.sum(squared_resid[i][-h:]) / np.sum(squared_resid[i][d:d+h])
            for i in range(self.model.k_endog)
        ])

        # Setup functions to calculate the p-values
        if not asymptotic:
            from scipy.stats import f
            pval_lower = lambda test_statistics: f.cdf(test_statistics, h, h)
            pval_upper = lambda test_statistics: f.sf(test_statistics, h, h)
        else:
            from scipy.stats import chi2
            pval_lower = lambda test_statistics: chi2.cdf(h*test_statistics, h)
            pval_upper = lambda test_statistics: chi2.sf(h*test_statistics, h)

        # Calculate the one- or two-sided p-values
        alternative = alternative.lower()
        if alternative in ['i', 'inc', 'increasing']:
            p_values = pval_upper(test_statistics)
        elif alternative in ['d', 'dec', 'decreasing']:
            test_statistics = 1. / test_statistics
            p_values = pval_upper(test_statistics)
        elif alternative in ['2', '2-sided', 'two-sided']:
            p_values = 2 * np.minimum(
                pval_lower(test_statistics),
                pval_upper(test_statistics)
            )
        else:
            raise ValueError('Invalid alternative.')

        return np.c_[test_statistics, p_values]

    def test_serial_correlation(self, lags=None, boxpierce=False):
        """
        Ljung-box test for no serial correlation of standardized residuals

        Null hypothesis is no serial correlation.

        Parameters
        ----------
        lags : None, int or array_like
            If lags is an integer then this is taken to be the largest lag
            that is included, the test result is reported for all smaller lag
            length.
            If lags is a list or array, then all lags are included up to the
            largest lag in the list, however only the tests for the lags in the
            list are reported.
            If lags is None, then the default maxlag is 12*(nobs/100)^{1/4}
        boxpierce : {False, True}
            If true, then additional to the results of the Ljung-Box test also
            the Box-Pierce test results are returned.

        Returns
        -------
        output : array
            An array with `(test_statistic, pvalue)` for each endogenous
            variable and each lag. The array is then sized
            `(k_endog, 2, lags)`. If the method is called as
            `ljungbox = res.test_serial_correlation()`, then `ljungbox[i]`
            holds the results of the Ljung-Box test (as would be returned by
            `statsmodels.stats.diagnostic.acorr_ljungbox`) for the `i`th
            endogenous variable.

        Notes
        -----
        If the first `d` loglikelihood values were burned (i.e. in the
        specified model, `loglikelihood_burn=d`), then this test is calculated
        ignoring the first `d` residuals.

        See Also
        --------
        statsmodels.stats.diagnostic.acorr_ljungbox

        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        d = self.loglikelihood_burn
        return np.c_[[
            acorr_ljungbox(
                self.filter_results.standardized_forecasts_error[i][d:]
            )
            for i in range(self.model.k_endog)
        ]]

    def get_prediction(self, start=None, end=None, dynamic=False, **kwargs):
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
            start = 0

        # Handle start and end (e.g. dates)
        start = self.model._get_predict_start(start)
        end, out_of_sample = self.model._get_predict_end(end)

        # Handle string dynamic
        dates = self.data.dates
        if isinstance(dynamic, str):
            if dates is None:
                raise ValueError("Got a string for dynamic and dates is None")
            dtdynamic = self.model._str_to_date(dynamic)
            try:
                dynamic_start = self.model._get_dates_loc(dates, dtdynamic)

                dynamic = dynamic_start - start
            except KeyError:
                raise ValueError("Dynamic must be in dates. Got %s | %s" %
                                 (str(dynamic), str(dtdynamic)))

        # Perform the prediction
        # This is a (k_endog x npredictions) array; don't want to squeeze in
        # case of npredictions = 1
        prediction_results = self.filter_results.predict(
            start, end+out_of_sample+1, dynamic, **kwargs
        )

        # Return a new mlemodel.PredictionResults object
        if self.data.dates is None:
            row_labels = self.data.row_labels
        else:
            row_labels = self.data.predict_dates
        return PredictionResultsWrapper(
            PredictionResults(self, prediction_results, row_labels=row_labels))

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
        if isinstance(steps, int):
            end = self.nobs+steps-1
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
        if start is None:
            start = 0

        # Handle start and end (e.g. dates)
        start = self.model._get_predict_start(start)
        end, out_of_sample = self.model._get_predict_end(end)

        # Handle string dynamic
        dates = self.data.dates
        if isinstance(dynamic, str):
            if dates is None:
                raise ValueError("Got a string for dynamic and dates is None")
            dtdynamic = self.model._str_to_date(dynamic)
            try:
                dynamic_start = self.model._get_dates_loc(dates, dtdynamic)

                dynamic = dynamic_start - start
            except KeyError:
                raise ValueError("Dynamic must be in dates. Got %s | %s" %
                                 (str(dynamic), str(dtdynamic)))

        # Perform the prediction
        # This is a (k_endog x npredictions) array; don't want to squeeze in
        # case of npredictions = 1
        prediction_results = self.filter_results.predict(
            start, end+out_of_sample+1, dynamic, **kwargs
        )
        predicted_mean = prediction_results.forecasts
        if predicted_mean.shape[0] == 1:
            predicted_mean = predicted_mean[0,:]
        else:
            predicted_mean = predicted_mean.T
        return predicted_mean

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
        if isinstance(steps, int):
            end = self.nobs+steps-1
        else:
            end = steps
        return self.predict(start=self.nobs, end=end, **kwargs)

    def simulate(self, nsimulations, measurement_shocks=None,
                 state_shocks=None, initial_state=None):
        """
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
        return self.model.simulate(self.params, nsimulations,
            measurement_shocks, state_shocks, initial_state)

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
        return self.model.impulse_responses(self.params, steps, impulse,
            orthogonalized, cumulative, **kwargs)

    def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None):
        """
        Diagnostic plots for the standardized residual assocaited with one
        endogenous variable.

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
        _ = _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        # Eliminate residuals associated with burned likelihoods
        d = self.loglikelihood_burn
        resid = self.filter_results.standardized_forecasts_error[variable, d:]

        # Top-left: residuals vs time
        ax = fig.add_subplot(221)
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            x = self.data.dates[self.loglikelihood_burn:]._mpl_repr()
        else:
            x = np.arange(len(resid))
        ax.plot(x, resid)
        ax.hlines(0, x[0], x[-1], alpha=0.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title('Standardized residual')

        # Top-right: histogram, Gaussian kernel density, Normal density
        ax = fig.add_subplot(222)
        ax.hist(resid, normed=True, label='Hist')
        from scipy.stats import gaussian_kde, norm
        kde = gaussian_kde(resid)
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
        qqplot(resid, line='s', ax=ax)
        ax.set_title('Normal Q-Q')

        # Bottom-right: Correlogram
        ax = fig.add_subplot(224)
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(resid, ax=ax, lags=10)
        ax.set_title('Correlogram')

        ax.set_ylim(-1,1)

        return fig

    def summary(self, alpha=.05, start=None, model_name=None):
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
        title = 'Statespace Model Results'

        if start is None:
            start = 0
        if self.data.dates is not None:
            dates = self.data.dates
            d = dates[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = dates[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.model.nobs)]

        if model_name is None:
            model_name = model.__class__.__name__

        # Diagnostic tests results
        het = self.test_heteroskedasticity()
        lb = self.test_serial_correlation()
        jb = self.test_normality()

        # Create the tables

        top_left = [
            ('Dep. Variable:', None),
            ('Model:', [model_name]),
            ('Date:', None),
            ('Time:', None),
            ('Sample:', [sample[0]]),
            ('', [sample[1]])
        ]

        top_right = [
            ('No. Observations:', [self.model.nobs]),
            ('Log Likelihood', ["%#5.3f" % self.llf]),
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('HQIC', ["%#5.3f" % self.hqic])
        ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        format_str = lambda array: [
            ', '.join(['{0:.2f}'.format(i) for i in array])
        ]
        diagn_left = [('Ljung-Box (Q):', format_str(lb[:,0,-1])),
                      ('Prob(Q):', format_str(lb[:,1,-1])),
                      ('Heteroskedasticity (H):', format_str(het[:,0])),
                      ('Prob(H) (two-sided):', format_str(het[:,1]))
                      ]

        diagn_right = [('Jarque-Bera (JB):', format_str(jb[:,0])),
                       ('Prob(JB):', format_str(jb[:,1])),
                       ('Skew:', format_str(jb[:,2])),
                       ('Kurtosis:', format_str(jb[:,3]))
                       ]

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)
        if len(self.params) > 0:
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
        'cov_params_cs': 'cov',
        'cov_params_default': 'cov',
        'cov_params_delta': 'cov',
        'cov_params_oim': 'cov',
        'cov_params_opg': 'cov',
        'cov_params_robust': 'cov',
        'cov_params_robust_cs': 'cov',
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
wrap.populate_wrapper(MLEResultsWrapper, MLEResults)


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
        self.model = Bunch(data=model.data.__class__(
            endog=prediction_results.endog.T,
            predict_dates=getattr(model.data, 'predict_dates', None))
        )
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

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        # TODO: this performs metadata wrapping, and that should be handled
        #       by attach_* methods. However, they don't currently support
        #       this use case.
        conf_int = super(PredictionResults, self).conf_int(
            method, alpha, **kwds)

        if self.model.data.predict_dates is not None:
            conf_int = pd.DataFrame(conf_int,
                                    index=self.model.data.predict_dates)

        return conf_int

    def summary_frame(self, endog=0, what='all', alpha=0.05):
        # TODO: finish and cleanup
        # import pandas as pd
        from statsmodels.compat.collections import OrderedDict
        #ci_obs = self.conf_int(alpha=alpha, obs=True) # need to split
        ci_mean = self.conf_int(alpha=alpha)
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


        self.table = to_include
        #OrderedDict doesn't work to preserve sequence
        # pandas dict doesn't handle 2d_array
        #data = np.column_stack(list(to_include.values()))
        #names = ....
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
wrap.populate_wrapper(PredictionResultsWrapper, PredictionResults)
