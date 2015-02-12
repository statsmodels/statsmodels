"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from scipy.stats import norm
from .kalman_filter import FilterResults

import statsmodels.tsa.base.tsa_model as tsbase
from .model import Model
from statsmodels.tools.numdiff import approx_hess_cs, approx_fprime_cs
from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels.tools.eval_measures import aic, bic, hqic


class MLEModel(Model):
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
    updater : callable or None
        Can be set with a callable accepting arguments
        (`model`, `params`) that can be used to update
        the state space representation for ad-hoc MLE.
    transformer : callable or None
        Can be set with a callable accepting arguments
        (`model`, `params`) that can be used to transform
        parameters to constrained parameters for ad-hoc MLE.
    untransformer : callable or None
        Can be set with a callable accepting arguments
        (`model`, `params`) that can be used to perform
        a reverse transformation of parameters for ad-hoc MLE.

    Notes
    -----
    This class extends the state space model with Kalman filtering to add in
    functionality for maximum likelihood estimation. In particular, it adds
    the concept of updating the state space representation based on a defined
    set of parameters, through the `update` method or `updater` attribute (see
    below for more details on which to use when), and it adds a `fit` method
    which uses a numerical optimizer to select the parameters that maximize
    the likelihood of the model.

    It is used in one of two ways:

    1. A base class
    2. Ad-hoc MLE

    **As a base class**

    The most typical usage of the MLEModel class is as a base class so that a
    specific state space model can be built as a subclass without having to
    deal with optimization-related functionality.

    In this case, the `start_params` `update` method must be overridden in the
    child class (and the `transform` and `untransform` methods, if needed).

    **Ad-hoc MLE**

    This class can also be instantiated directly for ad-hoc MLE, particularly
    if the model is very simple.

    In this case, the `start_params` attribute can be set directly and in place
    of the `update`, `transform`, and `untransform` methods, the attributes
    `updater`, `transformer`, and `untransformer` can be set with callback
    functions to perform that functionality.

    See Also
    --------
    MLEResults
    statsmodels.tsa.statespace.kalman_filter.KalmanFilter
    statsmodels.tsa.statespace.representation.Representation
    """

    def __init__(self, endog, k_states, exog=None, dates=None, freq=None,
                 **kwargs):
        # Set the default results class to be MLEResults
        kwargs.setdefault('results_class', MLEResults)

        super(MLEModel, self).__init__(endog, k_states, exog, dates, freq,
                                       **kwargs)

        # Initialize the parameters
        self.params = None

        # Initialize placeholders
        self.updater = None
        self.transformer = None
        self.untransformer = None

    def fit(self, start_params=None, transformed=True,
            method='lbfgs', maxiter=50, full_output=1,
            disp=5, callback=None, return_params=False,
            bfgs_tune=False, **kwargs):
        """
        Fits the model by maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
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
        bfgs_tune : boolean, optional
            BFGS methods by default use internal methods for approximating the
            score and hessian by finite differences. If `bfgs_tune=True` the
            maximizing parameters from the BFGS method are used as starting
            parameters for a second round of maximization using complex-step
            differentiation. Has no effect for other methods. Default is False.
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

        # Unconstrain the starting parameters
        if transformed:
            start_params = self.untransform_params(np.array(start_params))

        if method == 'lbfgs' or method == 'bfgs':
            kwargs.setdefault('approx_grad', True)
            kwargs.setdefault('epsilon', 1e-5)

        # Maximum likelihood estimation
        # Set the optional arguments for the loglikelihood function to
        # maximize the average loglikelihood, by default.
        fargs = (kwargs.get('average_loglike', True), False, False)
        mlefit = super(MLEModel, self).fit(start_params, method=method,
                                           fargs=fargs,
                                           maxiter=maxiter,
                                           full_output=full_output, disp=disp,
                                           callback=callback,
                                           skip_hessian=True, **kwargs)

        # Optionally tune the maximum likelihood estimates using complex step
        # gradient
        if bfgs_tune and method == 'lbfgs' or method == 'bfgs':
            kwargs['approx_grad'] = False
            del kwargs['epsilon']
            fargs = (kwargs.get('average_loglike', True), False, False)
            mlefit = super(MLEModel, self).fit(mlefit.params, method=method,
                                               fargs=fargs,
                                               maxiter=maxiter,
                                               full_output=full_output,
                                               disp=disp, callback=callback,
                                               skip_hessian=True,
                                               **kwargs)

        # Constrain the final parameters and update the model to be sure we're
        # using them (in case, for example, the last time update was called
        # via the optimizer it was a gradient calculation, etc.)
        self.update(mlefit.params, transformed=False)

        # Just return the fitted parameters if requested
        if return_params:
            self.filter(results='loglikelihood')
            return self.params
        # Otherwise construct the results class if desired
        else:
            res = self.filter()
            res.mlefit = mlefit
            res.mle_retvals = mlefit.mle_retvals
            res.mle_settings = mlefit.mle_settings
            return res

    def loglike(self, params=None, average_loglike=False, transformed=True,
                set_params=True, **kwargs):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        average_loglike : boolean, optional
            Whether or not to return the average loglikelihood (rather than
            the sum of loglikelihoods across all observations). Default is
            False.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        set_params : boolean
            Whether or not to copy `params` to the model object's params
            attribute. Default is True.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        [1]_ recommend maximizing the average likelihood to avoid scale issues;
        this can be achieved by setting `average_loglike=True`.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.

        See Also
        --------
        update : modifies the internal state of the Model to reflect new params
        """
        if params is not None:
            self.update(params, transformed=transformed, set_params=set_params)

        loglike = super(MLEModel, self).loglike(**kwargs)

        # Koopman, Shephard, and Doornik recommend maximizing the average
        # likelihood to avoid scale issues.
        if average_loglike:
            return loglike / self.nobs
        else:
            return loglike

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
        This is a numerical approximation.

        Both \*args and \*\*kwargs are necessary because the optimizer from
        `fit` must call this function and only supports passing arguments via
        \*args (for example `scipy.optimize.fmin_l_bfgs`).
        """
        nargs = len(args)
        if nargs < 1:
            kwargs.setdefault('average_loglike', True)
        if nargs < 2:
            kwargs.setdefault('transformed', False)
        if nargs < 3:
            kwargs.setdefault('set_params', False)

        initial_state = kwargs.pop('initial_state', None)
        initial_state_cov = kwargs.pop('initial_state_cov', None)
        if initial_state is not None and initial_state_cov is not None:
            # If initialization is stationary, we don't want to recalculate the
            # initial_state_cov for each new set of parameters here
            initialization = self.initialization
            _initial_state = self._initial_state
            _initial_state_cov = self._initial_state_cov
            _initial_variance = self._initial_variance

            self.initialize_known(initial_state, initial_state_cov)

        score = approx_fprime_cs(params, self.loglike, epsilon=1e-9, args=args,
                                 kwargs=kwargs)

        if initial_state is not None and initial_state_cov is not None:
            # Reset the initialization
            self.initialization = initialization
            self._initial_state = _initial_state
            self._initial_state_cov = _initial_state_cov
            self._initial_variance = _initial_variance

        return score

    def hessian(self, params, *args, **kwargs):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters.

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
        nargs = len(args)
        if nargs < 1:
            kwargs.setdefault('average_loglike', True)
        if nargs < 2:
            kwargs.setdefault('transformed', False)
        if nargs < 3:
            kwargs.setdefault('set_params', False)

        initial_state = kwargs.pop('initial_state', None)
        initial_state_cov = kwargs.pop('initial_state_cov', None)
        if initial_state is not None and initial_state_cov is not None:
            # If initialization is stationary, we don't want to recalculate the
            # initial_state_cov for each new set of parameters here
            initialization = self.initialization
            _initial_state = self._initial_state
            _initial_state_cov = self._initial_state_cov
            _initial_variance = self._initial_variance

            self.initialize_known(initial_state, initial_state_cov)

        hessian = approx_hess_cs(params, self.loglike, epsilon=1e-9, args=args,
                                 kwargs=kwargs)

        if initial_state is not None and initial_state_cov is not None:
            # Reset the initialization
            self.initialization = initialization
            self._initial_state = _initial_state
            self._initial_state_cov = _initial_state_cov
            self._initial_variance = _initial_variance

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

    @start_params.setter
    def start_params(self, values):
        self._start_params = np.asarray(values)

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        return self.model_names

    @property
    def model_names(self):
        """
        (list of str) The plain text names of all possible model parameters.
        """
        return self._get_model_names(latex=False)

    @property
    def model_latex_names(self):
        """
        (list of str) The latex names of all possible model parameters.
        """
        return self._get_model_names(latex=True)

    def _get_model_names(self, latex=False):
        if latex:
            names = ['param_%d' % i for i in range(len(self.start_params))]
        else:
            names = ['param.%d' % i for i in range(len(self.start_params))]
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
        if self.transformer is not None:
            constrained = self.transformer(self, unconstrained)
        else:
            constrained = unconstrained
        return constrained

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
        if self.untransformer is not None:
            unconstrained = self.untransformer(self, constrained)
        else:
            unconstrained = constrained
        return unconstrained

    def update(self, params, transformed=True, set_params=True):
        """
        Update the parameters of the model

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : boolean, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True.
        set_params : boolean
            Whether or not to copy `params` to the model object's params
            attribute. Usually is set to True unless a subclass has additional
            defined behavior in the case it is False (otherwise this is a noop
            except for possibly transforming the parameters). Default is True.

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        Since Model is a base class, this method should be overridden by
        subclasses to perform actual updating steps.
        """
        params = np.array(params)

        if not transformed:
            params = self.transform_params(params)
        if set_params:
            self.params = params

        if self.updater is not None:
            self.updater(self, params)

        return params

    @classmethod
    def from_formula(cls, formula, data, subset=None):
        """
        Not implemented for State space models
        """
        raise NotImplementedError


class MLEResults(FilterResults, tsbase.TimeSeriesModelResults):
    r"""
    Class to hold results from fitting a state space model.

    Parameters
    ----------
    model : Model instance
        The fitted model instance

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
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
    def __init__(self, model):
        self.data = model.data

        # Save the model output
        self._endog_names = model.endog_names
        self._exog_names = model.endog_names
        self._params = model.params
        self._param_names = model.param_names
        self._model_names = model.model_names
        self._model_latex_names = model.model_latex_names

        # Associate the names with the true parameters
        params = pd.Series(self._params, index=self._param_names)

        # Initialize the Statsmodels model base
        tsbase.TimeSeriesModelResults.__init__(self, model, params,
                                               normalized_cov_params=None,
                                               scale=1.)

        # Initialize the statespace representation
        super(MLEResults, self).__init__(model)

        # Setup the cache
        self._cache = resettable_cache()

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

    def cov_params(self):
        """
        The variance / covariance matrix. Computed using the numerical Hessian.
        """
        return self.cov_params_default

    @cache_readonly
    def cov_params_default(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian computed without using parameter transformations.
        """
        hessian = self.model.hessian(
            self._params, set_params=False, transformed=True,
            initial_state=self.initial_state,
            initial_state_cov=self.initial_state_cov
        )

        # Reset the matrices to the saved parameters (since they were
        # overwritten in the hessian call)
        self.model.update(self.model.params)

        return -np.linalg.inv(hessian*self.nobs)

    @cache_readonly
    def cov_params_delta(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian computed using parameter transformations and the Delta method
        (method of propagation of errors).
        """

        unconstrained = self.model.untransform_params(self._params)
        jacobian = self.model.transform_jacobian(unconstrained)
        hessian = self.model.hessian(
            unconstrained, set_params=False,
            initial_state=self.initial_state,
            initial_state_cov=self.initial_state_cov
        )

        # Reset the matrices to the saved parameters (since they were
        # overwritten in the hessian call)
        self.model.update(self.model.params)

        return jacobian.dot(-np.linalg.inv(hessian*self.nobs)).dot(jacobian.T)

    def fittedvalues(self):
        """
        (array) The predicted values of the model.
        """
        return self.forecasts

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        # return -2*self.llf + 2*np.log(np.log(self.nobs))*self.params.shape[0]
        return hqic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.loglikelihood[self.loglikelihood_burn:].sum()

    @cache_readonly
    def pvalues(self):
        """
        (array) The p-values associated with the z-statistics of the
        coefficients. Note that the coefficients are assumed to have a Normal
        distribution.
        """
        return norm.sf(np.abs(self.zvalues)) * 2

    def resid(self):
        """
        (array) The model residuals.
        """
        return self.forecasts_error

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse

    def predict(self, start=None, end=None, dynamic=False, full_results=False,
                **kwargs):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
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
        full_results : boolean, optional
            If True, returns a FilterResults instance; if False returns a
            tuple with forecasts, the forecast errors, and the forecast error
            covariance matrices. Default is False.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts.
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
        results = super(MLEResults, self).predict(
            start, end+out_of_sample+1, dynamic, full_results, **kwargs
        )

        # Note: to be consistent with Statsmodels, return only the forecasts
        # unless full_results is specified. Confidence intervals and the date
        # indices are left out for now, but will likely be moved to a separate
        # function in the future.
        if full_results:
            return results
        else:
            # (forecasts, forecasts_error, forecasts_error_cov) = results
            forecasts = results

        # Calculate the confidence intervals
        # critical_value = norm.ppf(1 - alpha / 2.)
        # std_errors = np.sqrt(forecasts_error_cov.diagonal().T)
        # confidence_intervals = np.c_[
        #     (forecasts - critical_value*std_errors)[:, :, None],
        #     (forecasts + critical_value*std_errors)[:, :, None],
        # ]

        # Return the dates if we have them
        # index = np.arange(start, end+out_of_sample+1)
        # if hasattr(self.data, 'predict_dates'):
        #     index = self.data.predict_dates
        #     if(isinstance(index, pd.DatetimeIndex)):
        #         index = index._mpl_repr()

        return forecasts

    def forecast(self, steps=1, **kwargs):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, optional
            The number of out of sample forecasts from the end of the
            sample. Default is 1.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts.
        """
        return self.predict(start=self.nobs, end=self.nobs+steps-1, **kwargs)

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

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)
        summary.add_table_params(self, alpha=alpha, xname=self._param_names,
                                 use_t=False)

        return summary
