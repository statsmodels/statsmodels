"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from scipy.stats import norm
from .representation import Representation, FilterResults

import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tools.numdiff import approx_hess_cs, approx_fprime_cs
from statsmodels.tools.decorators import cache_readonly, resettable_cache


class Model(Representation, tsbase.TimeSeriesModel):
    """
    State space model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    dates : array-like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    """

    def __init__(self, endog, k_states, exog=None, dates=None, freq=None,
                 *args, **kwargs):
        self.i = 0
        # Initialize the model base
        tsbase.TimeSeriesModel.__init__(self, endog=endog, exog=exog,
                                        dates=dates, freq=freq, missing='none')

        # Set the default results class to be StatespaceResults
        kwargs.setdefault('filter_results_class', StatespaceResults)

        # Initialize the statespace representation
        super(Model, self).__init__(self.endog, k_states, *args, **kwargs)

        # Initialize the parameters
        self.params = None

        # Set additional parameters
        self.nobs = self.endog.shape[1]

    def fit(self, start_params=None, transformed=True,
            method='lbfgs', maxiter=50, full_output=1,
            disp=5, callback=None, return_params=False,
            bfgs_tune=False, *args, **kwargs):

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
        mlefit = super(Model, self).fit(start_params, method=method,
                                        fargs=fargs,
                                        maxiter=maxiter,
                                        full_output=full_output, disp=disp,
                                        callback=callback, **kwargs)

        # Optionally tune the maximum likelihood estimates using complex step
        # gradient
        if bfgs_tune and method == 'lbfgs' or method == 'bfgs':
            kwargs['approx_grad'] = False
            del kwargs['epsilon']
            fargs = (kwargs.get('average_loglike', True), False, False)
            mlefit = super(Model, self).fit(mlefit.params, method=method,
                                            fargs=fargs,
                                            maxiter=maxiter,
                                            full_output=full_output, disp=disp,
                                            callback=callback, **kwargs)

        # Constrain the final parameters and update the model to be sure we're
        # using them (in case, for example, the last time update was called
        # via the optimizer it was a gradient calculation, etc.)
        self.update(mlefit.params, transformed=False)

        # Just return the fitted parameters if requested
        if return_params:
            self.filter(return_loglike=True)
            return self.params
        # Otherwise construct the results class if desired
        else:
            res = self.filter()
            res.mlefit = mlefit
            res.mle_retvals = mlefit.mle_retvals
            res.mle_settings = mlefit.mle_settings
            return res

    def loglike(self, params=None, average_loglike=False, transformed=True,
                set_params=True, *args, **kwargs):
        """
        Loglikelihood evaluation

        References
        ----------
        Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
        "Statistical Algorithms for Models in State Space Using SsfPack 2.2."
        Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        if params is not None:
            self.update(params, transformed=transformed, set_params=set_params)

        # By default, we do not need to consider recreating the entire
        # _statespace and Cython Kalman filter objects because only parameters
        # will be changing and not dimensions of matrices.
        kwargs.setdefault('recreate', False)

        loglike = super(Model, self).loglike(*args, **kwargs)

        # Koopman, Shephard, and Doornik recommend maximizing the average
        # likelihood to avoid scale issues.
        if average_loglike:
            return loglike / self.nobs
        else:
            return loglike

    def score(self, params, *args, **kwargs):
        nargs = len(args)
        if nargs < 1:
            kwargs.setdefault('average_loglike', True)
        if nargs < 2:
            kwargs.setdefault('transformed', False)
        if nargs < 3:
            kwargs.setdefault('set_params', False)
        return approx_fprime_cs(params, self.loglike, epsilon=1e-9, args=args, kwargs=kwargs)

    def hessian(self, params, *args, **kwargs):
        nargs = len(args)
        if nargs < 1:
            kwargs.setdefault('average_loglike', True)
        if nargs < 2:
            kwargs.setdefault('transformed', False)
        if nargs < 3:
            kwargs.setdefault('set_params', False)
        return approx_hess_cs(params, self.loglike, epsilon=1e-9, args=args, kwargs=kwargs)

    @property
    def start_params(self):
        raise NotImplementedError

    @property
    def params_names(self):
        return self.model_names

    @property
    def model_names(self):
        return self._get_model_names(latex=False)

    @property
    def model_latex_names(self):
        return self._get_model_names(latex=True)

    def _get_model_names(self, latex=False):
        if latex:
            names = ['param_%d' % i for i in range(len(self.start_params))]
        else:
            names = ['param.%d' % i for i in range(len(self.start_params))]
        return names

    def transform_jacobian(self, unconstrained):
        return approx_fprime_cs(unconstrained, self.transform_params)

    def transform_params(self, unconstrained):
        return unconstrained

    def untransform_params(self, constrained):
        return constrained

    def update(self, params, transformed=True, set_params=True):
        if not transformed:
            params = self.transform_params(params)
        if set_params:
            self.params = params
        return params


class StatespaceResults(FilterResults, tsbase.TimeSeriesModelResults):
    def __init__(self, model, kalman_filter, *args, **kwargs):
        self.data = model.data

        # Save the model output
        self._endog_names = model.endog_names
        self._exog_names = model.endog_names
        self._params = model.params
        self._params_names = model.params_names
        self._model_names = model.model_names
        self._model_latex_names = model.model_latex_names

        # Associate the names with the true parameters
        params = pd.Series(self._params, index=self._params_names)

        # Initialize the Statsmodels model base
        tsbase.TimeSeriesModelResults.__init__(self, model, params,
                                               normalized_cov_params=None,
                                               scale=1., *args, **kwargs)

        # Initialize the statespace representation
        super(StatespaceResults, self).__init__(model, kalman_filter, *args,
                                                **kwargs)

        # Setup the cache
        self._cache = resettable_cache()

    @cache_readonly
    def aic(self):
        return -2*self.llf + 2*self.params.shape[0]

    @cache_readonly
    def bic(self):
        return -2*self.llf + self.params.shape[0]*np.log(self.nobs)

    @cache_readonly
    def hqic(self):
        return -2*self.llf + 2*np.log(np.log(self.nobs))*self.params.shape[0]

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diagonal(self.cov_params))

    @cache_readonly
    def cov_params(self):
        # Uses Delta method (method of propagation of errors)
        unconstrained = self.model.untransform_params(self._params)
        jacobian = self.model.transform_jacobian(unconstrained)
        hessian = self.model.hessian(unconstrained)
        return jacobian.dot(-np.linalg.inv(hessian*self.nobs)).dot(jacobian.T)

    @cache_readonly
    def llf(self):
        return self.loglikelihood[self.loglikelihood_burn:].sum()

    def resid(self):
        return self.forecasts_error.copy()

    def fittedvalues(self):
        return self.forecasts.copy()

    @cache_readonly
    def pvalues(self):
        return norm.sf(np.abs(self.zvalues)) * 2

    @cache_readonly
    def zvalues(self):
        return self.params / self.bse

    def predict(self, start=None, end=None, dynamic=False, alpha=.05,
                full_results=False, *args, **kwargs):
        if start is None:
            start = 0

        # Handle start and end (e.g. dates)
        start = self.model._get_predict_start(start)
        end, out_of_sample = self.model._get_predict_end(end)

        # Perform the prediction
        res = super(StatespaceResults, self).predict(
            start, end+out_of_sample+1, dynamic, full_results, *args, **kwargs
        )

        if full_results:
            return res
        else:
            (forecasts, forecasts_error, forecasts_error_cov) = res

        # Calculate the confidence intervals
        critical_value = norm.ppf(1 - alpha / 2.)
        std_errors = np.sqrt(forecasts_error_cov.diagonal().T)
        confidence_intervals = np.c_[
            (forecasts - critical_value*std_errors)[:, :, None],
            (forecasts + critical_value*std_errors)[:, :, None],
        ]

        # Return the dates if we have them
        index = np.arange(start, end+out_of_sample+1)
        if hasattr(self.data, 'predict_dates'):
            index = self.data.predict_dates

        return forecasts, forecasts_error_cov, confidence_intervals, index

    def forecast(self, steps=1, alpha=.05, *args, **kwargs):
        return self.predict(start=self.nobs, end=self.nobs+steps-1, alpha=alpha,
                            *args, **kwargs)

    def summary(self, alpha=.05, start=None, *args, **kwargs):
        """Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals.

        Returns
        -------
        smry : Summary instance
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
            sample = [dates[start].strftime('%m-%d-%Y')]
            sample += ['- ' + dates[-1].strftime('%m-%d-%Y')]
        else:
            sample = [str(start), + ' - ' + str(self.model.nobs)]

        top_left = [
            ('Dep. Variable:', None),
            ('Model:', [kwargs.get('model', model.__class__.__name__)]),
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
        summary.add_table_params(self, alpha=alpha, xname=self._params_names,
                                 use_t=False)

        return summary
