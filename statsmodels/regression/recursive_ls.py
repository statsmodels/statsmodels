"""
Recursive least squares model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
from statsmodels.compat.collections import OrderedDict

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
    MLEModel, MLEResults, MLEResultsWrapper)
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly, resettable_cache
import statsmodels.base.wrapper as wrap

# Columns are alpha = 0.1, 0.05, 0.025, 0.01, 0.005
_cusum_squares_scalars = np.array([
    [1.0729830,   1.2238734,  1.3581015,  1.5174271,  1.6276236],
    [-0.6698868, -0.6700069, -0.6701218, -0.6702672, -0.6703724],
    [-0.5816458, -0.7351697, -0.8858694, -1.0847745, -1.2365861]
])


class RecursiveLS(MLEModel):
    r"""
    Recursive least squares

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like
        Array of exogenous regressors, shaped nobs x k.

    Notes
    -----
    Recursive least squares (RLS) corresponds to expanding window ordinary
    least squares (OLS).

    This model applies the Kalman filter to compute recursive estimates of the
    coefficients and recursive residuals.

    References
    ----------
    .. [*] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.

    """
    def __init__(self, endog, exog, **kwargs):
        # Standardize data
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog)

        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)

        # Make sure we have 2-dimensional array
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)

        self.k_exog = exog.shape[1]

        # Handle coefficient initialization
        # By default, do not calculate likelihood while it is controlled by
        # diffuse initial conditions.
        kwargs.setdefault('loglikelihood_burn', self.k_exog)
        kwargs.setdefault('initialization', 'approximate_diffuse')
        kwargs.setdefault('initial_variance', 1e9)

        # Initialize the state space representation
        super(RecursiveLS, self).__init__(
            endog, k_states=self.k_exog, exog=exog, **kwargs
        )

        # Setup the state space representation
        self['design'] = self.exog[:, :, None].T
        self['transition'] = np.eye(self.k_states)

        # Notice that the filter output does not depend on the measurement
        # variance, so we set it here to 1
        self['obs_cov', 0, 0] = 1.

    @classmethod
    def from_formula(cls, formula, data, subset=None):
        """
        Not implemented for state space models
        """
        return super(MLEModel, cls).from_formula(formula, data, subset)

    def fit(self):
        """
        Fits the model by application of the Kalman filter

        Returns
        -------
        RecursiveLSResults
        """
        # Get the smoother results with an arbitrary measurement variance
        smoother_results = self.smooth(return_ssm=True)
        # Compute the MLE of sigma2 (see Harvey, 1989 equation 4.2.5)
        resid = smoother_results.standardized_forecasts_error[0]
        sigma2 = (np.inner(resid, resid) /
                  (self.nobs - self.loglikelihood_burn))

        # Now construct a results class, where the params are the final
        # estimates of the regression coefficients
        self['obs_cov', 0, 0] = sigma2
        return self.smooth()

    def filter(self, return_ssm=False, **kwargs):
        # Get the state space output
        result = super(RecursiveLS, self).filter([], transformed=True,
                                                 cov_type='none',
                                                 return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            params = result.filtered_state[:, -1]
            cov_kwds = {
                'custom_cov_type': 'nonrobust',
                'custom_cov_params': result.filtered_state_cov[:, :, -1],
                'custom_description': ('Parameters and covariance matrix'
                                       ' estimates are RLS estimates'
                                       ' conditional on the entire sample.')
            }
            result = RecursiveLSResultsWrapper(
                RecursiveLSResults(self, params, result, cov_type='custom',
                                   cov_kwds=cov_kwds)
            )

        return result

    def smooth(self, return_ssm=False, **kwargs):
        # Get the state space output
        result = super(RecursiveLS, self).smooth([], transformed=True,
                                                 cov_type='none',
                                                 return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            params = result.filtered_state[:, -1]
            cov_kwds = {
                'custom_cov_type': 'nonrobust',
                'custom_cov_params': result.filtered_state_cov[:, :, -1],
                'custom_description': ('Parameters and covariance matrix'
                                       ' estimates are RLS estimates'
                                       ' conditional on the entire sample.')
            }
            result = RecursiveLSResultsWrapper(
                RecursiveLSResults(self, params, result, cov_type='custom',
                                   cov_kwds=cov_kwds)
            )

        return result

    @property
    def param_names(self):
        return self.exog_names

    @property
    def start_params(self):
        # Only parameter is the measurement disturbance standard deviation
        return np.zeros(0)

    def update(self, params, **kwargs):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : boolean, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.
        """
        pass


class RecursiveLSResults(MLEResults):
    """
    Class to hold results from fitting a recursive least squares model.

    Parameters
    ----------
    model : RecursiveLS instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the recursive least squares
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type='opg',
                 **kwargs):
        super(RecursiveLSResults, self).__init__(
            model, params, filter_results, cov_type, **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()

        # Save the model specification
        self.specification = Bunch(**{
            'k_exog': self.model.k_exog})

    @property
    def recursive_coefficients(self):
        """
        Estimates of regression coefficients, recursively estimated

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        out = None
        spec = self.specification
        start = offset = 0
        end = offset + spec.k_exog
        out = Bunch(
            filtered=self.filtered_state[start:end],
            filtered_cov=self.filtered_state_cov[start:end, start:end],
            smoothed=None, smoothed_cov=None,
            offset=offset
        )
        if self.smoothed_state is not None:
            out.smoothed = self.smoothed_state[start:end]
        if self.smoothed_state_cov is not None:
            out.smoothed_cov = (
                self.smoothed_state_cov[start:end, start:end])
        return out

    @cache_readonly
    def resid_recursive(self):
        """
        Recursive residuals

        Returns
        -------
        resid_recursive : array_like
            An array of length `nobs` holding the recursive
            residuals.

        Notes
        -----
        The first `k_exog` residuals are typically unreliable due to
        initialization.

        """
        # See Harvey (1989) section 5.4; he defines the standardized
        # innovations in 5.4.1, but they have non-unit variance, whereas
        # the standardized forecast errors assume unit variance. To convert
        # to Harvey's definition, we need to multiply by the standard
        # deviation.
        return (self.filter_results.standardized_forecasts_error.squeeze() *
                self.filter_results.obs_cov[0, 0]**0.5)

    @cache_readonly
    def cusum(self):
        r"""
        Cumulative sum of standardized recursive residuals statistics

        Returns
        -------
        cusum : array_like
            An array of length `nobs - k_exog` holding the
            CUSUM statistics.


        Notes
        -----
        The CUSUM statistic takes the form:

        .. math::

            W_t = \frac{1}{\hat \sigma} \sum_{j=k+1}^t w_j

        where :math:`w_j` is the recursive residual at time :math:`j` and
        :math:`\hat \sigma` is the estimate of the standard deviation
        from the full sample.

        Excludes the first `k_exog` datapoints.

        Due to differences in the way :math:`\hat \sigma` is calculated, the
        output of this function differs slightly from the output in the
        R package strucchange and the Stata contributed .ado file cusum6. The
        calculation in this package is consistent with the description of
        Brown et al. (1975)

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.

        """
        llb = self.loglikelihood_burn
        return (np.cumsum(self.resid_recursive[self.loglikelihood_burn:]) /
                np.std(self.resid_recursive[llb:], ddof=1))

    @cache_readonly
    def cusum_squares(self):
        r"""
        Cumulative sum of squares of standardized recursive residuals
        statistics

        Returns
        -------
        cusum_squares : array_like
            An array of length `nobs - k_exog` holding the
            CUSUM of squares statistics.

        Notes
        -----
        The CUSUM of squares statistic takes the form:

        .. math::

            s_t = \left ( \sum_{j=k+1}^t w_j^2 \right ) \Bigg /
                  \left ( \sum_{j=k+1}^T w_j^2 \right )

        where :math:`w_j` is the recursive residual at time :math:`j`.

        Excludes the first `k_exog` datapoints.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.

        """
        numer = np.cumsum(self.resid_recursive[self.loglikelihood_burn:]**2)
        denom = numer[-1]
        return numer / denom

    def plot_recursive_coefficient(self, variables=0, alpha=0.05,
                                   legend_loc='upper left', fig=None,
                                   figsize=None):
        r"""
        Plot the recursively estimated coefficients on a given variable

        Parameters
        ----------
        variables : int or str or iterable of int or string, optional
            Integer index or string name of the variable whose coefficient will
            be plotted. Can also be an iterable of integers or strings. Default
            is the first variable.
        alpha : float, optional
            The confidence intervals for the coefficient are (1 - alpha) %
        legend_loc : string, optional
            The location of the legend in the plot. Default is upper left.
        fig : Matplotlib Figure instance, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        All plots contain (1 - `alpha`) %  confidence intervals.
        """
        # Get variables
        if isinstance(variables, (int, str)):
            variables = [variables]
        k_variables = len(variables)

        # If a string was given for `variable`, try to get it from exog names
        exog_names = self.model.exog_names
        for i in range(k_variables):
            variable = variables[i]
            if isinstance(variable, str):
                variables[i] = exog_names.index(variable)

        # Create the plot
        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)

        for i in range(k_variables):
            variable = variables[i]
            ax = fig.add_subplot(k_variables, 1, i + 1)

            # Get dates, if applicable
            if hasattr(self.data, 'dates') and self.data.dates is not None:
                dates = self.data.dates._mpl_repr()
            else:
                dates = np.arange(self.nobs)
            llb = self.loglikelihood_burn

            # Plot the coefficient
            coef = self.recursive_coefficients
            ax.plot(dates[llb:], coef.filtered[variable, llb:],
                    label='Recursive estimates: %s' % exog_names[variable])

            # Legend
            handles, labels = ax.get_legend_handles_labels()

            # Get the critical value for confidence intervals
            if alpha is not None:
                critical_value = norm.ppf(1 - alpha / 2.)

                # Plot confidence intervals
                std_errors = np.sqrt(coef.filtered_cov[variable, variable, :])
                ci_lower = (
                    coef.filtered[variable] - critical_value * std_errors)
                ci_upper = (
                    coef.filtered[variable] + critical_value * std_errors)
                ci_poly = ax.fill_between(
                    dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2
                )
                ci_label = ('$%.3g \\%%$ confidence interval'
                            % ((1 - alpha)*100))

                # Only add CI to legend for the first plot
                if i == 0:
                    # Proxy artist for fill_between legend entry
                    # See http://matplotlib.org/1.3.1/users/legend_guide.html
                    p = plt.Rectangle((0, 0), 1, 1,
                                      fc=ci_poly.get_facecolor()[0])

                    handles.append(p)
                    labels.append(ci_label)

            ax.legend(handles, labels, loc=legend_loc)

            # Remove xticks for all but the last plot
            if i < k_variables - 1:
                ax.xaxis.set_ticklabels([])

        fig.tight_layout()

        return fig

    def _cusum_significance_bounds(self, alpha, ddof=0, points=None):
        """
        Parameters
        ----------
        alpha : float, optional
            The significance bound is alpha %.
        ddof : int, optional
            The number of periods additional to `k_exog` to exclude in
            constructing the bounds. Default is zero. This is usually used
            only for testing purposes.
        points : iterable, optional
            The points at which to evaluate the significance bounds. Default is
            two points, beginning and end of the sample.

        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lw, uw) because they burn the first k_exog + 1 periods instead of the
        first k_exog. If this change is performed
        (so that `tmp = (self.nobs - llb - 1)**0.5`), then the output here
        matches cusum6.

        The cusum6 behavior does not seem to be consistent with
        Brown et al. (1975); it is likely they did that because they needed
        three initial observations to get the initial OLS estimates, whereas
        we do not need to do that.
        """
        # Get the constant associated with the significance level
        if alpha == 0.01:
            scalar = 1.143
        elif alpha == 0.05:
            scalar = 0.948
        elif alpha == 0.10:
            scalar = 0.950
        else:
            raise ValueError('Invalid significance level.')

        # Get the points for the significance bound lines
        llb = self.loglikelihood_burn
        tmp = (self.nobs - llb - ddof)**0.5
        upper_line = lambda x: scalar * tmp + 2 * scalar * (x - llb) / tmp

        if points is None:
            points = np.array([llb, self.nobs])
        return -upper_line(points), upper_line(points)

    def plot_cusum(self, alpha=0.05, legend_loc='upper left',
                   fig=None, figsize=None):
        r"""
        Plot the CUSUM statistic and significance bounds.

        Parameters
        ----------
        alpha : float, optional
            The plotted significance bounds are alpha %.
        legend_loc : string, optional
            The location of the legend in the plot. Default is upper left.
        fig : Matplotlib Figure instance, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Evidence of parameter instability may be found if the CUSUM statistic
        moves out of the significance bounds.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.

        """
        # Create the plot
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        ax = fig.add_subplot(1, 1, 1)

        # Get dates, if applicable
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(self.nobs)
        llb = self.loglikelihood_burn

        # Plot cusum series and reference line
        ax.plot(dates[llb:], self.cusum, label='CUSUM')
        ax.hlines(0, dates[llb], dates[-1], color='k', alpha=0.3)

        # Plot significance bounds
        lower_line, upper_line = self._cusum_significance_bounds(alpha)
        ax.plot([dates[llb], dates[-1]], upper_line, 'k--',
                label='%d%% significance' % (alpha * 100))
        ax.plot([dates[llb], dates[-1]], lower_line, 'k--')

        ax.legend(loc=legend_loc)

        return fig

    def _cusum_squares_significance_bounds(self, alpha, points=None):
        """
        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lww, uww) because they use a different method for computing the
        critical value; in particular, they use tabled values from
        Table C, pp. 364-365 of "The Econometric Analysis of Time Series"
        Harvey, (1990), and use the value given to 99 observations for any
        larger number of observations. In contrast, we use the approximating
        critical values suggested in Edgerton and Wells (1994) which allows
        computing relatively good approximations for any number of
        observations.
        """
        # Get the approximate critical value associated with the significance
        # level
        llb = self.loglikelihood_burn
        n = 0.5 * (self.nobs - llb) - 1
        try:
            ix = [0.1, 0.05, 0.025, 0.01, 0.005].index(alpha / 2)
        except ValueError:
            raise ValueError('Invalid significance level.')
        scalars = _cusum_squares_scalars[:, ix]
        crit = scalars[0] / n**0.5 + scalars[1] / n + scalars[2] / n**1.5

        # Get the points for the significance bound lines
        if points is None:
            points = np.array([llb, self.nobs])
        line = (points - llb) / (self.nobs - llb)

        return line - crit, line + crit

    def plot_cusum_squares(self, alpha=0.05, legend_loc='upper left',
                           fig=None, figsize=None):
        r"""
        Plot the CUSUM of squares statistic and significance bounds.

        Parameters
        ----------
        alpha : float, optional
            The plotted significance bounds are alpha %.
        legend_loc : string, optional
            The location of the legend in the plot. Default is upper left.
        fig : Matplotlib Figure instance, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Evidence of parameter instability may be found if the CUSUM of squares
        statistic moves out of the significance bounds.

        Critical values used in creating the significance bounds are computed
        using the approximate formula of [1]_.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        .. [1] Edgerton, David, and Curt Wells. 1994.
           "Critical Values for the Cusumsq Statistic
           in Medium and Large Sized Samples."
           Oxford Bulletin of Economics and Statistics 56 (3): 355-65.

        """
        # Create the plot
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        ax = fig.add_subplot(1, 1, 1)

        # Get dates, if applicable
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(self.nobs)
        llb = self.loglikelihood_burn

        # Plot cusum series and reference line
        ax.plot(dates[llb:], self.cusum_squares, label='CUSUM of squares')
        ref_line = (np.arange(llb, self.nobs) - llb) / (self.nobs - llb)
        ax.plot(dates[llb:], ref_line, 'k', alpha=0.3)

        # Plot significance bounds
        lower_line, upper_line = self._cusum_squares_significance_bounds(alpha)
        ax.plot([dates[llb], dates[-1]], upper_line, 'k--',
                label='%d%% significance' % (alpha * 100))
        ax.plot([dates[llb], dates[-1]], lower_line, 'k--')

        ax.legend(loc=legend_loc)

        return fig


class RecursiveLSResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(RecursiveLSResultsWrapper, RecursiveLSResults)
