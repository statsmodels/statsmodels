"""
Self-Exciting Threshold Autoregression

References
----------

Hansen, Bruce. 1999.
"Testing for Linearity."
Journal of Economic Surveys 13 (5): 551-576.
"""

from __future__ import division
import numpy as np
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.tsatools import add_constant, lagmat
from statsmodels.regression.linear_model import OLS

class InvalidRegimeError(ValueError):
    pass


class SETAR(tsbase.TimeSeriesModel):
    """
    Self-Exciting Threshold Autoregressive Model

    Parameters
    ----------
    endog : array-like
        The endogenous variable.
    order : integer
        The order of the SETAR model, indication the number of regimes.
    ar_order : integer
        The order of the autoregressive parameters.
    delay : integer, optional
        The delay for the self-exciting threshold variable.
    thresholds : iterable, optional
        The threshold values separating the data into regimes.
    min_regime_frac : scalar, optional
        The minumum fraction of observations in each regime.
    max_delay : integer, optional
        The maximum delay parameter to consider if a grid search is used. If
        left blank, it is set to be the ar_order.
    threshold_grid_size : integer, optional
        The number of elements in the threshold grid if a grid search is used.
    """

    # TODO are there too many parameters here?
    def __init__(self, endog, order, ar_order,
                 delay=None, thresholds=None, min_regime_frac=0.1,
                 max_delay=None, threshold_grid_size=100,
                 dates=None, freq=None, missing='none'):
        super(SETAR, self).__init__(endog, None, dates, freq)

        if delay is not None and delay < 1 or delay > ar_order:
            raise ValueError('Delay parameter must be greater than zero'
                             ' and less than ar_order. Got %d.' % delay)

        # Unsure of statistical properties if length of sample changes when
        # estimating hyperparameters, which happens if delay can be greater
        # than ar_order, so that the number of initial observations changes
        if delay is None and max_delay > ar_order:
            raise ValueError('Maximum delay for grid search must not be '
                             ' greater than the autoregressive order.')

        if thresholds is not None and not len(thresholds)+1 == order:
            raise ValueError('Number of thresholds must match'
                             ' the order of the SETAR model')

        # Exogenous matrix
        self.exog = add_constant(lagmat(self.endog, ar_order))
        self.nobs_initial = ar_order
        self.nobs = len(self.endog) - ar_order

        # "Immutable" properties
        self.order = order
        self.ar_order = ar_order
        self.min_regime_frac = min_regime_frac
        self.min_regime_num = np.ceil(min_regime_frac * self.nobs)
        self.max_delay = max_delay if max_delay is not None else ar_order
        self.threshold_grid_size = threshold_grid_size

        # "Flexible" properties
        self.delay = delay
        # TODO I sort in case thresholds are in wrong order, but that seems
        #      like it may be wasteful, since it won't usually be the case?
        #      and feels like user error anyway...
        self.thresholds = np.sort(thresholds)
        self.regimes = None

    def build_datasets(self, delay, thresholds, order=None):
        if order is None:
            order = self.order

        endog = self.endog[self.nobs_initial:, ]
        exog_transpose = self.exog[self.nobs_initial:, ].T
        threshold_var = self.endog[self.nobs_initial-delay:-delay, ]
        indicators = np.searchsorted(thresholds, threshold_var)

        k = self.ar_order + 1
        exog_list = []
        for i in range(order):
            in_regime = (indicators == i)

            if in_regime.sum() < self.min_regime_num:
                raise InvalidRegimeError('Regime %d has too few observations:'
                                         ' threshold values may need to be'
                                         ' adjusted' % i)

            exog_list.append(np.multiply(exog_transpose, indicators == i).T)

        exog = np.concatenate(exog_list, 1)

        return endog, exog

    def fit(self):
        """
        Fits SETAR() model using arranged autoregression.

        Returns
        -------
        statsmodels.tsa.arima_model.SETARResults class

        See also
        --------
        statsmodels.regression.linear_model.OLS : this estimates each regime
        SETARResults : results class returned by fit

        """

        if self.delay is None or self.thresholds is None:
            self.delay, self.thresholds = self.select_hyperparameters()

        endog, exog = self.build_datasets(self.delay, self.thresholds)

        # TODO implement the SETARResults class to nicely show all
        #      regimes' results
        # TODO really just doing OLS on this dataset...is there a better way to
        #      do this?
        return OLS(endog, exog).fit()

    def select_hyperparameters(self):
        """
        Select delay and threshold hyperparameters via grid search
        """

        raise NotImplementedError


class SETARResults:
    pass
