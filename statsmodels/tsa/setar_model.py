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

        if delay < 1 or delay >= len(endog):
            raise ValueError('Delay parameter must be greater than zero'
                             ' and less than nobs. Got %d.' % delay)

        if thresholds is not None and not len(thresholds)+1 == order:
            raise ValueError('Number of thresholds must match'
                             ' the order of the SETAR model')

        # "Immutable" properties
        self.order = order
        self.k_ar = ar_order
        self.min_regime_frac = min_regime_frac
        self.max_delay = max_delay if max_delay is not None else ar_order
        self.threshold_grid_size = threshold_grid_size

        # "Flexible" properties
        self.delay = delay
        # TODO I sort in case thresholds are in wrong order, but that seems
        #      like it may be wasteful, since it won't usually be the case?
        #      and feels like user error anyway...
        self.thresholds = np.sort(thresholds)
        self.regimes = None

    def _get_regime_indicators(self, delay, thresholds):
        """
        Generate an indicator vector of regimes (0, ..., order-1)
        """
        return np.r_[
            [np.NaN]*delay,
            np.searchsorted(self.thresholds, self.endog[:-delay])
        ]

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

        nobs_initial = max(self.k_ar, self.delay)
        nobs = len(self.endog) - nobs_initial

        indicators = self._get_regime_indicators(self.delay, self.thresholds)
        indicator_matrix = (indicators[:, None] == range(self.order))

        lags = add_constant(lagmat(self.endog, self.k_ar))

        exog = np.multiply(
            np.bmat('lags '*self.order),
            np.kron(indicator_matrix, np.ones(self.k_ar+1))
        )[nobs_initial:, :]
        endog = self.endog[nobs_initial:, ]

        # Make sure each regime has enough datapoints
        if indicator_matrix.sum(0).min() < np.ceil(nobs*self.min_regime_frac):
            # TODO is this the right exception to throw?
            raise ValueError('Regime %d has too few observations:'
                             ' threshold values may need to be adjusted' %
                             indicator_matrix.sum(0).argmin())

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
