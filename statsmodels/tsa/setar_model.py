"""
Self-Exciting Threshold Autoregression

References
----------

Hansen, Bruce. 1999.
"Testing for Linearity."
Journal of Economic Surveys 13 (5): 551-576.

Hansen, Bruce E. 1997.
"Inference in TAR Models."
Studies in Nonlinear Dynamics & Econometrics 2 (1) (January 1).

"""

from __future__ import division
import numpy as np
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.tsatools import add_constant, lagmat
from statsmodels.regression.linear_model import OLS


class InvalidRegimeError(ValueError):
    pass


class SETAR(OLS, tsbase.TimeSeriesModel):
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
    trend : str {'c','nc'}
        Whether to include a constant or not
        'c' includes constant
        'nc' no constant
    min_regime_frac : scalar, optional
        The minumum fraction of observations in each regime.
    max_delay : integer, optional
        The maximum delay parameter to consider if a grid search is used. If
        left blank, it is set to be the ar_order.
    threshold_grid_size : integer, optional
        The approximate number of elements in the threshold grid if a grid
        search is used.


    Notes
    -----
    threshold_grid_size is only approximate because it uses values from the
    threshold variable itself, approximately evenly spaced, and there may be a
    few more elements in the grid search than requested


    References
    ----------
    See Hansen (1997) Table 1 for threshold critical values.

    """

    threshold_crits = {
        0.8:   4.50,    0.85: 5.10,     0.9: 5.94,
        0.925: 6.53,    0.95: 7.35,     0.975: 8.75,
        0.99:  10.59
    }

    # TODO are there too many parameters here?
    def __init__(self, endog, order, ar_order,
                 delay=None, thresholds=None, trend='c',
                 min_regime_frac=0.1, max_delay=None, threshold_grid_size=100,
                 dates=None, freq=None, missing='none'):

        if delay is not None and delay < 1 or delay > ar_order:
            raise ValueError('Delay parameter must be greater than zero'
                             ' and less than ar_order. Got %d.' % delay)

        # Unsure of statistical properties if length of sample changes when
        # estimating hyperparameters, which happens if delay can be greater
        # than ar_order, so that the number of initial observations changes
        if delay is None and max_delay > ar_order:
            raise ValueError('Maximum delay for grid search must not be '
                             ' greater than the autoregressive order.')

        if delay is None and thresholds is not None:
            raise ValueError('Thresholds cannot be specified without delay'
                             ' parameter.')

        if thresholds is not None and not len(thresholds) + 1 == order:
            raise ValueError('Number of thresholds must match'
                             ' the order of the SETAR model')

        # "Immutable" properties
        self.nobs_initial = ar_order
        self.nobs = endog.shape[0] - ar_order

        self.order = order
        self.ar_order = ar_order
        self.k_trend = int(trend == 'c')
        self.min_regime_frac = min_regime_frac
        self.min_regime_num = np.ceil(min_regime_frac * self.nobs)
        self.max_delay = max_delay if max_delay is not None else ar_order
        self.threshold_grid_size = threshold_grid_size

        # "Flexible" properties
        self.delay = delay
        self.thresholds = thresholds
        if self.thresholds:
            self.thresholds = np.sort(self.thresholds)

        # Estimation properties
        self.nobs_regimes = None
        self.objectives = {}
        self.ar1_resids = None

        # Make a copy of original datasets
        orig_endog = endog
        orig_exog = lagmat(orig_endog, ar_order)

        # Trends
        if self.k_trend:
            orig_exog = add_constant(orig_exog)

        # Create datasets / complete initialization
        endog = orig_endog[self.nobs_initial:]
        exog = orig_exog[self.nobs_initial:]
        super(SETAR, self).__init__(endog, exog,
                                    hasconst=self.k_trend, missing=missing)

        # Overwrite originals
        self.data.orig_endog = orig_endog
        self.data.orig_exog = orig_exog

    def initialize(self):
        """
        Initialize datasets

        Since we manipulate exog and endog as the delay and thresholds are
        changed / selected, this function (and its parent) are called to keep
        all variables up-to-date (mostly making sure shapes are the same)
        """
        self.data.endog = self.endog
        self.data.exog = self.exog
        self.weights = np.repeat(1., self.endog.shape[0])

        super(SETAR, self).initialize()

    def build_datasets(self, delay, thresholds):
        """
        Build the endogenous vector and exogenous matrix for SETAR(m)
        estimation.

        Primary purpose is to construct the exogenous dataset, which is the
        matrix of lags (up to ar_order, plus a constant term) horizontally
        duplicated once each for the number of regimes. Each duplication has
        the rows for which the model dicatates another regime set to zero.

        Also returns the endogenous vector of appropriate size (i.e. reduced by
        nobs_initial because the model is conditional on those observations).

        Parameters
        ----------
        delay : integer
            The delay for the self-exciting threshold variable.
        thresholds : iterable
            The threshold values separating the data into regimes.

        Returns
        -------
        endog : array-like
            Engodenous variable, (nobs - nobs_initial) x 1
        exog : array-like
            Exogenous matrix,
            (nobs - nobs_initial) x [(ar_order + k_trend) * order]
        nobs_regimes : iterable
            Number of observations in each regime
        """

        order = len(thresholds) + 1

        exog_transpose = self.exog.T
        threshold_var = self.exog[:, delay]
        indicators = np.searchsorted(thresholds, threshold_var)

        k = self.ar_order + self.k_trend
        exog_list = []
        nobs_regimes = ()
        for i in range(order):
            in_regime = (indicators == i)
            nobs_regime = in_regime.sum()

            if nobs_regime < self.min_regime_num:
                raise InvalidRegimeError('Regime %d has too few observations:'
                                         ' threshold values may need to be'
                                         ' adjusted' % i)

            exog_list.append(np.multiply(exog_transpose, indicators == i).T)
            nobs_regimes += (nobs_regime,)

        exog = np.concatenate(exog_list, 1)

        return self.endog, exog, nobs_regimes

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

        self.endog, self.exog, self.nobs_regimes = self.build_datasets(
            self.delay, self.thresholds
        )
        self.initialize()

        beta = self._fit()
        lfit = SETARResults(
            self, beta, normalized_cov_params=self.normalized_cov_params
        )

        return lfit

    def _grid_search_objective(self, delay, thresholds, XX, resids):
        """
        Objective function to maximize in SETAR(2) hyperparameter grid search

        Corresponds to f_2(\gamma, d) in Hansen (1999), but extended to any
        number of thresholds.

        Parameters
        ----------
        delay : integer
            The delay for the self-exciting threshold variable.
        thresholds : iterable
            The threshold values separating the data into regimes.
        XX : array-like
            (X'X)^{-1} from a SETAR(1) specification (i.e. AR(1))
        resids : array-like
            The residuals from a SETAR(1) specification (i.e. AR(1))
        """
        endog, exog, _ = self.build_datasets(delay, thresholds)

        # Intermediate calculations
        k = self.ar_order + self.k_trend
        X1 = exog[:, :-k]
        X = self.exog
        X1X1 = X1.T.dot(X1)
        XX1 = X.T.dot(X1)
        Mn = np.linalg.inv(
            X1X1 - XX1.T.dot(XX).dot(XX1)
        )

        # Return objective
        return resids.T.dot(X1).dot(Mn).dot(X1.T).dot(resids)

    def _select_hyperparameters_grid(self, thresholds, threshold_grid_size,
                                     XX, resids, delay_grid=None):

        if delay_grid is None:
            delay_grid = range(1, self.max_delay + 1)

        max_obj = 0
        params = (None, None)
        # Iterate over possible delay values
        for delay in delay_grid:

            # Build the appropriate threshold grid given delay
            threshold_var = np.unique(np.sort(self.endog[:-delay]))
            nobs = len(threshold_var)
            indices = np.arange(self.min_regime_num,
                                nobs - self.min_regime_num,
                                max(np.floor(nobs / threshold_grid_size), 1),
                                dtype=int)
            threshold_grid = threshold_var[indices]

            # Iterate over possible threshold values
            for threshold in threshold_grid:
                if threshold in thresholds:
                    continue
                try:
                    iteration_thresholds = np.sort([threshold] + thresholds)
                    obj = self._grid_search_objective(
                        delay, iteration_thresholds,
                        XX, resids
                    )
                    self.objectives[(delay,)+tuple(iteration_thresholds)] = obj
                    if obj > max_obj:
                        max_obj = obj
                        params = (delay, threshold)
                # Some threshold values don't allow enough values in each
                # regime; we just need to not select those thresholds
                except InvalidRegimeError:
                    pass

        return params

    def select_hyperparameters(self, threshold_grid_size=None, maxiter=100):
        """
        Select delay and threshold hyperparameters via grid search
        """

        # Cache calculations
        XX = np.linalg.inv(self.exog.T.dot(self.exog))    # (X'X)^{-1}
        self.ar1_resids = resids = self.endog - np.dot(   # SETAR(1) residuals
            self.exog,
            XX.dot(self.exog.T.dot(self.endog))
        )

        # Get default threshold grid size, if necessary
        if threshold_grid_size is None:
            threshold_grid_size = self.threshold_grid_size

        # Set delay grid if delay is specified
        delay_grid = [self.delay] if self.delay is not None else None

        # Estimate the delay and an initial value for the dominant threshold
        thresholds = []
        delay, threshold = self._select_hyperparameters_grid(
            thresholds, threshold_grid_size, XX, resids, delay_grid=delay_grid
        )
        thresholds.append(threshold)

        # Get remaining thresholds
        for i in range(2, self.order):

            # Get initial estimate of next threshold
            _, threshold = self._select_hyperparameters_grid(
                thresholds, threshold_grid_size, XX, resids,
                delay_grid=[delay]
            )
            thresholds.append(threshold)

            # Iterate threshold selection to convergence
            proposed = thresholds[:]
            iteration = 0
            while True:
                iteration += 1

                # Recalculate each threshold individually, holding the others
                # constant, starting at the first threshold
                for j in range(i):
                    _, threshold = self._select_hyperparameters_grid(
                        thresholds[:j] + thresholds[j + 1:],
                        threshold_grid_size, XX, resids,
                        delay_grid=[delay]
                    )
                    proposed[j] = threshold

                # If the recalculation produced no change, we've converged
                if proposed == thresholds:
                    break
                # If convergence is not happening fast enough
                if iteration > maxiter:
                    print ('Warning: Maximum number of iterations has been '
                           'exceeded.')
                    break

                thresholds = proposed[:]

        return delay, np.sort(thresholds)


class SETARResults:
    pass
