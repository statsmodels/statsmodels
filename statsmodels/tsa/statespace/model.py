"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .representation import Representation
from .kalman_filter import KalmanFilter

import statsmodels.tsa.base.tsa_model as tsbase


class Model(KalmanFilter, Representation, tsbase.TimeSeriesModel):
    """
    State space representation of a time series process, with Kalman filter and
    Statsmodels integration.

    This intermediate class joins the state space representation and filtering
    classes with the Statsmodels `TimeSeriesModel`.

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
        matrices, for Kalman filtering options, for Kalman smoothing
        options, or for Simulation smoothing options.
        See `Representation`, `KalmanFilter`, and `KalmanSmoother` for more
        details.

    See Also
    --------
    statsmodels.tsa.statespace.tsa.base.tsa_model.TimeSeriesModel
    statsmodels.tsa.statespace.mlemodel.MLEModel
    statsmodels.tsa.statespace.kalman_filter.KalmanFilter
    statsmodels.tsa.statespace.representation.Representation
    """
    def __init__(self, endog, k_states, exog=None, dates=None, freq=None,
                 **kwargs):
        # Initialize the model base
        tsbase.TimeSeriesModel.__init__(self, endog=endog, exog=exog,
                                        dates=dates, freq=freq, missing='none')

        # Need to modify the endog variable
        endog = self.endog

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

        # Now endog is C-ordered and in long format (nobs x k_endog). To get
        # F-ordered and in wide format just need to transpose.
        endog = endog.T

        # Initialize the statespace representation
        super(Model, self).__init__(endog.shape[0], k_states, **kwargs)
        # Bind the data to the model
        self.bind(endog)
