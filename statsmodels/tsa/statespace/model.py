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
    State space model

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

    Attributes
    ----------
    start_params : array
        Starting parameters for maximum likelihood estimation.
    params_names : list of str
        List of human readable parameter names (for parameters actually
        included in the model).
    model_names : list of str
        The plain text names of all possible model parameters.
    model_latex_names : list of str
        The latex names of all possible model parameters.

    See Also
    --------
    statsmodels.tsa.statespace.KalmanFilter
    """
    def __init__(self, endog, k_states, exog=None, dates=None, freq=None,
                 *args, **kwargs):
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
        super(Model, self).__init__(endog.shape[0], k_states, *args, **kwargs)
        # Bind the data to the model
        self.bind(endog)
