"""
Shared result type for the AR/ARMA parameter estimation functions in
statsmodels.tsa.arima.estimators.

Author: Chad Fulton
License: BSD-3
"""
from typing import NamedTuple


class ARMAEstimationResult(NamedTuple):
    """
    Result of an ARIMA parameter estimator.

    Common to :func:`~statsmodels.tsa.arima.estimators.burg.burg`,
    :func:`~statsmodels.tsa.arima.estimators.gls.gls`,
    :func:`~statsmodels.tsa.arima.estimators.hannan_rissanen.hannan_rissanen`,
    :func:`~statsmodels.tsa.arima.estimators.innovations.innovations`,
    :func:`~statsmodels.tsa.arima.estimators.innovations.innovations_mle`,
    :func:`~statsmodels.tsa.arima.estimators.yule_walker.yule_walker`,
    :func:`~statsmodels.tsa.arima.estimators.durbin_levinson.durbin_levinson`,
    and :func:`~statsmodels.tsa.arima.estimators.statespace.statespace`.

    Parameters
    ----------
    parameters : SARIMAXParams or list of SARIMAXParams
        Contains the parameter estimates from the final iteration. A list
        of ``SARIMAXParams`` objects for :func:`innovations` and
        :func:`durbin_levinson`, which estimate at multiple orders; a
        single ``SARIMAXParams`` object for all other estimators.
    other_results : Bunch
        Estimator-specific ancillary results, always including at least
        ``spec``, the ``SARIMAXSpecification`` instance corresponding to
        the input arguments. See the individual estimator's docstring for
        the full set of components.
    """

    parameters: object
    other_results: object
