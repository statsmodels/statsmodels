from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd

from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
    array_like,
    dict_like,
    int_like,
    string_like,
)

ArrayLike1D = np.ndarray | pd.Series | list[float]


def _safe_arma_fit(y, order, model_kw, trend, fit_kw, start_params=None):
    from statsmodels.tsa.arima.model import ARIMA

    try:
        with warnings.catch_warnings():
            # A candidate model that falls back to zeros as starting values
            warnings.filterwarnings(
                "ignore", message="Non-invertible starting MA", category=EstimationWarning
            )
            return ARIMA(y, order=order, **model_kw, trend=trend).fit(
                start_params=start_params, **fit_kw
            )
    except LinAlgError:
        # SVD convergence failure on badly misspecified models
        return None

    except ValueError as error:
        if start_params is not None:  # do not recurse again
            # user supplied start_params only get one chance
            return None
        # try a little harder, should be handled in fit really
        elif "initial" not in error.args[0] or "initial" in str(error):
            start_params = [0.1] * sum(order)
            if trend == "c":
                start_params = [0.1] + start_params
            return _safe_arma_fit(y, order, model_kw, trend, fit_kw, start_params)
        else:
            return None
    except Exception:  # no idea what happened
        return None


def arma_order_select_ic(
    y, max_ar=4, max_ma=2, ic="bic", trend="c", model_kw=None, fit_kw=None
):
    """
    Compute information criteria for many ARMA models

    Parameters
    ----------
    y : array_like
        Array of time-series data.
    max_ar : int, optional
        Maximum number of AR lags to use. Default 4.
    max_ma : int, optional
        Maximum number of MA lags to use. Default 2.
    ic : str or sequence of str, optional
        Information criteria to report. Either a single string or a list
        of different criteria is possible.
    trend : {'n', 'c'}, optional
        The trend to use when fitting the ARMA models.
    model_kw : dict, optional
        Keyword arguments to be passed to the ``ARIMA`` model.
    fit_kw : dict, optional
        Keyword arguments to be passed to ``ARIMA.fit``.

    Returns
    -------
    Bunch
        Dict-like object with attribute access. Each ic is an attribute with a
        DataFrame for the results. The AR order used is the row index. The ma
        order used is the column index. The minimum orders are available as
        ``ic_min_order``, a tuple ``(p, q)`` with the AR and MA orders of the
        model with the smallest criterion. Models with a NaN criterion, such
        as those whose fit failed, are ignored.

        .. versionchanged:: 0.15.1

           The minimum orders ignore models with a NaN criterion. They used
           to point at the first such model whenever a fit had failed.

    Raises
    ------
    ValueError
        If the fit of every model fails, or if no model has a non-NaN value
        of a criterion in ``ic``.

    Notes
    -----
    This method can be used to tentatively identify the order of an ARMA
    process, provided that the time series is stationary and invertible. This
    function computes the full exact MLE estimate of each model and can
    therefore be a little slow. A faster, approximate estimator of
    ``ARIMA.fit`` can be selected with fit_kw, for example
    ``fit_kw={"method": "hannan_rissanen"}``.

    A model whose fit fails has NaN information criteria, and the minimum
    orders ignore it.

    Examples
    --------

    >>> from statsmodels.tsa.arima_process import arma_generate_sample
    >>> import statsmodels.api as sm
    >>> import numpy as np

    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> arparams = np.r_[1, -arparams]
    >>> maparams = np.r_[1, maparams]
    >>> nobs = 250
    >>> np.random.seed(2014)
    >>> y = arma_generate_sample(arparams, maparams, nobs)
    >>> res = sm.tsa.arma_order_select_ic(y, ic=["aic", "bic"], trend="n")
    >>> res.aic_min_order
    >>> res.bic_min_order
    """
    max_ar = int_like(max_ar, "max_ar")
    max_ma = int_like(max_ma, "max_ma")
    if max_ar < 0 or max_ma < 0:
        # a negative bound used to make one of the order ranges empty and
        # leak a bare numpy error from the argmin over the empty grid
        raise ValueError(
            f"max_ar and max_ma must be non-negative, got max_ar={max_ar}, "
            f"max_ma={max_ma}"
        )
    trend = string_like(trend, "trend", options=("n", "c"))
    model_kw = dict_like(model_kw, "model_kw", optional=True)
    fit_kw = dict_like(fit_kw, "fit_kw", optional=True)

    ar_range = list(range(max_ar + 1))
    ma_range = list(range(max_ma + 1))
    if isinstance(ic, str):
        ic = [ic]
    else:
        msg = "ic must be a str or a sequence of str"
        try:
            ic = list(ic)
        except TypeError:
            raise ValueError(msg) from None
        if not all(isinstance(name, str) for name in ic):
            raise ValueError(msg)
        # Elements of an ndarray are np.str_
        ic = [str(name) for name in ic]

    results = np.zeros((len(ic), max_ar + 1, max_ma + 1))
    model_kw = {} if model_kw is None else model_kw
    fit_kw = {} if fit_kw is None else fit_kw
    y_arr = array_like(y, "y", contiguous=True)
    nfailed = 0
    for ar in ar_range:
        for ma in ma_range:
            mod = _safe_arma_fit(y_arr, (ar, 0, ma), model_kw, trend, fit_kw)
            if mod is None:
                nfailed += 1
                results[:, ar, ma] = np.nan
                continue

            for i, criteria in enumerate(ic):
                results[i, ar, ma] = getattr(mod, criteria)
    if nfailed == len(ar_range) * len(ma_range):
        raise ValueError(
            "No ARMA model could be fitted: the fit failed for every order "
            f"(p, q) with p <= {max_ar} and q <= {max_ma}."
        )

    dfs = [pd.DataFrame(res, columns=ma_range, index=ar_range) for res in results]

    res = dict(zip(ic, dfs, strict=True))

    # add the minimums to the results dict, ignoring the NaN criteria of
    # failed fits
    min_res = {}
    for i, result in res.items():
        values = result.to_numpy()
        if np.isnan(values).all():
            raise ValueError(
                f"No fitted ARMA model has a non-NaN {i}, so {i}_min_order "
                "is undefined."
            )
        ncols = values.shape[1]
        loc = np.nanargmin(values)
        min_res.update({i + "_min_order": (loc // ncols, loc % ncols)})
    res.update(min_res)

    return Bunch(**res)
