from __future__ import annotations

import warnings

import numpy as np
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


_INFO_CRITERIA = ("aic", "aicc", "bic", "hqic")


def _safe_arma_fit(y, order, model_kw, trend, fit_kw):
    """Return (results, None), or (None, error) if the fit raises error."""
    from statsmodels.tsa.arima.model import ARIMA

    try:
        with warnings.catch_warnings():
            # A candidate model that falls back to zeros as starting values
            warnings.filterwarnings(
                "ignore", message="Non-invertible starting MA", category=EstimationWarning
            )
            return ARIMA(y, order=order, **model_kw, trend=trend).fit(**fit_kw), None
    except TypeError:
        # Comes from the arguments, for example a mistyped keyword in
        # model_kw or fit_kw, and not from a model that cannot be fitted
        raise
    except Exception as error:
        # For example a LinAlgError or a ValueError from a badly misspecified
        # model, or a ValueError from an estimator that cannot fit this order
        return None, error


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
        Information criteria to report, from "aic", "aicc", "bic" and
        "hqic". Either a single name or a sequence of names.

        .. versionchanged:: 0.15.1

           An unknown name raises a ValueError before any model is fitted.
           It used to raise an AttributeError from the first fitted model.
    trend : {'n', 'c'}, optional
        The trend to use when fitting the ARMA models.
    model_kw : dict, optional
        Keyword arguments to be passed to the ``ARIMA`` model.
    fit_kw : dict, optional
        Keyword arguments to be passed to ``ARIMA.fit``.

        .. versionchanged:: 0.15.1

           A TypeError from ``model_kw`` or ``fit_kw``, such as one for a
           mistyped keyword, is raised. It used to make every fit fail.

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
        If ``ic`` contains an unknown name, if the fit of every model fails,
        or if no model has a non-NaN value of a criterion in ``ic``.
    TypeError
        If ``model_kw`` or ``fit_kw`` are not valid arguments of ``ARIMA``
        or ``ARIMA.fit``, for example because of a mistyped keyword.

    Warns
    -----
    EstimationWarning
        If the fits of some, but not all, models fail. The warning lists their
        orders and the error raised by the first.

        .. versionadded:: 0.15.1

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
    unknown = [name for name in ic if name not in _INFO_CRITERIA]
    if unknown:
        raise ValueError(
            f"Unknown information criteria in ic: {unknown}. The valid names "
            f"are {list(_INFO_CRITERIA)}."
        )

    results = np.zeros((len(ic), max_ar + 1, max_ma + 1))
    model_kw = {} if model_kw is None else model_kw
    fit_kw = {} if fit_kw is None else fit_kw
    y_arr = array_like(y, "y", contiguous=True)
    failed = []
    for ar in ar_range:
        for ma in ma_range:
            mod, error = _safe_arma_fit(y_arr, (ar, 0, ma), model_kw, trend, fit_kw)
            if mod is None:
                failed.append(((ar, ma), error))
                results[:, ar, ma] = np.nan
                continue

            for i, criteria in enumerate(ic):
                results[i, ar, ma] = getattr(mod, criteria)
    if failed:
        order, error = failed[0]
        first = f"The first, order {order}, raised {type(error).__name__}: {error}"
        nmodels = len(ar_range) * len(ma_range)
        if len(failed) == nmodels:
            raise ValueError(
                "No ARMA model could be fitted: the fit failed for every order "
                f"(p, q) with p <= {max_ar} and q <= {max_ma}. {first}"
            ) from error
        orders = ", ".join(str(pq) for pq, _ in failed)
        warnings.warn(
            f"The fits of {len(failed)} of {nmodels} ARMA models failed, and "
            f"their information criteria are NaN. Orders (p, q): {orders}. "
            f"{first}",
            EstimationWarning,
            stacklevel=2,
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
