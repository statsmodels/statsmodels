from statsmodels.compat.pandas import PD_LT_2

import functools

import numpy as np


def _safe_is_pandas_categorical_dtype(dt):
    """Check whether a dtype is a pandas categorical dtype across versions"""
    import pandas as pd

    if PD_LT_2:
        return pd.api.types.is_categorical_dtype(dt)
    return isinstance(dt, pd.CategoricalDtype)


def monkey_patch_cat_dtype():
    """Patch patsy to use a version-compatible categorical dtype check"""
    try:
        import patsy.util

        patsy.util.safe_is_pandas_categorical_dtype = _safe_is_pandas_categorical_dtype
    except ImportError:
        # Future protection is using formulaic for formulas without patsy
        pass


# Vendored from patsy.util
def get_all_sorted_knots(
    x, n_inner_knots=None, inner_knots=None, lower_bound=None, upper_bound=None
):
    """
    Get all knots locations with lower and upper exterior knots included

    If needed, inner knots are computed as equally spaced quantiles of the
    input data falling between given lower and upper bounds.

    Parameters
    ----------
    x : ndarray
        The 1-d array data values.
    n_inner_knots : int, optional
        Number of inner knots to compute.
    inner_knots : array_like, optional
        Provided inner knots if any.
    lower_bound : float, optional
        The lower exterior knot location. If unspecified, the
        minimum of ``x`` values is used.
    upper_bound : float, optional
        The upper exterior knot location. If unspecified, the
        maximum of ``x`` values is used.

    Returns
    -------
    ndarray
        The array of ``n_inner_knots + 2`` distinct knots.

    Raises
    ------
    ValueError
        For various invalid parameter sets or if unable to compute
        ``n_inner_knots + 2`` distinct knots.
    """
    if lower_bound is None and x.size == 0:
        raise ValueError(
            "Cannot set lower exterior knot location: empty "
            "input data and lower_bound not specified."
        )
    elif lower_bound is None and x.size != 0:
        lower_bound = np.min(x)

    if upper_bound is None and x.size == 0:
        raise ValueError(
            "Cannot set upper exterior knot location: empty "
            "input data and upper_bound not specified."
        )
    elif upper_bound is None and x.size != 0:
        upper_bound = np.max(x)

    if upper_bound < lower_bound:
        raise ValueError(
            f"lower_bound > upper_bound ({lower_bound!r} > {upper_bound!r})"
        )

    if inner_knots is None and n_inner_knots is not None:
        if n_inner_knots < 0:
            raise ValueError(
                f"Invalid requested number of inner knots: {n_inner_knots!r}"
            )

        x = x[(lower_bound <= x) & (x <= upper_bound)]
        x = np.unique(x)

        if x.size != 0:
            inner_knots_q = np.linspace(0, 100, n_inner_knots + 2)[1:-1]
            # .tolist() is necessary to work around a bug in numpy 1.8
            inner_knots = np.asarray(np.percentile(x, inner_knots_q.tolist()))
        elif n_inner_knots == 0:
            inner_knots = np.array([])
        else:
            raise ValueError(
                f"No data values between lower_bound(={lower_bound!r}) and "
                f"upper_bound(={upper_bound!r}): cannot compute requested "
                f"{n_inner_knots!r} inner knot(s)."
            )
    elif inner_knots is not None:
        inner_knots = np.unique(inner_knots)
        if n_inner_knots is not None and n_inner_knots != inner_knots.size:
            raise ValueError(
                f"Needed number of inner knots={n_inner_knots!r} does not match "
                f"provided number of inner knots={inner_knots.size!r}."
            )
        n_inner_knots = inner_knots.size
        if np.any(inner_knots < lower_bound):
            raise ValueError(
                f"Some knot values ({inner_knots[inner_knots < lower_bound]}) fall below lower bound "
                f"({lower_bound!r})."
            )
        if np.any(inner_knots > upper_bound):
            raise ValueError(
                f"Some knot values ({inner_knots[inner_knots > upper_bound]}) fall above upper bound "
                f"({upper_bound!r})."
            )
    else:
        raise ValueError("Must specify either 'n_inner_knots' or 'inner_knots'.")

    all_knots = np.concatenate(([lower_bound, upper_bound], inner_knots))
    all_knots = np.unique(all_knots)
    if all_knots.size != n_inner_knots + 2:
        raise ValueError(
            f"Unable to compute n_inner_knots(={n_inner_knots!r}) + 2 distinct "
            f"knots: {x.size!r} data value(s) found between "
            f"lower_bound(={lower_bound!r}) and upper_bound(={upper_bound!r})."
        )

    return all_knots


@functools.cache
def ensure_patsy_compat():
    """Apply the patsy categorical dtype compatibility patch, if possible"""
    try:
        import patsy.util  # noqa: F401
    except ImportError:
        # patsy not installed skip applying the patch now
        return
    try:
        monkey_patch_cat_dtype()
    except Exception:
        # Intentionally ignored to avoid breaking import time behavior
        return
