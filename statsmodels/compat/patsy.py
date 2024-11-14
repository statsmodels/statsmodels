from statsmodels.compat.pandas import PD_LT_2

import numpy as np
import pandas as pd


def _safe_is_pandas_categorical_dtype(dt):
    if PD_LT_2:
        return pd.api.types.is_categorical_dtype(dt)
    return isinstance(dt, pd.CategoricalDtype)


def monkey_patch_cat_dtype():
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
    """Gets all knots locations with lower and upper exterior knots included.

    If needed, inner knots are computed as equally spaced quantiles of the
    input data falling between given lower and upper bounds.

    :param x: The 1-d array data values.
    :param n_inner_knots: Number of inner knots to compute.
    :param inner_knots: Provided inner knots if any.
    :param lower_bound: The lower exterior knot location. If unspecified, the
     minimum of ``x`` values is used.
    :param upper_bound: The upper exterior knot location. If unspecified, the
     maximum of ``x`` values is used.
    :return: The array of ``n_inner_knots + 2`` distinct knots.

    :raise ValueError: for various invalid parameters sets or if unable to
     compute ``n_inner_knots + 2`` distinct knots.
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
            "lower_bound > upper_bound (%r > %r)" % (lower_bound, upper_bound)
        )

    if inner_knots is None and n_inner_knots is not None:
        if n_inner_knots < 0:
            raise ValueError(
                "Invalid requested number of inner knots: %r" % (n_inner_knots,)
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
                "No data values between lower_bound(=%r) and "
                "upper_bound(=%r): cannot compute requested "
                "%r inner knot(s)." % (lower_bound, upper_bound, n_inner_knots)
            )
    elif inner_knots is not None:
        inner_knots = np.unique(inner_knots)
        if n_inner_knots is not None and n_inner_knots != inner_knots.size:
            raise ValueError(
                "Needed number of inner knots=%r does not match "
                "provided number of inner knots=%r." % (n_inner_knots, inner_knots.size)
            )
        n_inner_knots = inner_knots.size
        if np.any(inner_knots < lower_bound):
            raise ValueError(
                "Some knot values (%s) fall below lower bound "
                "(%r)." % (inner_knots[inner_knots < lower_bound], lower_bound)
            )
        if np.any(inner_knots > upper_bound):
            raise ValueError(
                "Some knot values (%s) fall above upper bound "
                "(%r)." % (inner_knots[inner_knots > upper_bound], upper_bound)
            )
    else:
        raise ValueError("Must specify either 'n_inner_knots' or 'inner_knots'.")

    all_knots = np.concatenate(([lower_bound, upper_bound], inner_knots))
    all_knots = np.unique(all_knots)
    if all_knots.size != n_inner_knots + 2:
        raise ValueError(
            "Unable to compute n_inner_knots(=%r) + 2 distinct "
            "knots: %r data value(s) found between "
            "lower_bound(=%r) and upper_bound(=%r)."
            % (n_inner_knots, x.size, lower_bound, upper_bound)
        )

    return all_knots
