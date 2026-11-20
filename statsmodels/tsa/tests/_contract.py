"""
Input forms for testing the documented contract of array_like arguments.

A 1-D time series documented as ``array_like`` should give the same result in
each of these forms as an ndarray does. Use them with
``pytest.mark.parametrize`` over ``SERIES_FORMS``::

    @pytest.mark.parametrize("form", SERIES_FORMS)
    def test_acf_input_forms(form):
        x = np.random.default_rng(0).standard_normal(100)
        assert_allclose(acf(series_forms(x)[form]), acf(x), rtol=1e-12)
"""

import numpy as np
import pandas as pd

SERIES_FORMS = ("list", "tuple", "series", "column", "dataframe")


def series_forms(x):
    """
    The forms of a 1-D series that ``array_like`` promises to accept

    Parameters
    ----------
    x : array_like
        A 1-D series.

    Returns
    -------
    dict of str to object
        Keyed by the names in ``SERIES_FORMS``: a list, a tuple, a Series and
        a one-column DataFrame with a monthly DatetimeIndex, and an (n, 1)
        column.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    index = pd.date_range("2000-01-01", periods=x.shape[0], freq="MS")
    return {
        "list": x.tolist(),
        "tuple": tuple(x.tolist()),
        "series": pd.Series(x, index=index, name="x"),
        "column": x[:, None],
        "dataframe": pd.DataFrame({"x": x}, index=index),
    }
