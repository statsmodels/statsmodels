#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Re-shaping various array-like objects.

"""
import numpy as np
import pandas as pd


def atleast_2dcols(x):
    """Ensure an input array is at least two-dimensional.  If a new dimension
    is added, the output array has 1 column.

    Parameters
    ----------
    x : array_like
       The array that needs to be made at least two-dimensional

    Returns
    -------
    y : array
        Values in this array are identical to the inputs, but an extra
        dimension may have been added

    Notes
    -----
    This is similar to np.atleast_2d.  The key difference is where the new
    dimension is inserted.  If a 1-dimensional array is passed to
    np.atleast_2d, the result will have 1 row that matches the original input.
    If the same array is passed to atleast_2dcols, the result will have 1
    column that matches the original input.

    Examples
    --------
    >>> np.atleast_2d(np.arange(5)).shape
    (1, 5)
    >>> atleast_2dcols(np.arange(5)).shape
    (5, 1)

    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x
