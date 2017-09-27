#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Re-shaping various array-like objects.

"""
import numpy as np
import pandas as pd


def atleast_2dcols(x):
    """

    This is similar to np.atleast_2d.  The key difference is where the new
    dimension is inserted.  If a 1-dimensional array is passed to np.atleast_2d,
    the result will have 1 row that matches the original input.  If the same
    array is passed to atleast_2dcols, the result will have 1 column that
    matches the original input.

    >>> np.atleast_2d(np.arange(5)).shape
    (1, 5)
    >>> atleast_2dcols(np.arange(5)).shape
    (5, 1)
    
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x
