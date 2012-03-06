# -*- coding: utf-8 -*-
"""Convenience Wrappers

Created on Sat Oct 30 14:56:35 2010

Author: josef-pktd
License: BSD
"""

import numpy as np
import statsmodels.api as sm
from statsmodels import GLS, WLS, OLS

def remove_nanrows(y, x):
    '''remove common rows in [y,x] that contain at least one nan

    TODO: this should be made more flexible,
     arbitrary number of arrays and 1d or 2d arrays

    duplicate: Skipper added sm.tools.drop_missing

    '''
    mask = ~np.isnan(y)
    mask *= ~(np.isnan(x).any(-1))  #* or &
    y = y[mask]
    x = x[mask]
    return y, x


def linmod(y, x, weights=None, sigma=None, add_const=True, filter_missing=True,
           **kwds):
    '''get linear model with extra options for entry

    dispatches to regular model class and does not wrap the output

    If several options are exclusive, for example sigma and weights, then the
    chosen class depends on the implementation sequence.
    '''

    if filter_missing:
        y, x = remove_nanrows(y, x)
        #do the same for masked arrays

    if add_const:
        x = sm.add_constant(x, prepend=True)

    if not sigma is None:
        return GLS(y, x, sigma=sigma, **kwds)
    elif not weights is None:
        return WLS(y, x, weights=weights, **kwds)
    else:
        return OLS(y, x, **kwds)

