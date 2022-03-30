"""
Created on Mar 30, 2022 1:21:54 PM

Author: Josef Perktold
License: BSD-3
"""

import numpy as np


def _mover_confint(stat1, stat2, ci1, ci2, contrast="diff"):

    if contrast == "diff":
        stat = stat1 - stat2
        low_half = np.sqrt((stat1 - ci1[0])**2 + (stat2 - ci2[1])**2)
        upp_half = np.sqrt((stat1 - ci1[1])**2 + (stat2 - ci2[0])**2)
        ci = (stat - low_half, stat + upp_half)

    elif contrast == "sum":
        stat = stat1 + stat2
        low_half = np.sqrt((stat1 - ci1[0])**2 + (stat2 - ci2[0])**2)
        upp_half = np.sqrt((stat1 - ci1[1])**2 + (stat2 - ci2[1])**2)
        ci = (stat - low_half, stat + upp_half)

    elif contrast == "ratio":
        # stat = stat1 / stat2
        prod = stat1 * stat2
        term1 = stat2**2 - (ci2[1] - stat2)**2
        term2 = stat2**2 - (ci2[0] - stat2)**2
        low = (prod -
               np.sqrt(prod**2 - term1 * (stat1**2 - (ci1[0] - stat1)**2))
               ) / term1
        upp = (prod +
               np.sqrt(prod**2 - term2 * (stat1**2 - (ci1[1] - stat1)**2))
               ) / term2
        ci = (low, upp)

    return ci


def _mover_confint_sum(stat, ci):

        stat_ = stat.sum(0)
        low_half = np.sqrt(np.sum((stat_ - ci[0])**2))
        upp_half = np.sqrt(np.sum((stat_ - ci[1])**2))
        ci = (stat - low_half, stat + upp_half)
        return ci
