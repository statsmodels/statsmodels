#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Common functions used in preparing or validating model inputs.

"""

import numpy as np
import pandas as pd

from statsmodels.tools.data import _is_using_pandas



def prepare_exog(exog):
    k_exog = 0
    # Exogenous data
    if exog is not None:
        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)

        # Make sure we have 2-dimensional array
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)

        k_exog = exog.shape[1]
    return (k_exog, exog)

