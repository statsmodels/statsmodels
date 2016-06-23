#! /usr/bin/env python

"""Name of dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Engel (1857) food expenditure data"""
SOURCE      = """
This dataset was used in Koenker and Bassett (1982) and distributed alongside
the ``quantreg`` package for R.

Koenker, R. and Bassett, G (1982) Robust Tests of Heteroscedasticity based on
Regression Quantiles; Econometrica 50, 43-61.

Roger Koenker (2012). quantreg: Quantile Regression. R package version 4.94.
http://CRAN.R-project.org/package=quantreg
"""

DESCRSHORT  = """Engel food expenditure data."""

DESCRLONG   = """Data on income and food expenditure for 235 working class households in 1857 Belgium."""

#suggested notes
NOTE        = """::

    Number of observations: 235
    Number of variables: 2
    Variable name definitions:
        income - annual household income (Belgian francs)
        foodexp - annual household food expenditure (Belgian francs)
"""

import numpy as np
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    ##### SET THE INDICES #####
    #NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray(data, endog_idx=0, exog_idx=None, dtype=float)

def load_pandas():
    data = _get_data()
    ##### SET THE INDICES #####
    #NOTE: None for exog_idx is the complement of endog_idx
    return du.process_recarray_pandas(data, endog_idx=0, exog_idx=None,
                                      dtype=float)

def _get_data():
    filepath = dirname(abspath(__file__))
    ##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    with open(filepath + '/engel.csv', 'rb') as f:
        data = np.recfromtxt(f,
                             delimiter=",", names = True, dtype=float)
    return data
