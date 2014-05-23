"""Wage Data"""

__docformat__ = 'csv'

COPYRIGHT   = """public domain"""
TITLE       = __doc__
SOURCE      = """
M. Blackburn and D. Neumark (1992), "Unobserved Ability, Efficiency Wages, and
Interindustry Wage Differentials," Quarterly Journal of Economics 107, 1421-1436.
"""

DESCRSHORT  = """Wage data."""

DESCRLONG   = DESCRSHORT

NOTE        = """ """

from numpy import recfromtxt, column_stack, array, log
import numpy.lib.recfunctions as nprf
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

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
    data = np.recfromtxt(open(filepath + '/wage1.csv', 'rb'),
            delimiter=",", names = True, dtype=float)
    return data
