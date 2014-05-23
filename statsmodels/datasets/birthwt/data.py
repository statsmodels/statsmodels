"""Birthweight Data"""

__docformat__ = 'csv'

COPYRIGHT   = """????"""
TITLE       = __doc__
SOURCE      = """
Hosmer, D.W. and Lemeshow, S. (1989) Applied Logistic Regression. New York: Wiley
"""

DESCRSHORT  = """Birth weight data."""

DESCRLONG   = """Data on risk factors associated with low birth rate."""

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
    data = np.recfromtxt(open(filepath + '/birthwt.csv', 'rb'),
            delimiter=",", names = True, dtype=float)
    return data
