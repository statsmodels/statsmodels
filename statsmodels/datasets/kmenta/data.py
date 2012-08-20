"""Partly Artificial Data on the U. S. Economy"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Partly Artificial Data on the U. S. Economy"""
SOURCE      = """Kmenta, J. (1986) Elements of Econometrics, Second Edition, Macmillan.

CSV file is extracted from R package systemfit.

The data are available from Table ̃13-1 (p. ̃687), and the results are presented in Table ̃13-2 (p. ̃712) of this book.
"""

DESCRSHORT  = """Kmenta dataset on food market (example on page 685)"""

DESCRLONG   = DESCRSHORT

NOTE        = """
Number of observations: 20
Number of variables: 5
Variable name definitions:
    consump = 
    price = 
    income = 
    farmPrice =
    trend = 
"""

import numpy as np
from statsmodels.tools import categorical
from statsmodels.tools import datautils as du
from os.path import dirname, abspath

def load():
    """
    Load the Kmenta data and return a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    """
    data = _get_data()
    #raw_data = categorical(data, col='state', drop=True)
    ds = du.process_recarray(data, endog_idx=[0, 1], stack=False)
    #ds.raw_data = raw_data
    return ds

def load_pandas():
    from pandas import DataFrame  
    data = _get_data()
    #raw_data = categorical(data, col='state', drop=True)
    ds = du.process_recarray(data, endog_idx=[0, 1], stack=False)
    #ds.raw_data = DataFrame(raw_data) # why not DataFrame(data) ?
    return ds

def _get_data():
    filepath = dirname(abspath(__file__))
    data = np.recfromtxt(open(filepath + '/src/kmenta.csv','rb'),delimiter=',',names=True)
    return data

