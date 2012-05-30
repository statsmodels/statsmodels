"""Munnell Productivity Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Munnell Productivity Data"""
SOURCE      = """This is the Munnell (1990) and Baltagi (2005) Productivity Data.

This dataset is used in Greene's Econometrics Analysis in Example 10.1 (and others).

Data are available at : http://www.wiley.com/legacy/wileychi/baltagi/supp/PRODUC.prn
"""

DESCRSHORT  = """Munnell Productivity Data, 48 Continental U.S. States, 17 years (1970-1986)"""

DESCRLONG   = DESCRSHORT

#suggested notes
NOTE        = """
Number of observations: 816 (48 States, 17 years)
Number of variables: 7
Variable name definitions:
   gsp = gross state product
   pc = private capital
   hwy = highway capital
   water = water utility capital
   util = utility capital
   emp = employment (labor) 
   unemp = unemployment rate
   p_cap = ?
"""

import numpy as np
from statsmodels.tools import categorical
from statsmodels.tools import datautils as du
from os.path import dirname, abspath

def load():
    """
    Load the Munnell data and return a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    raw_data has the state variable expanded to dummy variables for each
    state (ie., there is no reference dummy)
    """
    data = _get_data()
    raw_data = categorical(data, col='state', drop=True)
    ds = du.process_recarray(data, endog_idx=7, stack=False)
    ds.raw_data = raw_data
    return ds

def load_pandas():
    from pandas import DataFrame  
    data = _get_data()
    raw_data = categorical(data, col='state', drop=True)
    ds = du.process_recarray(data, endog_idx=7, stack=False)
    ds.raw_data = DataFrame(raw_data) # why not DataFrame(data) ?
    return ds

def _get_data():
    filepath = dirname(abspath(__file__))
    data = np.recfromtxt(open(filepath + '/munnell.csv','rb'),delimiter=',',names=True)
    return data

