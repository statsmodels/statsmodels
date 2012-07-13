"""Name of dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Misra"""
SOURCE      = """
http://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml
"""

DESCRSHORT  = """Misra, D., NIST (1978).  
               Dental Research Monomolecular Adsorption Study."""

DESCRLONG   = """These data are the result of a NIST study regarding
               dental research in monomolecular adsorption.  The
               response variable is volume, and the predictor
               variable is pressure."""

#suggested notes
NOTE        = """
Number of observations:14
Number of variables:2
Variable name definitions:1 Response Variable  (y = volume)
                          1 Predictor Variable (x = pressure)
Procedure:Nonlinear Least Squares Regression
Model:y = b1*(1-exp[-b2*x])  +  e

"""

from numpy import recfromtxt, column_stack, array
from pandas import Series, DataFrame
import statsmodels.tools.datautils as du
from statsmodels.tools import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the Misra data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    
    data = _get_data()
    return du.process_recarray(data, endog_idx=0, dtype=float)

def _get_data():
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/misra.csv', 'rb'), delimiter=",",
            names=True, dtype=float)
    return data
