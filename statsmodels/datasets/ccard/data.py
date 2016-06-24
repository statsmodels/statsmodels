"""Bill Greene's credit scoring data."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission of the original author, who
retains all rights."""
TITLE       = __doc__
SOURCE      = """
William Greene's `Econometric Analysis`

More information can be found at the web site of the text:
http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm
"""

DESCRSHORT  = """William Greene's credit scoring data"""

DESCRLONG   = """More information on this data can be found on the
homepage for Greene's `Econometric Analysis`. See source.
"""

NOTE        = """::

    Number of observations - 72
    Number of variables - 5
    Variable name definitions - See Source for more information on the
                                variables.
"""

from numpy import recfromtxt
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

def load():
    """Load the credit card data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray(data, endog_idx=0, dtype=float)

def load_pandas():
    """Load the credit card data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray_pandas(data, endog_idx=0)

def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + "/ccard.csv", 'rb') as f:
        data = recfromtxt(f, delimiter=",", names=True, dtype=float)
    return data
