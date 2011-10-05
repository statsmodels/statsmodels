"""Bill Greene's credit scoring data."""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

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

NOTE        = """
Number of observations - 72
Number of variables - 5
Variable name definitions - See Source for more information on the variables.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.tools.datautils import Dataset
from os.path import dirname, abspath

def load():
    """Load the credit card data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/ccard.csv', 'rb'), delimiter=",",
            names=True, dtype=float)
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] \
                    for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
