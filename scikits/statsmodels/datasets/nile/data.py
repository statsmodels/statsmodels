__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

"""Name of dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """E.g., This is public domain."""
TITLE       = """Title of the dataset"""
SOURCE      = """
This section should provide a link to the original dataset if possible and
attribution and correspondance information for the dataset's original author
if so desired.
"""

DESCRSHORT  = """A short description."""

DESCRLONG   = """A longer description of the dataset."""

#suggested notes
NOTE        = """
Number of observations:
Number of variables:
Variable name definitions:

Any other useful information that does not fit into the above categories.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.tools import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the Nile data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    data = recfromtxt(open(filepath + '/nile.csv', 'rb'), delimiter=",",
            names=True, dtype=float, usecols=(1))
    names = list(data.dtype.names)
##### SET THE INDEX #####
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
##### SET THE INDEX #####
#    exog = column_stack(data[i] for i in names[1:]).astype(float)
#    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog,
            endog_name = endog_name)
    return dataset
