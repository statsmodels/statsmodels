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
from pandas import Series

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
    names = list(data.dtype.names)
    endog_name = 'volume'
    endog = array(data[endog_name], dtype=float)
    dataset = Dataset(data=data, names=[endog_name], endog=endog,
                      endog_name = endog_name)
    return dataset

def load_pandas():
    data = _get_data()
    # TODO: time series
    endog = Series(data['volume'], index=data['year'].astype(int))
    dataset = Dataset(data=data, names=list(data.dtype.names),
                      endog=endog, endog_name='volume')
    return dataset

def _get_data():
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/nile.csv', 'rb'), delimiter=",",
            names=True, dtype=float)
    return data
