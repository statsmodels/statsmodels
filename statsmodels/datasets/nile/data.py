"""Nile River Flows."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Nile River flows at Ashwan 1871-1970"""
SOURCE      = """
This data is first analyzed in:

    Cobb, G. W. 1978. "The Problem of the Nile: Conditional Solution to a
        Changepoint Problem." *Biometrika*. 65.2, 243-51.
"""

DESCRSHORT  = """This dataset contains measurements on the annual flow of
the Nile as measured at Ashwan for 100 years from 1871-1970."""

DESCRLONG   = DESCRSHORT + " There is an apparent changepoint near 1898."

#suggested notes
NOTE        = """::

    Number of observations: 100
    Number of variables: 2
    Variable name definitions:

        year - the year of the observations
        volumne - the discharge at Aswan in 10^8, m^3
"""

from numpy import recfromtxt, array
from pandas import Series, DataFrame

from statsmodels.datasets.utils import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the Nile data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    names = list(data.dtype.names)
    endog_name = 'volume'
    endog = array(data[endog_name], dtype=float)
    dataset = Dataset(data=data, names=[endog_name], endog=endog,
                      endog_name=endog_name)
    return dataset

def load_pandas():
    data = DataFrame(_get_data())
    # TODO: time series
    endog = Series(data['volume'], index=data['year'].astype(int))
    dataset = Dataset(data=data, names=list(data.columns),
                      endog=endog, endog_name='volume')
    return dataset

def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + '/nile.csv', 'rb') as f:
        data = recfromtxt(f, delimiter=",",
                          names=True, dtype=float)
    return data
