"""Yearly sunspots data 1700-2008"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This data is public domain."""
TITLE       = __doc__
SOURCE      = """
http://www.ngdc.noaa.gov/stp/solar/solarda3.html

The original dataset contains monthly data on sunspot activity in the file
./src/sunspots_yearly.dat.  There is also sunspots_monthly.dat.
"""

DESCRSHORT  = """Yearly (1700-2008) data on sunspots from the National
Geophysical Data Center."""

DESCRLONG   = DESCRSHORT

NOTE        = """::

    Number of Observations - 309 (Annual 1700 - 2008)
    Number of Variables - 1
    Variable name definitions::

        SUNACTIVITY - Number of sunspots for each year

    The data file contains a 'YEAR' variable that is not returned by load.
"""

from numpy import recfromtxt, array
from pandas import Series, DataFrame

from statsmodels.datasets.utils import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the yearly sunspot data and returns a data class.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    This dataset only contains data for one variable, so the attributes
    data, raw_data, and endog are all the same variable.  There is no exog
    attribute defined.
    """
    data = _get_data()
    endog_name = 'SUNACTIVITY'
    endog = array(data[endog_name], dtype=float)
    dataset = Dataset(data=data, names=[endog_name], endog=endog,
                      endog_name=endog_name)
    return dataset

def load_pandas():
    data = DataFrame(_get_data())
    # TODO: time series
    endog = Series(data['SUNACTIVITY'], index=data['YEAR'].astype(int))
    dataset = Dataset(data=data, names=list(data.columns),
                      endog=endog, endog_name='volume')
    return dataset

def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + '/sunspots.csv', 'rb') as f:
        data = recfromtxt(f, delimiter=",",
                          names=True, dtype=float)
        return data
