"""Yearly sunspots data 1700-2008"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This data is public domain."""
TITLE       = __doc__
SOURCE      = """
http://www.ngdc.noaa.gov/stp/SOLAR/ftpsunspotnumber.html

The original dataset contains monthly data on sunspot activity in the file
./src/sunspots_yearly.dat.  There is also sunspots_monthly.dat.
"""

DESCRSHORT  = """Yearly (1700-2008) data on sunspots from the National
Geophysical Data Center."""

DESCRLONG   = DESCRSHORT

NOTE        = """
Number of Observations - 309 (Annual 1700 - 2008)
Number of Variables - 1
Variable name definitions::

    SUNACTIVITY - Number of sunspots for each year

The data file contains a 'YEAR' variable that is not returned by load.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.tools import Dataset
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
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/sunspots.csv', 'rb'), delimiter=",",
            names=True, dtype=float, usecols=(1))
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names
    dataset = Dataset(data=data, names=names, endog=endog,
            endog_name=endog_name)
    return dataset

