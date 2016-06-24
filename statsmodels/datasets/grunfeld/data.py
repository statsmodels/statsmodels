"""Grunfeld (1950) Investment Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = __doc__
SOURCE      = """This is the Grunfeld (1950) Investment Data.

The source for the data was the original 11-firm data set from Grunfeld's Ph.D.
thesis recreated by Kleiber and Zeileis (2008) "The Grunfeld Data at 50".
The data can be found here.
http://statmath.wu-wien.ac.at/~zeileis/grunfeld/

For a note on the many versions of the Grunfeld data circulating see:
http://www.stanford.edu/~clint/bench/grunfeld.htm
"""

DESCRSHORT  = """Grunfeld (1950) Investment Data for 11 U.S. Firms."""

DESCRLONG   = DESCRSHORT

NOTE        = """::

    Number of observations - 220 (20 years for 11 firms)

    Number of variables - 5

    Variables name definitions::

        invest  - Gross investment in 1947 dollars
        value   - Market value as of Dec. 31 in 1947 dollars
        capital - Stock of plant and equipment in 1947 dollars
        firm    - General Motors, US Steel, General Electric, Chrysler,
                Atlantic Refining, IBM, Union Oil, Westinghouse, Goodyear,
                Diamond Match, American Steel
        year    - 1935 - 1954

    Note that raw_data has firm expanded to dummy variables, since it is a
    string categorical variable.
"""

from numpy import recfromtxt, column_stack, array
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

def load():
    """
    Loads the Grunfeld data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    raw_data has the firm variable expanded to dummy variables for each
    firm (ie., there is no reference dummy)
    """
    from statsmodels.tools import categorical
    data = _get_data()
    raw_data = categorical(data, col='firm', drop=True)
    ds = du.process_recarray(data, endog_idx=0, stack=False)
    ds.raw_data = raw_data
    return ds

def load_pandas():
    """
    Loads the Grunfeld data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    raw_data has the firm variable expanded to dummy variables for each
    firm (ie., there is no reference dummy)
    """
    from pandas import DataFrame
    from statsmodels.tools import categorical
    data = _get_data()
    raw_data = categorical(data, col='firm', drop=True)
    ds = du.process_recarray_pandas(data, endog_idx=0)
    ds.raw_data = DataFrame(raw_data)
    return ds

def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + '/grunfeld.csv','rb') as f:
        data = recfromtxt(f, delimiter=",",
                          names=True, dtype="f8,f8,f8,a17,f8")
    return data
