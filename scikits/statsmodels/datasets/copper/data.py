__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

"""World Copper Prices 1951-1975 dataset."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission from the original author,
who retains all rights."""
TITLE       = "World Copper Market 1951-1975 Dataset"
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unified Approach`

http://jgill.wustl.edu/research/books.html
"""

DESCRSHORT  = """World Copper Market 1951-1975"""

DESCRLONG   = """This data describes the world copper market from 1951 through 1975.  In an
example, in Gill, the outcome variable (of a 2 stage estimation) is the world
consumption of copper for the 25 years.  The explanatory variables are the
world consumption of copper in 1000 metric tons, the constant dollar adjusted
price of copper, the price of a substitute, aluminum, an index of real per
capita income base 1970, an annual measure of manufacturer inventory change,
and a time trend.
"""

NOTE = """
Number of Observations - 25

Number of Variables - 6

Variable name definitions::

    WORLDCONSUMPTION - World consumption of copper (in 1000 metric tons)
    COPPERPRICE - Constant dollar adjusted price of copper
    INCOMEINDEX - An index of real per capita income (base 1970)
    ALUMPRICE - The price of aluminum
    INVENTORYINDEX - A measure of annual manufacturer inventory trend
    TIME - A time trend

Years are included in the data file though not returned by load.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.tools.datautils import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the copper data and returns a Dataset class.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()

    names = list(data.dtype.names)
    endog_name = names[0]
    exog_name = names[1:]

    endog = array(data[names[0]], dtype=float)
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name = endog_name, exog_name=exog_name)
    return dataset

def _get_data():
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/copper.csv', 'rb'), delimiter=",",
                      names=True, dtype=float, usecols=(1,2,3,4,5,6))
    return data

def load_pandas():
    pass
