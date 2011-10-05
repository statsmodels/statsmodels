"""Stack loss data"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']


__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain. """
TITLE       = __doc__
SOURCE      = """
Brownlee, K. A. (1965), "Statistical Theory and Methodology in
Science and Engineering", 2nd edition, New York:Wiley.
"""

DESCRSHORT  = """Stack loss plant data of Brownlee (1965)"""

DESCRLONG   = """The stack loss plant data of Brownlee (1965) contains
21 days of measurements from a plant's oxidation of ammonia to nitric acid.
The nitric oxide pollutants are captured in an absorption tower."""

NOTE        = """
Number of Observations - 21

Number of Variables - 4

Variable name definitions::

    STACKLOSS - 10 times the percentage of ammonia going into the plant that
                escapes from the absoroption column
    AIRFLOW   - Rate of operation of the plant
    WATERTEMP - Cooling water temperature in the absorption tower
    ACIDCONC  - Acid concentration of circulating acid minus 50 times 10.
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.tools import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/stackloss.csv',"rb"), delimiter=",",
            names=True, dtype=float)
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
