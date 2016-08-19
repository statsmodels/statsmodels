"""El Nino dataset, 1950 - 2010"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This data is in the public domain."""

TITLE       = """El Nino - Sea Surface Temperatures"""

SOURCE      = """
National Oceanic and Atmospheric Administration's National Weather Service

ERSST.V3B dataset, Nino 1+2
http://www.cpc.ncep.noaa.gov/data/indices/
"""

DESCRSHORT  = """Averaged monthly sea surface temperature - Pacific Ocean."""

DESCRLONG   = """This data contains the averaged monthly sea surface
temperature in degrees Celcius of the Pacific Ocean, between 0-10 degrees South
and 90-80 degrees West, from 1950 to 2010.  This dataset was obtained from
NOAA.
"""

NOTE = """::

    Number of Observations - 61 x 12

    Number of Variables - 1

    Variable name definitions::

        TEMPERATURE - average sea surface temperature in degrees Celcius
                      (12 columns, one per month).
"""


from numpy import recfromtxt, column_stack, array
from pandas import DataFrame

from statsmodels.datasets.utils import Dataset
from os.path import dirname, abspath


def load():
    """
    Load the El Nino data and return a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The elnino Dataset instance does not contain endog and exog attributes.
    """
    data = _get_data()
    names = data.dtype.names
    dataset = Dataset(data=data, names=names)
    return dataset


def load_pandas():
    dataset = load()
    dataset.data = DataFrame(dataset.data)
    return dataset


def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + '/elnino.csv', 'rb') as f:
        data = recfromtxt(f, delimiter=",",
                          names=True, dtype=float)
    return data
