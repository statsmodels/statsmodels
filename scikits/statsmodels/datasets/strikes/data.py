#! /usr/bin/env python
"""U.S. Strike Duration Data"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = __doc__
SOURCE      = """
This is a subset of the data used in Kennan (1985). It was originally
published by the Bureau of Labor Statistics.

::

    Kennan, J. 1985. "The duration of contract strikes in US manufacturing.
        `Journal of Econometrics` 28.1, 5-28.
"""

DESCRSHORT  = """Contains data on the length of strikes in US manufacturing and
unanticipated industrial production."""

DESCRLONG   = """Contains data on the length of strikes in US manufacturing and
unanticipated industrial production. The data is a subset of the data originally
used by Kennan. The data here is data for the months of June only to avoid
seasonal issues."""

#suggested notes
NOTE        = """
Number of observations - 62

Number of variables - 2

Variable name definitions::

            duration - duration of the strike in days
            iprod - unanticipated industrial production
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the strikes data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    data = recfromtxt(open(filepath + '/strikes.csv', 'rb'), delimiter=",",
            names=True, dtype=float)
    names = list(data.dtype.names)
##### SET THE INDEX #####
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
##### SET THE INDEX #####
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
