"""Survata Presidential Pre-Election Survey Data (2016)"""
from statsmodels.datasets.utils import Dataset

__docformat__ = 'restructuredtext'

COPYRIGHT = """This is BSD licensed."""
TITLE = __doc__
SOURCE = """Survata Presidential Pre-Election Survey Data (2016)

This survey was conducted by Survata, an independent research firm in San Francisco.

Survata interviewed 1000 online respondents between September 02, 2016 and September 11, 2016.

Respondents were reached across the Survata publisher network, where they take a survey to
unlock premium content, like articles and ebooks.

Respondents received no cash compensation for their participation.

More information on Survata's methodology can be found at survata.com/methodology.
"""

DESCRSHORT = """Pre-Election Survey Data For U.S.A Undecided Swing State Voters (2016)"""

DESCRLONG = DESCRSHORT

NOTE = """::

    Number of observations - 1000 (including 1 incomplete response)

"""

from numpy import recfromtxt
from os.path import dirname, abspath


def load():
    """
    Loads the 2016 Presidential data and returns a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    names = list(data.dtype.names)
    dataset = Dataset(data=data, names=names)
    return dataset


def load_pandas():
    from pandas import DataFrame

    data = DataFrame(_get_data())
    dataset = Dataset(data=data, names=list(data.columns))
    return dataset


def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + '/presidential_survey_filtered.csv', 'rb') as f:
        data = recfromtxt(f, delimiter=",", names=True, dtype=float)
    return data
