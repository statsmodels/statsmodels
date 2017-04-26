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

This class will load a subset of the full survey data. The subset includes three questions:

1) "Which candidate do you think you are most likely to ultimately vote for in the 2016 presidential election?"
    The answers are in columns 1 through 6, with a '1' representing 'selected' and '0'
    representing 'not selected'. Each respondent was only allowed to select 1 candidate,
    making this a traditional single select categorical variable.
2) "Please check all of the statements you believe are true". Respondents were presented with a list of
    assertions (shown in columns 7 through 11). Respondents could check as many options as they liked,
    making this a multiple response question.
3) "Please select any factors that contribute to your not being sure who you'll vote for". The reasons shown
    in columns 12 through 16 were shown and respondents could pick as many as they liked,
    making this also a multiple response question. 

The full survey data, including the answers to 11 additional questions, is available in
the /src subdirectory.

IMPORTANT NOTE: in the original survey from which these responses were
 collected, questions #2 and #3 included additional answer choices. Because 
 there were many answer choices, only a randomly selected subset of answer
 choices was displayed to each respondent. In this data set, answer choices that
 were not selected are marked with a '0' regardless of whether they were 
 not selected because 1) they were not displayed or 2) because they were 
 displayed but not selected. So please be extremely careful about drawing 
 any conclusions about the world from this data.
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
