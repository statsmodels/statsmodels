"""Breast Cancer Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """???"""
TITLE       = """Breast Cancer Data"""
SOURCE      = """
This is the breast cancer data used in Owen's empirical likelihood.  It is taken from
Rice, J.A. Mathematical Statistics and Data Analysis.
http://www.cengage.com/statistics/discipline_content/dataLibrary.html
"""

DESCRSHORT  = """Breast Cancer and county population"""

DESCRLONG   = """The number of breast cancer observances in various counties"""

#suggested notes
NOTE        = """::

    Number of observations: 301
    Number of variables: 2
    Variable name definitions:

        cancer - The number of breast cancer observances
        population - The population of the county

"""
from statsmodels.datasets import utils as du


def load_pandas():
    data = _get_data()
    ##### SET THE INDICES #####
    #NOTE: None for exog_idx is the complement of endog_idx
    return du.process_pandas(data, endog_idx=0, exog_idx=None)


def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    return du.as_numpy_dataset(load_pandas())


def _get_data():
    return du.load_csv(__file__, 'cancer.csv', convert_float=True)
