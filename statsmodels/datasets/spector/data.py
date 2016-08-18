"""Spector and Mazzeo (1980) - Program Effectiveness Data"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission of the original author, who
retains all rights. """
TITLE       = __doc__
SOURCE      = """
http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm

The raw data was downloaded from Bill Greene's Econometric Analysis web site,
though permission was obtained from the original researcher, Dr. Lee Spector,
Professor of Economics, Ball State University."""

DESCRSHORT  = """Experimental data on the effectiveness of the personalized
system of instruction (PSI) program"""

DESCRLONG   = DESCRSHORT

NOTE        = """::

    Number of Observations - 32

    Number of Variables - 4

    Variable name definitions::

        Grade - binary variable indicating whether or not a student's grade
                improved.  1 indicates an improvement.
        TUCE  - Test score on economics test
        PSI   - participation in program
        GPA   - Student's grade point average
"""

import numpy as np
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

def load():
    """
    Load the Spector dataset and returns a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray(data, endog_idx=3, dtype=float)

def load_pandas():
    """
    Load the Spector dataset and returns a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray_pandas(data, endog_idx=3, dtype=float)

def _get_data():
    filepath = dirname(abspath(__file__))
    ##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    with open(filepath + '/spector.csv',"rb") as f:
        data = np.recfromtxt(f, delimiter=" ",
                             names=True, dtype=float, usecols=(1,2,3,4))
    return data
