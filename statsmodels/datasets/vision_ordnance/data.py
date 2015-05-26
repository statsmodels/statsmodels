"""Eye testing case records."""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """The Biometrika trust"""
TITLE       = __doc__
SOURCE      = """
R irr package, credited there to:

Stuart, A. (1953). The Estimation and Comparison of Strengths of
Association in Contingency Tables. Biometrika, 40, 105-110.
"""

DESCRSHORT  = """Contingency table of left and right eye test results."""

DESCRLONG = """This is an 4x4 contingency table of vision test results
for the left and right eye in a sample of 7,477 distinct women.  The
women were employed in the Royal Ordnance factories, and the tests
were conducted between 1943 and 1946.  The test outcome is an ordinal
grade from 1 to 4."""

NOTE        = """::

    Number of Observations - 7,477
    Number of Variables - 2
    Variable name definitions::

        left - the grade recorded for the left eye
        right - the grade recorded for the right eye
"""

import numpy as np
import pandas as pd
import utils
import os


def load():
    """
    Load the eye testing case records data set and return a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """

    filepath = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(filepath + '/vision_table.csv'))
    return utils.Dataset(data=data, title="Eye testing results")
