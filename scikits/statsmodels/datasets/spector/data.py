"""Spector and Mazzeo (1980) - Program Effectiveness Data"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

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

NOTE        = """
Number of Observations - 32

Number of Variables - 4

Variable name definitions::

    Grade - binary variable indicating whether or not a student's grade
            improved.  1 indicates an improvement.
    TUCE  - Test score on economics test
    PSI   - participation in program
    GPA   - Student's grade point average
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.tools import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the Spector dataset and returns a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    data = recfromtxt(open(filepath + '/spector.csv',"rb"), delimiter=" ",
            names=True, dtype=float, usecols=(1,2,3,4))
    names = list(data.dtype.names)
    endog = array(data[names[3]], dtype=float)
    endog_name = names[3]
    exog = column_stack(data[i] for i in names[:3]).astype(float)
    exog_name = names[:3]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
