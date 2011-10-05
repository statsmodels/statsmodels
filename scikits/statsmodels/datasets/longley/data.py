"""Longley dataset"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = __doc__
SOURCE      = """
The classic 1967 Longley Data

http://www.itl.nist.gov/div898/strd/lls/data/Longley.shtml

::

    Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the
        Electronic Comptuer from the Point of View of the User."  Journal of
        the American Statistical Association.  62.319, 819-41.
"""

DESCRSHORT  = """"""

DESCRLONG   = """The Longley dataset contains various US macroeconomic
variables that are known to be highly collinear.  It has been used to appraise
the accuracy of least squares routines."""

NOTE        = """
Number of Observations - 16

Number of Variables - 6

Variable name definitions::

        TOTEMP - Total Employment
        GNPDEFL - GNP deflator
        GNP - GNP
        UNEMP - Number of unemployed
        ARMED - Size of armed forces
        POP - Population
        YEAR - Year (1947 - 1962)
"""

from numpy import recfromtxt, array, column_stack
from scikits.statsmodels.tools.datautils import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the Longley data and return a Dataset class.

    Returns
    -------
    Dataset instance
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath+'/longley.csv',"rb"), delimiter=",", names=True,
            dtype=float, usecols=(1,2,3,4,5,6,7))
    names = list(data.dtype.names)
    endog = array(data[names[0]], dtype=float)
    endog_name = names[0]
    exog = column_stack(data[i] for i in names[1:]).astype(float)
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name=endog_name, exog_name = exog_name)
    return dataset
