"""First 100 days of the US House of Representatives 1995"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission from the original author,
who retains all rights."""
TITLE       = __doc__
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unifited Approach`

http://jgill.wustl.edu/research/books.html
"""

DESCRSHORT  = """Number of bill assignments in the 104th House in 1995"""

DESCRLONG   = """The example in Gill, seeks to explain the number of bill
assignments in the first 100 days of the US' 104th House of Representatives.
The response variable is the number of bill assignments in the first 100 days
over 20 Committees.  The explanatory variables in the example are the number of
assignments in the first 100 days of the 103rd House, the number of members on
the committee, the number of subcommittees, the log of the number of staff
assigned to the committee, a dummy variable indicating whether
the committee is a high prestige committee, and an interaction term between
the number of subcommittees and the log of the staff size.

The data returned by load are not cleaned to represent the above example.
"""

NOTE = """::

    Number of Observations - 20
    Number of Variables - 6
    Variable name definitions::

        BILLS104 - Number of bill assignments in the first 100 days of the
                   104th House of Representatives.
        SIZE     - Number of members on the committee.
        SUBS     - Number of subcommittees.
        STAFF    - Number of staff members assigned to the committee.
        PRESTIGE - PRESTIGE == 1 is a high prestige committee.
        BILLS103 - Number of bill assignments in the first 100 days of the
                   103rd House of Representatives.

    Committee names are included as a variable in the data file though not
    returned by load.
"""

from numpy import recfromtxt, column_stack, array
from statsmodels.datasets import utils as du
from os.path import dirname, abspath

def load():
    """Load the committee data and returns a data class.

    Returns
    --------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_recarray(data, endog_idx=0, dtype=float)

def load_pandas():
    data = _get_data()
    return du.process_recarray_pandas(data, endog_idx=0, dtype=float)

def _get_data():
    filepath = dirname(abspath(__file__))
    with open(filepath + '/committee.csv', 'rb') as f:
        data = recfromtxt(f, delimiter=",",
                          names=True, dtype=float, usecols=(1,2,3,4,5,6))
    return data
