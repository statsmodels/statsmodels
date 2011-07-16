# -*- coding: utf-8 -*-
"""World Fertility Survey: Fiji"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']


__docformat__ = 'restructuredtext'

COPYRIGHT   = """Available for use in academic research.  See SOURCE."""
TITLE       = __doc__
SOURCE      = """
The source data was obtained from Germa¡n Rodriguez's web site at Princeton
http://data.princeton.edu/wws509/datasets/#ceb, with the following refernce.

::

    Little, R. J. A. (1978). Generalized Linear Models for Cross-Classified Data
        from the WFS. World Fertility Survey Technical Bulletins, Number 5.

It originally comes from the World Fertility Survey for Fiji
http://opr.princeton.edu/archive/wfs/fj.aspx.

The terms of use for the original dataset are:

Data may be used for academic research, provided that credit is given in any
publication resulting from the research to the agency that conducted the
survey and that two copies of any publication are sent to::

 	Mr. Naibuku Navunisaravi
    Government Statistician
    Bureau of Statistics
    Government Buildings
    P.O. Box 2221
    Suva
    Fiji
"""

DESCRSHORT  = """Fiji Fertility Survey"""

DESCRLONG   = """World Fertily Surveys: Fiji Fertility Survey.
Data represents grouped individual data."""

#suggested notes
NOTE        = """
Number of observations - 70
Number of variables - 7
Variable name definitions::

    totchild - total number of children ever born in the group
    dur      - marriage duration (1=0-4, 2=5-9, 3=10-14, 4=15-19, 5=20-24,
               6=25-29)
    res      - residence (1=Suva, 2=Urban, 3=Rural)
    edu      - education (1=none, 2=lower primary, 3=upper primary,
               4=secondary+)
    nwomen   - number of women in the group
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath
from scikits.statsmodels.tools import categorical

def load():
    """
    Load the Fiji WFS data and return a Dataset class instance.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.
    """
    filepath = dirname(abspath(__file__))
##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    data = recfromtxt(open(filepath + '/wfs.csv', 'rb'), delimiter=",",
            names=True, dtype=float, usecols=(1,2,3,4,6))
    names = ["totchild"] +  list(data.dtype.names)
##### SET THE INDEX #####
    endog = array(data[names[4]]*data[names[5]], dtype=float)
    endog_name = names[0]
##### SET THE INDEX #####
    exog = column_stack(data[i] for i in names[1:4]+[names[5]]).astype(float)
    exog_name = names[1:4] + [names[5]]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
