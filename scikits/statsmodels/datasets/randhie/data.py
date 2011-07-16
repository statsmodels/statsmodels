"""RAND Health Insurance Experiment Data"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is in the public domain."""
TITLE       = __doc__
SOURCE      = """
The data was collected by the RAND corporation as part of the Health
Insurance Experiment (HIE).

http://www.rand.org/health/projects/hie/

This data was used in::

    Cameron, A.C. amd Trivedi, P.K. 2005.  `Microeconometrics: Methods
        and Applications,` Cambridge: New York.

And was obtained from: <http://cameron.econ.ucdavis.edu/mmabook/mmadata.html>

See randhie/src for the original data and description.  The data included
here contains only a subset of the original data.  The data varies slightly
compared to that reported in Cameron and Trivedi.
"""

DESCRSHORT  = """The RAND Co. Health Insurance Experiment Data"""

DESCRLONG   = """"""

NOTE        = """
Number of observations - 20,190
Number of variables - 10
Variable name definitions::

    mdvis   - Number of outpatient visits to an MD
    lncoins - ln(coinsurance + 1), 0 <= coninsurance <= 100
    idp     - 1 if individual deductible plan, 0 otherwise
    lpi     - ln(max(1, annual participation incentive payment))
    fmde    - 0 if idp = 1; ln(max(1, MDE/(0.01 coinsurance))) otherwise
    physlm  - 1 if the person has a physical limitation
    disea   - number of chronic diseases
    hlthg   - 1 if self-rated health is good
    hlthf   - 1 if self-rated health is fair
    hlthp   - 1 if self-rated health is poor
    (Omitted category is excellent self-rated health)
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Loads the RAND HIE data and returns a Dataset class.

    ----------
    endog - structured array of response variable, mdvis
    exog - strucutured array of design

    Returns
    Load instance:
        a class of the data with array attrbutes 'endog' and 'exog'
    """
    filepath = dirname(abspath(__file__))
##### EDIT THE FOLLOWING TO POINT TO DatasetName.csv #####
    data = recfromtxt(open(filepath + '/randhie.csv',"rb"), delimiter=",",
            names=True, dtype=float)
    names = list(data.dtype.names)
    endog = array(data[names[0]]).astype(float)
    endog_name = names[0]
    exog = data[list(names[1:])]
    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
            endog_name = endog_name, exog_name=exog_name)
    return dataset
