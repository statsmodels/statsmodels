"""
Test Results for Semi-Parametric regression. Validated against R where possible.

References
----------
[1] R 'np' vignette : http://cran.r-project.org/web/packages/np/vignettes/np.pdf

"""
import os
import numpy as np

cur_dir = os.path.abspath(os.path.dirname(__file__))

class Wage1(object):

    """ Results in this class are based on the example given in the example of
        section 8 in [1]. Currently results are not validated against R output
        as not all analogous options are avaliable. """

    def __init__(self):
        self.nobs = 526


class BirthWt(object):

    """ Results in this class are based on the example given in the example of
        section 9 in [1]. Currently results are not validated against R output
        as not all analogous options are avaliable. """
        
    def __init__(self):
        self.nobs = 189


