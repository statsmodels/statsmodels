"""
Tools for nonparametric statistics, mainly density estimation and regression.

For an overview of this module, see docs/source/nonparametric.rst
"""
from statsmodels import PytestTester
test = PytestTester()

from . import bandwidths
from . import kde
from . import kde_methods
from . import kde_utils
from . import kernels
