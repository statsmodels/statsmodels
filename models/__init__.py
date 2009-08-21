#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

#from version import __version__
from models.info import __doc__

from regression import *
from glm import *
from rlm import *

__all__ = filter(lambda s:not s.startswith('_'),dir())
from numpy.testing import Tester
test = Tester().test
