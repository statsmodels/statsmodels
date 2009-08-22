#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from version import __version__
from info import __doc__

from regression import *
from glm import *
from rlm import *
import model
import datasets # why did I have to add this but robust and family are imported?
# tried add_subpackage in both setup.py

__all__ = filter(lambda s:not s.startswith('_'),dir())
from numpy.testing import Tester
#test = Tester().test(extra_argv=["--exe"])
test = Tester().test
#TODO:
#def test():
#'''
#on posix systems, this need to be the test call
#'''
#    Tester().test(extra_argv=["--exe"])
