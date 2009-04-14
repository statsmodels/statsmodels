#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from nipy.fixes.scipy.stats.models.info import __doc__

import nipy.fixes.scipy.stats.models.model
import nipy.fixes.scipy.stats.models.formula
import nipy.fixes.scipy.stats.models.regression
import nipy.fixes.scipy.stats.models.robust
import nipy.fixes.scipy.stats.models.family
from nipy.fixes.scipy.stats.models.glm import Model as glm
from nipy.fixes.scipy.stats.models.rlm import Model as rlm

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import Tester
test = Tester().test
