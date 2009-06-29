#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from models.info import __doc__

import nipy.fixes.scipy.stats.models.model
import nipy.fixes.scipy.stats.models.formula
import nipy.fixes.scipy.stats.models.regression
import nipy.fixes.scipy.stats.models.robust
import nipy.fixes.scipy.stats.models.family
import nipy.fixes.scipy.stats.models.glm
import nipy.fixes.scipy.stats.models.rlm
from nipy.fixes.scipy.stats.models.glm import Model as GLM
from nipy.fixes.scipy.stats.models.rlm import Model as RLM

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import Tester
test = Tester().test
