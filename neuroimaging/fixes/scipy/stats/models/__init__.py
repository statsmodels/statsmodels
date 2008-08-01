#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from neuroimaging.fixes.scipy.stats.models.info import __doc__

import neuroimaging.fixes.scipy.stats.models.model
import neuroimaging.fixes.scipy.stats.models.formula
import neuroimaging.fixes.scipy.stats.models.regression
import neuroimaging.fixes.scipy.stats.models.robust
import neuroimaging.fixes.scipy.stats.models.family
from neuroimaging.fixes.scipy.stats.models.glm import Model as glm
from neuroimaging.fixes.scipy.stats.models.rlm import Model as rlm

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import Tester
test = Tester().test
