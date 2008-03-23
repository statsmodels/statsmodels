#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from neuroimaging.fixes.scipy.stats_models.info import __doc__

import neuroimaging.fixes.scipy.stats_models.model
import neuroimaging.fixes.scipy.stats_models.formula
import neuroimaging.fixes.scipy.stats_models.regression
import neuroimaging.fixes.scipy.stats_models.robust
import neuroimaging.fixes.scipy.stats_models.family
from neuroimaging.fixes.scipy.stats_models.glm import Model as glm
from neuroimaging.fixes.scipy.stats_models.rlm import Model as rlm

__all__ = filter(lambda s:not s.startswith('_'),dir())

from neuroimaging.externals.scipy.testing.pkgtester import Tester
test = Tester().test
