#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from models.info import __doc__

import models.model
import models.formula
import models.regression
import models.robust
import models.family
import models.glm
import models.rlm
from models.glm import GLMtwo as GLM
from rlm import Model as RLM

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import Tester
test = Tester().test
