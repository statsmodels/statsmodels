#collect some imports of verified (at least one example) functions
from scikits.statsmodels.sandbox.stats.multicomp import \
             multipletests, fdrcorrection0, fdrcorrection_twostage, tukeyhsd

from scikits.statsmodels import NoseWrapper as Tester
test = Tester().test
