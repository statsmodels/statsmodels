"""
Contains custom errors and warnings.

Errors should derive from Exception or another custom error. Custom errors are
only needed it standard errors, for example ValueError or TypeError, are not
accurate descriptions of the reason for the error.

Warnings should derive from either an existing warning or another custom
warning, and should usually be accompanied by a sting using the format
warning_name_doc that services as a generic message to use when the warning is
raised.
"""

import warnings


# Errors
class PerfectSeparationError(Exception):
    pass


class MissingDataError(Exception):
    pass


class X13NotFoundError(Exception):
    pass


class X13Error(Exception):
    pass


# Warning
class X13Warning(Warning):
    pass


class IOWarning(RuntimeWarning):
    pass


class ModuleUnavailableWarning(Warning):
    pass


module_unavailable_doc = """
The module {0} is not available. Cannot run in parallel.
"""


class ModelWarning(UserWarning):
    pass


class ConvergenceWarning(ModelWarning):
    pass


convergence_doc = """
Failed to converge on a solution.
"""


class CacheWriteWarning(ModelWarning):
    pass


class IterationLimitWarning(ModelWarning):
    pass


iteration_limit_doc = """
Maximum iteration reached.
"""


class InvalidTestWarning(ModelWarning):
    pass


class NotImplementedWarning(ModelWarning):
    pass


class OutputWarning(ModelWarning):
    pass


class DomainWarning(ModelWarning):
    pass


class ValueWarning(ModelWarning):
    pass


class EstimationWarning(ModelWarning):
    pass


class SingularMatrixWarning(ModelWarning):
    pass


class HypothesisTestWarning(ModelWarning):
    pass


class InterpolationWarning(ModelWarning):
    pass


class PrecisionWarning(ModelWarning):
    pass


class SpecificationWarning(ModelWarning):
    pass


class HessianInversionWarning(ModelWarning):
    pass


class CollinearityWarning(ModelWarning):
    pass


warnings.simplefilter('always', category=ModelWarning)
warnings.simplefilter("always", (ConvergenceWarning, CacheWriteWarning,
                                 IterationLimitWarning, InvalidTestWarning))
