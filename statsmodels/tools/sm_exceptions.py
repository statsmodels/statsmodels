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



class ConvergenceWarning(UserWarning):
    pass


convergence_doc = """
Failed to converge on a solution.
"""


class CacheWriteWarning(UserWarning):
    pass


class IterationLimitWarning(UserWarning):
    pass


iteration_limit_doc = """
Maximum iteration reached.
"""


class InvalidTestWarning(UserWarning):
    pass


class NotImplementedWarning(UserWarning):
    pass


class OutputWarning(UserWarning):
    pass


class DomainWarning(UserWarning):
    pass


class ValueWarning(UserWarning):
    pass


class EstimationWarning(UserWarning):
    pass


class SingularMatrixWarning(UserWarning):
    pass


class HypothesisTestWarning(UserWarning):
    pass


class InterpolationWarning(UserWarning):
    pass


class PrecisionWarning(UserWarning):
    pass


class SpecificationWarning(UserWarning):
    pass


class HessianInversionWarning(UserWarning):
    pass


class ColinearityWarning(UserWarning):
    pass
