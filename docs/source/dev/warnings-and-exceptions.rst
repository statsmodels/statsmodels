=======================
Exceptions and Warnings
=======================

Exceptions
----------

Errors derive from Exception or another custom error. Custom errors are
only needed if standard errors, for example ValueError or TypeError, are not
accurate descriptions of the cause for the error.

.. module:: statsmodels.tools.sm_exceptions
   :synopsis: Exceptions and Warnings

.. currentmodule:: statsmodels.tools.sm_exceptions

.. autosummary::
   :toctree: generated/

   ParseError
   PerfectSeparationError
   X13NotFoundError
   X13Error

Warnings
--------

Warnings derive from either an existing warning or another custom
warning, and are often accompanied by a string using the format
``warning_name_doc`` that services as a generic message to use when the
warning is raised.


.. currentmodule:: statsmodels.tools.sm_exceptions

.. autosummary::
   :toctree: generated/

   X13Warning
   IOWarning
   ModuleUnavailableWarning
   ModelWarning
   ConvergenceWarning
   CacheWriteWarning
   IterationLimitWarning
   InvalidTestWarning
   NotImplementedWarning
   OutputWarning
   DomainWarning
   ValueWarning
   EstimationWarning
   SingularMatrixWarning
   HypothesisTestWarning
   InterpolationWarning
   PrecisionWarning
   SpecificationWarning
   HessianInversionWarning
   CollinearityWarning
