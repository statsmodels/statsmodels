.. currentmodule:: statsmodels.sandbox.regression.gmm


.. _gmm:


Generalized Method of Moments :mod:`gmm`
========================================

:mod:`scikits.statmodels.gmm` contains model classes and functions that are based on
estimation with Generalized Method of Moments.
Currently the general non-linear case is implemented. An example class for the standard
linear instrumental variable model is included. This has been introduced as a test case, it
works correctly but it does not take the linear structure into account. For the linear
case we intend to introduce a specific implementation which will be faster and numerically
more accurate.

Currently, GMM takes arbitrary non-linear moment conditions and calculates the estimates
either for a given weighting matrix or iteratively by alternating between estimating
the optimal weighting matrix and estimating the parameters. Implementing models with
different moment conditions is done by subclassing GMM. In the minimal implementation
only the moment conditions, `momcond` have to be defined.

.. currentmodule:: statsmodels.sandbox.regression.gmm


Module Reference
""""""""""""""""

.. autosummary::
   :toctree: generated/

   GMM
   GMMResults
   IV2SLS

not sure what the status is on the following

.. autosummary::
   :toctree: generated/

   IVGMM
   NonlinearIVGMM
   DistQuantilesGMM
