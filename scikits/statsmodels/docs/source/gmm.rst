.. currentmodule:: scikits.statsmodels.sandbox.regression.gmm


.. _tsa:


Generalized Method of Moments :mod:`gmm`
========================================

:mos:`scikits.statmodels.gmm` contains model classes and functions that are based on
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
