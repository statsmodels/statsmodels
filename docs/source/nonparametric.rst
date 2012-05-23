.. currentmodule:: statsmodels.nonparametric

.. _nonparametric:


Nonparametric Methods :mod:`nonparametric`
==========================================

Introduction
------------

This section collects various methods in nonparametric statistics. This
includes currently kernel density estimation for univariate data, and
lowess.

sandbox.nonparametric contains additional functions that are work in progress
or don't have unit tests yet. We are planning to include here nonparametric
density estimators, especially based on kernel or orthogonal polynomials,
smoothers, and tools for nonparametric models and methods in other parts of
statsmodels.


Module Reference
----------------

Currently, the public functions and classes are

.. autosummary::
   :toctree: generated/

   smoothers_lowess.lowess
   kde.KDE

helper functions for kernel bandwidths

.. autosummary::
   :toctree: generated/

   bandwidths.bw_scott
   bandwidths.bw_silverman
   bandwidths.select_bandwidth

