.. _sandbox:


Sandbox
=======

This sandbox contains code that is for various reasons not ready to be
included in statsmodels proper. It contains modules from the old stats.models
code that have not been tested, verified and updated to the new statsmodels
structure: cox survival model, mixed effects model with repeated measures,
generalized additive model and the formula framework. The sandbox also
contains code that is currently being worked on until it fits the pattern
of statsmodels or is sufficiently tested.

All sandbox modules have to be explicitly imported to indicate that they are
not yet part of the core of statsmodels. The quality and testing of the
sandbox code varies widely.


Module Reference
----------------

Time Series analysis :mod:`tsa`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this part we develop models and functions that will be useful for time
series analysis. Most of the models and function have been moved to
:mod:`statsmodels.tsa`.

Moving Window Statistics
""""""""""""""""""""""""

Most moving window statistics, like rolling mean, moments (up to 4th order), min,
max, mean, and variance, are covered by the functions for `Moving (rolling)
statistics/moments <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#window-functions>`_ in Pandas.

.. module:: statsmodels.sandbox.tsa
   :synopsis: Experimental time-series analysis models

.. currentmodule:: statsmodels.sandbox.tsa

.. autosummary::
   :toctree: generated/

   movstat.movorder
   movstat.movmean
   movstat.movvar
   movstat.movmoment
