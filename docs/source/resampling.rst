.. currentmodule:: statsmodels.resampling


.. _resampling:


Resampling :mod:`resampling`
============================


Introduction
------------

Resampling methods (such as the bootstrap, the jackknife, and permutation
tests) are useful for estimating the precision of sample statistics by using subsets of available data or drawing randomly with replacement from such data.

Currently, :mod:`resampling` provides only one method (Efron's original Bootstrap) to form confidence intervals for a given statistic on sample data. This is available as :func:bootstrap_confidence_interval.


References
^^^^^^^^^^

The original reference for the bootstrap is::

    Efron, B. "Bootstrap methods: another look at the jackknife". The Annals of Statistics 7 (1): 1–26, 1979



Examples
--------

This code estimates 95% confidence intervals for the median income of the
population, assuming the sample data is independently distributed:

.. code-block:: python

  import numpy as np
  import statsmodels.api as sm

  data = sm.datasets.engel.load_pandas().data
  income = data['income'].values

  (lower, upper) = bootstrap_confidence_interval(income,
                                                 np.median,
                                                 1000,
                                                 alpha=0.05)
  assert lower < np.median(income) < upper


