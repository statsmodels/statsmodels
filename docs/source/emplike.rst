.. currentmodule:: statsmodels.emplike


.. _emplike:


Empirical Likelihood :mod:`emplike`
====================================


Introduction
------------

Empirical likelihood is a method of nonparametric inference and estimation that lifts the
obligation of having to specify a family of underlying distributions.  Moreover, empirical
likelihood methods do not require re-sampling but still
uniquely determine confidence regions whose shape mirrors the shape of the data.
In essence, empirical likelihood attempts to combine the benefits of parametric
and nonparametric methods while limiting their shortcomings.  The main difficulties  of
empirical likelihood is the computationally intensive methods required to conduct inference.
:mod:`statsmodels.emplike` attempts to provide a user-friendly interface that allows the
end user to effectively conduct empirical likelihood analysis without having to concern
themselves with the computational burdens.

Currently, :mod:`emplike` provides methods to conduct hypothesis tests and form confidence
intervals for descriptive statistics.  Empirical likelihood estimation and inference
in a regression, accelerated failure time and instrumental variable model are
currently under development.

References
^^^^^^^^^^

The main reference for empirical likelihood is::

    Owen, A.B. "Empirical Likelihood." Chapman and Hall, 2001.



Examples
--------

.. ipython:: python

  import numpy as np
  import statsmodels.api as sm

  # Generate Data
  x = np.random.standard_normal(50)

  # initiate EL
  el = sm.emplike.DescStat(x)

  # confidence interval for the mean
  el.ci_mean()

  # test variance is 1
  el.test_var(1)


Module Reference
----------------

.. module:: statsmodels.emplike
   :synopsis: Empirical likelihood tools

.. autosummary::
   :toctree: generated/

   descriptive.DescStat
   descriptive.DescStatUV
   descriptive.DescStatMV
