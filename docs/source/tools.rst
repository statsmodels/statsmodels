.. currentmodule:: statsmodels.tools


.. _tools:

Tools
=====

Our tool collection contains some convenience functions for users and
functions that were written mainly for internal use.

Additional to this tools directory, several other subpackages have their own
tools modules, for example :mod:`statsmodels.tsa.tsatools`


Module Reference
----------------

Basic tools :mod:`tools`
^^^^^^^^^^^^^^^^^^^^^^^^

These are basic and miscellaneous tools. The full import path is
`statsmodels.tools.tools`.

.. autosummary::
   :toctree: generated/

   tools.add_constant

The next group are mostly helper functions that are not separately tested or
insufficiently tested.

.. autosummary::
   :toctree: generated/

   tools.categorical
   tools.ECDF
   tools.clean0
   tools.fullrank
   tools.isestimable
   tools.monotone_fn_inverter
   tools.rank
   tools.recipr
   tools.recipr0
   tools.unsqueeze


Measure for fit performance :mod:`eval_measures`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first group of function in this module are standalone versions of
information criteria, aic bic and hqic. The function with `_sigma` suffix
take the error sum of squares as argument, those without, take the value
of the log-likelihood, `llf`, as argument.

The second group of function are measures of fit or prediction performance,
which are mostly one liners to be used as helper functions. All of those
calculate a performance or distance statistic for the difference between two
arrays. For example in the case of Monte Carlo or cross-validation, the first
array would be the estimation results for the different replications or draws,
while the second array would be the true or observed values.

.. currentmodule:: statsmodels.tools

.. autosummary::
   :toctree: generated/

   eval_measures.aic
   eval_measures.aic_sigma
   eval_measures.aicc
   eval_measures.aicc_sigma
   eval_measures.bic
   eval_measures.bic_sigma
   eval_measures.hqic
   eval_measures.hqic_sigma

   eval_measures.bias
   eval_measures.iqr
   eval_measures.maxabs
   eval_measures.meanabs
   eval_measures.medianabs
   eval_measures.medianbias
   eval_measures.mse
   eval_measures.rmse
   eval_measures.stde
   eval_measures.vare
