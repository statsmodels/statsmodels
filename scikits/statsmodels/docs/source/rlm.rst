.. currentmodule:: scikits.statsmodels.robust


.. _rlm:

Robust Linear Models
====================

Introduction
------------


.. automodule:: scikits.statsmodels.robust.robust_linear_model


Examples
--------

::

    import scikits.statsmodels.api as sm
    data = sm.datasets.stackloss.load()
    data.exog = sm.add_constant(data.exog)
    rlm_model = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    print rlm_results.params

see also the `examples` and the `tests` folders


Module Reference
----------------

Model and Result Classes
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   RLM
   RLMResults

.. _norms:

Norms
^^^^^

.. currentmodule:: scikits.statsmodels.robust.norms

.. autosummary::
   :toctree: generated/

   AndrewWave
   Hampel
   HuberT
   LeastSquares
   RamsayE
   RobustNorm
   TrimmedMean
   TukeyBiweight
   estimate_location


.. currentmodule:: scikits.statsmodels.robust.scale

Scale
^^^^^

.. autosummary::
   :toctree: generated/

    Huber
    HuberScale
    mad
    huber
    hubers_scale
    stand_mad


Technical Documentation
-----------------------

.. toctree::
   :maxdepth: 1

   rlm_techn1
