.. currentmodule:: statsmodels.robust


.. _rlm:

Robust Linear Models
====================

Robust linear models with support for the M-estimators listed under `Norms`_.

See `Module Reference`_ for commands and arguments.

Examples
--------

.. ipython:: python

    # Load modules and data
    import statsmodels.api as sm
    data = sm.datasets.stackloss.load(as_pandas=False)
    data.exog = sm.add_constant(data.exog)

    # Fit model and print summary
    rlm_model = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    print(rlm_results.params)

Detailed examples can be found here:

* `Robust Models 1 <examples/notebooks/generated/robust_models_0.html>`__
* `Robust Models 2 <examples/notebooks/generated/robust_models_1.html>`__

Technical Documentation
-----------------------

.. toctree::
   :maxdepth: 1

   rlm_techn1

References
^^^^^^^^^^

* PJ Huber. ‘Robust Statistics’ John Wiley and Sons, Inc., New York. 1981.
* PJ Huber. 1973, ‘The 1972 Wald Memorial Lectures: Robust Regression: Asymptotics, Conjectures, and Monte Carlo.’ The Annals of Statistics, 1.5, 799-821.
* R Venables, B Ripley. ‘Modern Applied Statistics in S’ Springer, New York,

Module Reference
----------------

.. module:: statsmodels.robust

Model Classes
^^^^^^^^^^^^^

.. module:: statsmodels.robust.robust_linear_model
.. currentmodule:: statsmodels.robust.robust_linear_model

.. autosummary::
   :toctree: generated/

   RLM

Model Results
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   RLMResults

.. _norms:

Norms
^^^^^

.. module:: statsmodels.robust.norms
.. currentmodule:: statsmodels.robust.norms

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


Scale
^^^^^

.. module:: statsmodels.robust.scale
.. currentmodule:: statsmodels.robust.scale

.. autosummary::
   :toctree: generated/

    Huber
    HuberScale
    mad
    hubers_scale
