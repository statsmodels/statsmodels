
.. module:: statsmodels.treatment
   :synopsis: Treatment Effect

.. currentmodule:: statsmodels.treatment



.. _treatment:


Treatment Effects :mod:`treatment`
==================================

:mod:`statsmodels.treatment` contains a model and a results class for
the estimation of treatment effects under conditional independence.

Methods for for estimating treatment effects are available in as methods
in the :class:`~statsmodels.treatment.treatment_effects.TreatmentEffect`. Standard Errors are computed using GMM from
the moment conditions of the treatment model, outcome model and effects
statistics, average treatment effect ATE, potential outcome means POM, and
for some methods optionally average treatment effect on the treated ATT.

See also overview notebook in
`Treatment Effect <examples/notebooks/generated/treatment_effect.ipynb>`_

.. currentmodule:: statsmodels.treatment


.. autosummary::
   :toctree: generated/

   treatment_effects.TreatmentEffect
   treatment_effects.TreatmentEffectResults

Overlap and covariate balance
-----------------------------

After constructing a ``TreatmentEffect`` with a fitted selection model, inspect
its numerical diagnostics before interpreting weighted estimates::

    teff.overlap_summary()
    teff.balance_table()
    teff.balance_table(effect_group="treated")

The overlap summary describes original, unclipped propensity scores by treatment
group, including counts outside the clipping bounds. The balance table reports
selection-model covariate means and standardized mean differences before and
after weighting, using the estimator's clipped propensity scores. Supply ``exog``
to examine other numeric covariates in the same observation order.

Standardization uses a fixed pooled unweighted sample standard deviation for all
weighting targets and numeric columns, including binary columns. Constant
columns have undefined (NaN) standardized differences. These are descriptive
diagnostics; they neither establish overlap nor rule out unmeasured confounding.
Neither method drops observations or refits the models.

.. autosummary::
   :toctree: generated/

   treatment_effects.TreatmentEffect.overlap_summary
   treatment_effects.TreatmentEffect.balance_table
