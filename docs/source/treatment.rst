
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
