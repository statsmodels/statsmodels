
.. module:: statsmodels.treatment
.. currentmodule:: statsmodels.treatment
   :synopsis: Treatment Effect


.. _othermod:


Treatment Effects :mod:`treatment`
==================================

:mod:`statsmodels.treatment` contains a model and a results class for
the estimation of treatment effects under conditional independence.

Methods for for estimating treatment effects are available in as methods
in the :class:`TreatmentEffect`. Standard Errors are computed using GMM from
the moment conditions of the treatment model, outcome model and effects
statistics, average treatment effect ATE, potential outcome means POM, and
for some methods optionally average treatment effect on the treated ATT.

See also overview notebook in
`Treatment Effect <examples/notebooks/generated/treatment_effect.html>`__

.. module:: statsmodels.treatment.treatment_effects
.. currentmodule:: statsmodels.treatment.treatment_effects


.. autosummary::
   :toctree: generated/

   TreatmentEffect
   TreatmentEffectResults
