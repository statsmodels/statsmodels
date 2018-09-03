.. module:: statsmodels.base.optimizer
.. currentmodule:: statsmodels.base.optimizer

Optimization
============

Many of the models we use such as :ref:`GLM <glm>` and
:ref:`discrete models <discretemod>` allow for the optional selection of a
scipy optimizer. This can either be defaut or an option based on the model
selected.

statsmodels supports the following optimizers:

- ``newton`` - Newton-Raphson iteration. While not directly from scipy, we
  consider it an optimizer because only the objective function and the score
  are required parameters.
- ``nm`` - scipy's ``fmin_nm``
- ``bfgs`` - Broyden–Fletcher–Goldfarb–Shanno optimization, scipy's
  ``fmin_bfgs``.
- ``lbfgs`` - A more memory-efficient (limited memory) implementation of
  ``bfgs``. Scipy's ``fmin_l_bfgs_b``.
- ``cg`` - Conjugate gradient optimization. Scipy's ``fmin_cg``.
- ``ncg`` - Newton conjugate gradient. Scipy's ``fmin_ncg``.
- ``powell`` - Powell's method. Scipy's ``fmin_powell``.
- ``basinhopping`` - Basin hopping. This is part of scipy's ``basinhopping``
  tools.

Model Class
-----------

Generally, there is no need for an end-user to directly call these functions
and classes. However, we provide the class because the different optimization
techniques have unique keyword arguments that may be useful to the user.

.. autosummary::
   :toctree: generated/

   Optimizer
   _fit_newton
   _fit_bfgs
   _fit_lbfgs
   _fit_nm
   _fit_cg
   _fit_ncg
   _fit_powell
   _fit_basinhopping
