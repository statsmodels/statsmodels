.. currentmodule:: scikits.statsmodels.glm


.. _glm:


Generalized Linear Models
=========================

Introduction
------------

.. automodule:: scikits.statsmodels.glm


Examples
--------
    >>> import scikits.statsmodels as sm
    >>> data = sm.datasets.scotland.Load()
    >>> data.exog = sm.add_constant(data.exog)

    Instantiate a gamma family model with the default link function.

    >>> gamma_model = sm.GLM(data.endog, data.exog,
            family=sm.family.Gamma())
    >>> gamma_results = gamma_model.fit()

see also the `examples` and the `tests` folders


Module Reference
----------------

Model Class
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLM

Results Class
^^^^^^^^^^^^^

.. toctree::

   glmresults

Families
^^^^^^^^



The distribution families currently implemented are

.. currentmodule:: scikits.statsmodels.family.family

.. autosummary::
   :toctree: generated/

   Family
   Binomial
   Gamma
   Gaussian
   InverseGaussian
   NegativeBinomial
   Poisson


Link Functions
^^^^^^^^^^^^^^

The link functions currently implemented are the following. Not all link
functions are available for each distribution family. The list of
available link functions can be obtained by ::

>>> ssm.family.family.<familyname>.available ?

.. currentmodule:: scikits.statsmodels.family.links

.. autosummary::
   :toctree: generated/

   Link

   CDFLink
   CLogLog
   Log
   Logit
   NegativeBinomial
   Power
   cauchy
   cloglog
   identity
   inverse
   inverse_squared
   log
   logit
   nbinom
   probit

Technical Documentation
-----------------------

.. toctree::
   :maxdepth: 1

   glm_techn1
   glm_techn2
