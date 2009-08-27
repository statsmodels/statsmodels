.. currentmodule:: scikits.statsmodels.glm


.. _glm:

.. toctree::
   :maxdepth: 1

   glm.techn

Generalized Linear Models
=========================

Introduction
------------

Generalized linear models is currently implemented for several families
more text
b
b

Examples
--------



Reference
---------


Model and Result Classes
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLM
   GLMResults

Families
^^^^^^^^

The distribution families currently implemented are

.. currentmodule:: scikits.statsmodels.family.family

.. autosummary::
   :toctree: generated/

   Binomial
   Family
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

   glm_techn
