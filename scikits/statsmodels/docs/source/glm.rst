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

Reference
---------

Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach.
    SAGE QASS Series.

Green, PJ. 1984.  "Iteratively reweighted least squares for maximum
    likelihood estimation, and some robust and resistant alternatives."
    Journal of the Royal Statistical Society, Series B, 46, 149-192.

Hardin, J.W. and Hilbe, J.M. 2007.  "Generalized Linear Models and
    Extensions."  2nd ed.  Stata Press, College Station, TX.

McCullagh, P. and Nelder, J.A.  1989.  "Generalized Linear Models." 2nd ed.
    Chapman & Hall, Boca Rotan.


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
