.. currentmodule:: statsmodels.genmod.generalized_linear_model

.. _glm:

Generalized Linear Models
=========================

Generalized linear models currently supports estimation using the one-parameter
exponential families

See `Module Reference`_ for commands and arguments.

Examples
--------

::

    # Load modules and data
    import statsmodels.api as sm
    data = sm.datasets.scotland.load()
    data.exog = sm.add_constant(data.exog)

    # Instantiate a gamma family model with the default link function.
    gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
    gamma_results = gamma_model.fit()

Detailed examples can be found here:

.. toctree::
   :maxdepth: 1

   examples/notebooks/generated/glm
   examples/notebooks/generated/glm_formula

Technical Documentation
-----------------------

..   ..glm_techn1
..   ..glm_techn2

References
^^^^^^^^^^

* Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach. SAGE QASS Series.
* Green, PJ. 1984. “Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives.” Journal of the Royal Statistical Society, Series B, 46, 149-192.
* Hardin, J.W. and Hilbe, J.M. 2007. “Generalized Linear Models and Extensions.” 2nd ed. Stata Press, College Station, TX.
* McCullagh, P. and Nelder, J.A. 1989. “Generalized Linear Models.” 2nd ed. Chapman & Hall, Boca Rotan.

Module Reference
----------------

Model Class
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLM

Results Class
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLMResults

.. _families:

Families
^^^^^^^^

The distribution families currently implemented are

.. currentmodule:: statsmodels.genmod.families.family

.. autosummary::
   :toctree: generated/
   :template: autosummary/glmfamilies.rst

   Family
   Binomial
   Gamma
   Gaussian
   InverseGaussian
   NegativeBinomial
   Poisson


.. _links:

Link Functions
^^^^^^^^^^^^^^

The link functions currently implemented are the following. Not all link
functions are available for each distribution family. The list of
available link functions can be obtained by

::

    >>> sm.families.family.<familyname>.links

.. currentmodule:: statsmodels.genmod.families.links

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
   inverse_power
   inverse_squared
   log
   logit
   nbinom
   probit


