.. currentmodule:: statsmodels.genmod.generalized_linear_model

.. _glm:

Generalized Linear Models
=========================

Generalized linear models currently supports estimation using the one-parameter
exponential families.

See `Module Reference`_ for commands and arguments.

Examples
--------

.. ipython:: python
   :okwarning:

   # Load modules and data
   import statsmodels.api as sm
   data = sm.datasets.scotland.load(as_pandas=False)
   data.exog = sm.add_constant(data.exog)

   # Instantiate a gamma family model with the default link function.
   gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
   gamma_results = gamma_model.fit()
   print(gamma_results.summary())

Detailed examples can be found here:

* `GLM <examples/notebooks/generated/glm.html>`__
* `Formula <examples/notebooks/generated/glm_formula.html>`__

Technical Documentation
-----------------------

..   ..glm_techn1
..   ..glm_techn2

The statistical model for each observation :math:`i` is assumed to be

 :math:`Y_i \sim F_{EDM}(\cdot|\theta,\phi,w_i)` and
 :math:`\mu_i = E[Y_i|x_i] = g^{-1}(x_i^\prime\beta)`.

where :math:`g` is the link function and :math:`F_{EDM}(\cdot|\theta,\phi,w)`
is a distribution of the family of exponential dispersion models (EDM) with
natural parameter :math:`\theta`, scale parameter :math:`\phi` and weight
:math:`w`.
Its density is given by

 :math:`f_{EDM}(y|\theta,\phi,w) = c(y,\phi,w)
 \exp\left(\frac{y\theta-b(\theta)}{\phi}w\right)\,.`

It follows that :math:`\mu = b'(\theta)` and
:math:`Var[Y|x]=\frac{\phi}{w}b''(\theta)`. The inverse of the first equation
gives the natural parameter as a function of the expected value
:math:`\theta(\mu)` such that

 :math:`Var[Y_i|x_i] = \frac{\phi}{w_i} v(\mu_i)`

with :math:`v(\mu) = b''(\theta(\mu))`. Therefore it is said that a GLM is
determined by link function :math:`g` and variance function :math:`v(\mu)`
alone (and :math:`x` of course).

Note that while :math:`\phi` is the same for every observation :math:`y_i`
and therefore does not influence the estimation of :math:`\beta`,
the weights :math:`w_i` might be different for every :math:`y_i` such that the
estimation of :math:`\beta` depends on them.

================================================= ============================== ============================== ======================================== =========================================== ============================================================================ =====================
Distribution                                      Domain                         :math:`\mu=E[Y|x]`             :math:`v(\mu)`                           :math:`\theta(\mu)`                         :math:`b(\theta)`                                                            :math:`\phi`
================================================= ============================== ============================== ======================================== =========================================== ============================================================================ =====================
Binomial :math:`B(n,p)`                           :math:`0,1,\ldots,n`           :math:`np`                     :math:`\mu-\frac{\mu^2}{n}`              :math:`\log\frac{p}{1-p}`                   :math:`n\log(1+e^\theta)`                                                    1
Poisson :math:`P(\mu)`                            :math:`0,1,\ldots,\infty`      :math:`\mu`                    :math:`\mu`                              :math:`\log(\mu)`                           :math:`e^\theta`                                                             1
Neg. Binom. :math:`NB(\mu,\alpha)`                :math:`0,1,\ldots,\infty`      :math:`\mu`                    :math:`\mu+\alpha\mu^2`                  :math:`\log(\frac{\alpha\mu}{1+\alpha\mu})` :math:`-\frac{1}{\alpha}\log(1-\alpha e^\theta)`                             1
Gaussian/Normal :math:`N(\mu,\sigma^2)`           :math:`(-\infty,\infty)`       :math:`\mu`                    :math:`1`                                :math:`\mu`                                 :math:`\frac{1}{2}\theta^2`                                                  :math:`\sigma^2`
Gamma :math:`N(\mu,\nu)`                          :math:`(0,\infty)`             :math:`\mu`                    :math:`\mu^2`                            :math:`-\frac{1}{\mu}`                      :math:`-\log(-\theta)`                                                       :math:`\frac{1}{\nu}`
Inv. Gauss. :math:`IG(\mu,\sigma^2)`              :math:`(0,\infty)`             :math:`\mu`                    :math:`\mu^3`                            :math:`-\frac{1}{2\mu^2}`                   :math:`-\sqrt{-2\theta}`                                                     :math:`\sigma^2`
Tweedie :math:`p\geq 1`                           depends on :math:`p`           :math:`\mu`                    :math:`\mu^p`                            :math:`\frac{\mu^{1-p}}{1-p}`               :math:`\frac{\alpha-1}{\alpha}\left(\frac{\theta}{\alpha-1}\right)^{\alpha}` :math:`\phi`
================================================= ============================== ============================== ======================================== =========================================== ============================================================================ =====================

The Tweedie distribution has special cases for :math:`p=0,1,2` not listed in the
table and uses :math:`\alpha=\frac{p-2}{p-1}`.

Correspondence of mathematical variables to code:

* :math:`Y` and :math:`y` are coded as ``endog``, the variable one wants to
  model
* :math:`x` is coded as ``exog``, the covariates alias explanatory variables
* :math:`\beta` is coded as ``params``, the parameters one wants to estimate
* :math:`\mu` is coded as ``mu``, the expectation (conditional on :math:`x`)
  of :math:`Y`
* :math:`g` is coded as ``link`` argument to the ``class Family``
* :math:`\phi` is coded as ``scale``, the dispersion parameter of the EDM
* :math:`w` is not yet supported (i.e. :math:`w=1`), in the future it might be
  ``var_weights``
* :math:`p` is coded as ``var_power`` for the power of the variance function
  :math:`v(\mu)` of the Tweedie distribution, see table
* :math:`\alpha` is either

  * Negative Binomial: the ancillary parameter ``alpha``, see table
  * Tweedie: an abbreviation for :math:`\frac{p-2}{p-1}` of the power :math:`p`
    of the variance function, see table


References
^^^^^^^^^^

* Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach. SAGE QASS Series.
* Green, PJ. 1984. “Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives.” Journal of the Royal Statistical Society, Series B, 46, 149-192.
* Hardin, J.W. and Hilbe, J.M. 2007. “Generalized Linear Models and Extensions.” 2nd ed. Stata Press, College Station, TX.
* McCullagh, P. and Nelder, J.A. 1989. “Generalized Linear Models.” 2nd ed. Chapman & Hall, Boca Rotan.

Module Reference
----------------

.. module:: statsmodels.genmod.generalized_linear_model
   :synopsis: Generalized Linear Models (GLM)

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
   PredictionResults

.. _families:

Families
^^^^^^^^

The distribution families currently implemented are

.. module:: statsmodels.genmod.families.family
.. currentmodule:: statsmodels.genmod.families.family

.. autosummary::
   :toctree: generated/

   Family
   Binomial
   Gamma
   Gaussian
   InverseGaussian
   NegativeBinomial
   Poisson
   Tweedie


.. _links:

Link Functions
^^^^^^^^^^^^^^

The link functions currently implemented are the following. Not all link
functions are available for each distribution family. The list of
available link functions can be obtained by

::

    >>> sm.families.family.<familyname>.links

.. module:: statsmodels.genmod.families.links
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

.. _varfuncs:

Variance Functions
^^^^^^^^^^^^^^^^^^

Each of the families has an associated variance function. You can access
the variance functions here:

::

    >>> sm.families.<familyname>.variance

.. module:: statsmodels.genmod.families.varfuncs
.. currentmodule:: statsmodels.genmod.families.varfuncs

.. autosummary::
   :toctree: generated/

   VarianceFunction
   constant
   Power
   mu
   mu_squared
   mu_cubed
   Binomial
   binary
   NegativeBinomial
   nbinom
