.. currentmodule:: statsmodels.genmod.generalized_estimating_equations

.. _gee:

Generalized Estimating Equations
================================

Generalized Estimating Equations estimate generalized linear models for
panel, cluster or repeated measures data when the observations are possibly
correlated withing a cluster but uncorrelated across clusters. It supports
estimation of the same one-parameter exponential families as Generalized
Linear models (`GLM`).

See `Module Reference`_ for commands and arguments.

Examples
--------

The following illustrates a Poisson regression with exchangeable correlation
within clusters using data on epilepsy seizures.

.. ipython:: python

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    data = sm.datasets.get_rdataset('epil', package='MASS').data

    fam = sm.families.Poisson()
    ind = sm.cov_struct.Exchangeable()
    mod = smf.gee("y ~ age + trt + base", "subject", data,
                  cov_struct=ind, family=fam)
    res = mod.fit()
    print(res.summary())


Several notebook examples of the use of GEE can be found on the Wiki:
`Wiki notebooks for GEE <https://github.com/statsmodels/statsmodels/wiki/Examples#generalized-estimating-equations-gee>`_


References
^^^^^^^^^^

* KY Liang and S Zeger. "Longitudinal data analysis using generalized
  linear models". Biometrika (1986) 73 (1): 13-22.
* S Zeger and KY Liang. "Longitudinal Data Analysis for Discrete and
  Continuous Outcomes". Biometrics Vol. 42, No. 1 (Mar., 1986),
  pp. 121-130
* A Rotnitzky and NP Jewell (1990). "Hypothesis testing of regression
  parameters in semiparametric generalized linear models for cluster
  correlated data", Biometrika, 77, 485-497.
* Xu Guo and Wei Pan (2002). "Small sample performance of the score test in
  GEE".
  http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
* LA Mancl LA, TA DeRouen (2001). A covariance estimator for GEE with improved
  small-sample properties.  Biometrics. 2001 Mar;57(1):126-34.


Module Reference
----------------

.. module:: statsmodels.genmod.generalized_estimating_equations
   :synopsis: Generalized estimating equations

Model Class
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GEE
   NominalGEE
   OrdinalGEE

.. module:: statsmodels.genmod.qif
   :synopsis: Quadratic inference functions

.. currentmodule:: statsmodels.genmod.qif

.. autosummary::
   :toctree: generated/

   QIF

Results Classes
^^^^^^^^^^^^^^^

.. currentmodule:: statsmodels.genmod.generalized_estimating_equations

.. autosummary::
   :toctree: generated/

   GEEResults
   GEEMargins

.. currentmodule:: statsmodels.genmod.qif

.. autosummary::
   :toctree: generated/

   QIFResults

Dependence Structures
^^^^^^^^^^^^^^^^^^^^^

The dependence structures currently implemented are

.. module:: statsmodels.genmod.cov_struct
   :synopsis: Covariance structures for Generalized Estimating Equations (GEE)

.. currentmodule:: statsmodels.genmod.cov_struct

.. autosummary::
   :toctree: generated/

   CovStruct
   Autoregressive
   Exchangeable
   GlobalOddsRatio
   Independence
   Nested


Families
^^^^^^^^

The distribution families are the same as for GLM, currently implemented are

.. module:: statsmodels.genmod.families.family
   :synopsis: Generalized Linear Model (GLM) families

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


Link Functions
^^^^^^^^^^^^^^

The link functions are the same as for GLM, currently implemented are the
following. Not all link functions are available for each distribution family.
The list of available link functions can be obtained by

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
