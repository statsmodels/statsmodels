.. currentmodule:: statsmodels.stats.anova

.. _anova_ref:

ANOVA
=====

Analysis of Variance models containing anova_lm for ANOVA analysis with a
linear OLSModel, and AnovaRM for repeated measures ANOVA, within ANOVA for
balanced data.

Examples
--------

.. ipython:: python

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    moore = sm.datasets.get_rdataset("Moore", "carData",
                                     cache=True) # load data
    data = moore.data
    data = data.rename(columns={"partner.status":
                                "partner_status"}) # make name pythonic
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                    data=data).fit()

    table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame
    print(table)

A more detailed example for `anova_lm` can be found here:

*  `ANOVA <examples/notebooks/generated/interactions_anova.html>`_


Repeated-measures designs
------------------------

:class:`AnovaRM` provides repeated-measures ANOVA for fully balanced
within-subject designs. Between-subject effects and corrections for
violations of sphericity are currently not implemented. In addition,
the implementation requires a balanced design, which can limit its
use with longitudinal data containing missing observations.

Possible alternatives include:

* **Linear mixed-effects models**
  (:class:`~statsmodels.regression.mixed_linear_model.MixedLM`):
  These models account for within-subject dependence through
  subject-specific random effects and can accommodate unbalanced
  longitudinal data under appropriate modeling assumptions.

* **Generalized estimating equations**
  (:class:`~statsmodels.genmod.generalized_estimating_equations.GEE`):
  GEE models account for within-subject correlation and estimate
  population-averaged effects. They support a range of response
  distributions and working correlation structures.

* **OLS with subject fixed effects**:
  Including subject indicators in an OLS model accounts for
  time-invariant differences between subjects. Cluster-robust
  standard errors can account for within-subject dependence.
  However, reliable cluster-robust inference generally requires
  a sufficiently large number of independent clusters.

These approaches differ in their assumptions and interpretation.
In particular, population-averaged GEE estimates and
subject-specific mixed-effects estimates can represent different
estimands, especially for nonlinear models. The appropriate
approach depends on the research question, study design, and
assumptions about the data.

Module Reference
----------------

.. module:: statsmodels.stats.anova
   :synopsis: Analysis of Variance

.. autosummary::
   :toctree: generated/

   anova_lm
   AnovaRM
