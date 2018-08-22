.. currentmodule:: statsmodels.stats.anova

.. _anova:

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

*  `ANOVA <examples/notebooks/generated/interactions_anova.html>`__

Module Reference
----------------

.. module:: statsmodels.stats.anova
   :synopsis: Analysis of Variance

.. autosummary::
   :toctree: generated/

   anova_lm
   AnovaRM
