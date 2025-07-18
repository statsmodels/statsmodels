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

*  `ANOVA <examples/notebooks/generated/interactions_anova.ipynb>`_

Module Reference
----------------

.. module:: statsmodels.stats.anova
   :synopsis: Analysis of Variance

.. autosummary::
   :toctree: generated/

   anova_lm
   AnovaRM

Post-Hoc Testing
----------------

The :class:`AnovaResults` object returned by `AnovaRM.fit` provides methods for post-hoc testing, including Tukey's HSD and general pairwise t-tests with multiple comparison correction.

.. code-block:: python

    # perform repeated measures anova
    res = AnovaRM(data, 'DV', 'id', within=['A', 'B']).fit()

    # perform Tukey's HSD post-hoc test
    tukey_res = res.pairwise_tukeyhsd()
    print(tukey_res)

    # perform pairwise t-tests
    from scipy import stats
    ttest_res = res.allpairtest(stats.ttest_ind, method="bonf")
    print(ttest_res[0])
