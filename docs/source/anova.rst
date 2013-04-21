.. currentmodule:: statsmodels.stats.anova

.. _anova:

ANOVA
=====

Analysis of Variance models 

Examples
--------

.. ipython:: python

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    moore = sm.datasets.get_rdataset("Moore", "car", 
                                     cache=True) # load data
    data = moore.data
    data = data.rename(columns={"partner.status" : 
                                "partner_status"}) # make name pythonic
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                    data=data).fit()

    table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame
    print table
    
A more detailed example can be found here:

.. toctree::
  :maxdepth: 1

  examples/generated/example_interactions

Module Reference
----------------

.. autosummary::
   :toctree: generated/

   anova_lm
