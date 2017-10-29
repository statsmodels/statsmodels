.. module:: statsmodels.stats.power
   :synopsis: Power and Sample Size Calculations

.. currentmodule:: statsmodels.stats.power

Power :mod:`power`
=======================

The :mod:`power` module currently implements power and sample size calculations for t-tests, normality tests, F-tests and Chisquare goodness of fit tests.

Power analysis is commonly used for prospective experimental design under the null hypothesis significance testing framework. As researchers, we need to know whether or not we'll be able to detect an effect size given the likely effect sizes and our required confidence levels.

The key parameters for designing our experiment are:

* :code:`effect_size`: The magnitude of the difference between groups in our experiment that we want to be able to statistically detect.
* :code:`nobs`: The sample size for our experiment or study.
* :code:`alpha`: The "significance level" (commonly set to 0.05) = P(type I error) = the likelihood of detecting an effect that is not "really" there.
* :code:`power`: Commonly set to 0.80, the "power" is the likelihood of detecting an effect that really is there = 1 - P(Type II error).

If we know three of these parameters, we can calculate the fourth. 

See also: 

* `Power Analysis in R <https://www.statmethods.net/stats/power.html>`_
* `How not to run an A/B Test <https://www.evanmiller.org/how-not-to-run-an-ab-test.html>`_

Examples
--------

Normal Test for Difference in Proportions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Commonly, especially when designing "A/B" experiments, we know the effect-size we want to detect as well as the required signficance level and power for our experiment, and we want to know what our required sample size is. 

Let's imagine we are running an online A/B test comparing the conversion rate of a button when it is colored green vs. when the button is colored blue. We will use our industry standard settings for confidence level and power

.. ipython:: python

    required_alpha = .05
    required_power = .80

And we know, from previous tests, that we want to be able to detect a relative 5% difference in conversion rates off of a baseline of 10%. 

.. ipython:: python
    
    import statsmodels.api as sm
    baseline = 0.10
    relative_change = 0.05
    es = sm.stats.proportion_effectsize(baseline, baseline*(1+relative_change))

We can calculate our required sample size using the prop_ind_solve_power function

.. ipython:: python
    
    import statsmodels.stats.power as smp
    smp.prop_ind_solve_power(effect_size=es, power=required_power, alpha=required_alpha, ratio=1)
    # TODO: Ensure this returns sample size for one arm


Note that this family of "solver" functions identifies which three of the four power parameters you provided, and finds the fourth. Since we didn't provide an "nobs1" argument, that's what gets returned to us.

Two Independent Sample T-Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the difference in proportions, we can use the :code:`tt_ind_solve_power` function to calculate
tt_ind_solve_power 


One Sample Or Paired Sample T-Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tt_solve_power 

Two Sample Z-Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
zt_ind_solve_power

Technical Documentation
-----------------------
