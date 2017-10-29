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

Similar to the difference in proportions, we can use the :code:`tt_ind_solve_power` function to analyze the power of an experiment where we'll use a t-test (e.g., if we want to compare the difference in means between two groups in our study).

Imagine we're working on a study to compare the spending habits between two groups of students where the outcome of interest is the average amount spent on books.

In this example, let's assume that we have are conducting a study where we know what our sample-size will be in advance, and we want to calculate the smallest effect-size we'll be able to detect with 80% power.

.. ipython:: python
    
    import statsmodels.api as sm
    import statsmodels.stats.power as smp
    required_alpha = .05
    required_power = .80
    total_sample = 200 # 200 students in our study
    nobs1 = total_sample / 2 # We're going to assign half to each experiment arm
    # Solve for effect-size
    smp.tt_ind_solve_power(nobs1=nobs1, alpha=required_alpha, power=required_power, ratio=1) 

This indicates that within our study we'll only be able to detect effect sizes greater than or equal to 0.398.

Here the "effect-size" is the unit-less standardized effect size (i.e., the difference between the two means divided by the standard deviation).

One Sample Or Paired Sample T-Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tt_solve_power 

Two Sample Z-Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
zt_ind_solve_power

Technical Documentation
-----------------------
