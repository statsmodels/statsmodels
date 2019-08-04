.. _sandbox:


Sandbox
=======

This sandbox contains code that is for various reasons not ready to be
included in statsmodels proper. It contains modules from the old stats.models
code that have not been tested, verified and updated to the new statsmodels
structure: cox survival model, mixed effects model with repeated measures,
generalized additive model and the formula framework. The sandbox also
contains code that is currently being worked on until it fits the pattern
of statsmodels or is sufficiently tested.

All sandbox modules have to be explicitly imported to indicate that they are
not yet part of the core of statsmodels. The quality and testing of the
sandbox code varies widely.


Examples
--------

There are some examples in the `sandbox.examples` folder. Additional
examples are directly included in the modules and in subfolders of
the sandbox.


Module Reference
----------------


Time Series analysis :mod:`tsa`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this part we develop models and functions that will be useful for time
series analysis. Most of the models and function have been moved to
:mod:`statsmodels.tsa`.

Moving Window Statistics
""""""""""""""""""""""""

Most moving window statistics, like rolling mean, moments (up to 4th order), min,
max, mean, and variance, are covered by the functions for `Moving (rolling)
statistics/moments <https://pandas.pydata.org/pandas-docs/stable/computation.html#moving-rolling-statistics-moments>`_ in Pandas.

.. module:: statsmodels.sandbox.tsa
   :synopsis: Experimental time-series analysis models

.. currentmodule:: statsmodels.sandbox.tsa

.. autosummary::
   :toctree: generated/

   movstat.movorder
   movstat.movmean
   movstat.movvar
   movstat.movmoment


Regression and ANOVA
^^^^^^^^^^^^^^^^^^^^

.. module:: statsmodels.sandbox.regression.anova_nistcertified
   :synopsis: Experimental ANOVA estimator

.. currentmodule:: statsmodels.sandbox.regression.anova_nistcertified

The following two ANOVA functions are fully tested against the NIST test data
for balanced one-way ANOVA. ``anova_oneway`` follows the same pattern as the
oneway anova function in scipy.stats but with higher precision for badly
scaled problems. ``anova_ols`` produces the same results as the one way anova
however using the OLS model class. It also verifies against the NIST tests,
with some problems in the worst scaled cases. It shows how to do simple ANOVA
using statsmodels in three lines and is also best taken as a recipe.


.. autosummary::
   :toctree: generated/

   anova_oneway
   anova_ols


The following are helper functions for working with dummy variables and
generating ANOVA results with OLS. They are best considered as recipes since
they were written with a specific use in mind. These function will eventually
be rewritten or reorganized.

.. module:: statsmodels.sandbox.regression
   :synopsis: Experimental regression tools

.. currentmodule:: statsmodels.sandbox.regression

.. autosummary::
   :toctree: generated/

   try_ols_anova.data2dummy
   try_ols_anova.data2groupcont
   try_ols_anova.data2proddummy
   try_ols_anova.dropname
   try_ols_anova.form2design

The following are helper functions for group statistics where groups are
defined by a label array. The qualifying comments for the previous group
apply also to this group of functions.


.. autosummary::
   :toctree: generated/

   try_catdata.cat2dummy
   try_catdata.convertlabels
   try_catdata.groupsstats_1d
   try_catdata.groupsstats_dummy
   try_catdata.groupstatsbin
   try_catdata.labelmeanfilter
   try_catdata.labelmeanfilter_nd
   try_catdata.labelmeanfilter_str

Additional to these functions, sandbox regression still contains several
examples, that are illustrative of the use of the regression models of
statsmodels.



Systems of Regression Equations and Simultaneous Equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following are for fitting systems of equations models.  Though the returned
parameters have been verified as accurate, this code is still very
experimental, and the usage of the models will very likely change significantly
before they are added to the main codebase.

.. module:: statsmodels.sandbox.sysreg
   :synopsis: Experimental system regression models

.. currentmodule:: statsmodels.sandbox.sysreg

.. autosummary::
   :toctree: generated/

   SUR
   Sem2SLS

Miscellaneous
^^^^^^^^^^^^^
.. module:: statsmodels.sandbox.tools.tools_tsa
   :synopsis: Experimental tools for working with time-series

.. currentmodule:: statsmodels.sandbox.tools.tools_tsa


Descriptive Statistics Printing
"""""""""""""""""""""""""""""""

.. module:: statsmodels.sandbox
   :synopsis: Experimental tools that have not been fully vetted

.. currentmodule:: statsmodels.sandbox

.. autosummary::
   :toctree: generated/

   descstats.sign_test
   descstats.descstats




Original stats.models
^^^^^^^^^^^^^^^^^^^^^

None of these are fully working. The formula framework is used by cox and
mixed.

**Mixed Effects Model with Repeated Measures using an EM Algorithm**

:mod:`statsmodels.sandbox.mixed`


**Cox Proportional Hazards Model**

:mod:`statsmodels.sandbox.cox`

**Generalized Additive Models**

:mod:`statsmodels.sandbox.gam`

**Formula**

:mod:`statsmodels.sandbox.formula`


