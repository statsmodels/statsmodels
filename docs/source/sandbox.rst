.. currentmodule:: statsmodels.sandbox


.. _sandbox:


Sandbox
=======

Introduction
------------

This sandbox contains code that is for various resons not ready to be
included in statsmodels proper. It contains modules from the old stats.models
code that have not been tested, verified and updated to the new statsmodels
structure: cox survival model, mixed effects model with repeated measures,
generalized additive model and the formula framework. The sandbox also
contains code that is currently being worked on until it fits the pattern
of statsmodels or is sufficiently tested.

All sandbox modules have to be explicitly imported to indicate that they are
not yet part of the core of statsmodels. The quality and testing of the
sandbox code varies widely.


.. automodule:: statsmodels.sandbox


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
:mod:`statsmodels.tsa`. Currently, GARCH models remain in development stage in
`sandbox.tsa`.


.. currentmodule:: statsmodels.sandbox



Moving Window Statistics
""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   tsa.movmean
   tsa.movmoment
   tsa.movorder
   tsa.movstat
   tsa.movvar




Regression and ANOVA
^^^^^^^^^^^^^^^^^^^^

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

.. currentmodule:: statsmodels.sandbox.sysreg

.. autosummary::
   :toctree: generated/

   SUR
   Sem2SLS

Miscellaneous
^^^^^^^^^^^^^
 .. currentmodule:: statsmodels.sandbox.tools.tools_tsa


Tools for Time Series Analysis
""""""""""""""""""""""""""""""

nothing left in here


Tools: Principal Component Analysis
"""""""""""""""""""""""""""""""""""

.. currentmodule:: statsmodels.sandbox.tools.tools_pca

.. autosummary::
   :toctree: generated/

   pca
   pcasvd


Descriptive Statistics Printing
"""""""""""""""""""""""""""""""

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


