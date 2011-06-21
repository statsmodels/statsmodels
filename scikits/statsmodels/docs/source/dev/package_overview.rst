Package Overview
================

Mission Statement
~~~~~~~~~~~~~~~~~
Statsmodels is a python package for statistical modelling that is released under
the `BSD license <http://www.opensource.org/licenses/bsd-license.php>`_.

Design
~~~~~~
.. TODO perhaps a flow chart would be the best presentation here?

For the most part, statsmodels is an object-oriented library of statistical
models.  Our working definition of a statistical model is an object that has
both endogenous and exogenous data defined as well as a statistical
relationship.  In place of endogenous and exogenous one can often substitute
the terms left hand side (LHS) and right hand side (RHS), dependent and
independent variables, regressand and regressors, outcome and design, response
variable and explanatory variable, respectively.  The usage is quite often
domain specific; however, we have chosen to use `endog` and `exog` almost
exclusively, since the principal developers of statsmodels have a background
in econometrics, and this feels most natural.  This means that all of the
models are objects with `endog` and `exog` defined, though in some cases
`exog` is None for convenience (for instance, with an autoregressive process).
Each object also defines a `fit` (or similar) method that returns a
model-specific results object.  In addition there are some functions, e.g. for
statistical tests or convenience functions.

Code Organization
~~~~~~~~~~~~~~~~~

See the :ref:`Internal Class Guide <model>`.
