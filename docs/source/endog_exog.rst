.. _endog_exog:

``endog``, ``exog``, what's that?
=================================

statsmodels is using ``endog`` and ``exog`` as names for the data, the
observed variables that are used in an estimation problem. Other names that
are often used in different statistical packages or text books are, for
example,

===================== ======================
endog                 exog
===================== ======================
y                     x
y variable            x variable
left hand side (LHS)  right hand side (RHS)
dependent variable    independent variable
regressand            regressors
outcome               design
response variable     explanatory variable
===================== ======================

The usage is quite often domain and model specific; however, we have chosen
to use `endog` and `exog` almost exclusively. A mnemonic hint to keep the two
terms apart is that exogenous has an "x", as in x-variable, in its name.

`x` and `y` are one letter names that are sometimes used for temporary
variables and are not informative in itself. To avoid one letter names we
decided to use descriptive names and settled on ``endog`` and ``exog``.
Since this has been criticized, this might change in future.

Background
----------

Some informal definitions of the terms are

`endogenous`: caused by factors within the system

`exogenous`: caused by factors outside the system

*Endogenous variables designates variables in an economic/econometric model
that are explained, or predicted, by that model.*
http://stats.oecd.org/glossary/detail.asp?ID=794

*Exogenous variables designates variables that appear in an
economic/econometric model, but are not explained by that model (i.e. they are
taken as given by the model).*  http://stats.oecd.org/glossary/detail.asp?ID=890

In econometrics and statistics the terms are defined more formally, and
different definitions of exogeneity (weak, strong, strict) are used depending
on the model. The usage in statsmodels as variable names cannot always be
interpreted in a formal sense, but tries to follow the same principle.


In the simplest form, a model relates an observed variable, y, to another set
of variables, x, in some linear or nonlinear form ::

   y = f(x, beta) + noise
   y = x * beta + noise

However, to have a statistical model we need additional assumptions on the
properties of the explanatory variables, x, and the noise. One standard
assumption for many basic models is that x is not correlated with the noise.
In a more general definition, x being exogenous means that we do not have to
consider how the explanatory variables in x were generated, whether by design
or by random draws from some underlying distribution, when we want to estimate
the effect or impact that x has on y, or test a hypothesis about this effect.

In other words, y is *endogenous* to our model, x is *exogenous* to our model
for the estimation.

As an example, suppose you run an experiment and for the second session some
subjects are not available anymore.
Is the drop-out relevant for the conclusions you draw for the experiment?
In other words, can we treat the drop-out decision as exogenous for our
problem.

It is up to the user to know (or to consult a text book to find out) what the
underlying statistical assumptions for the models are. As an example, ``exog``
in ``OLS`` can have lagged dependent variables if the error or noise term is
independently distributed over time (or uncorrelated over time). However, if
the error terms are autocorrelated, then OLS does not have good statistical
properties (is inconsistent) and the correct model will be ARMAX.
``statsmodels`` has functions for regression diagnostics to test whether some of
the assumptions are justified or not.

